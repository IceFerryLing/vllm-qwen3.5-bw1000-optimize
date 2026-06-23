# AITER unified attention

## 结论

- AITER 的 `unified_attention` 实现层面做了跨 page/block 的 online softmax。
- 它不是把每个 page 当独立 chunk 分别 softmax 后直接拼接或相加。
- 但该路径仍必须用输出一致性和 profile 双验证；profile 只能证明快，不能证明语义正确。
- 对 Qwen3.5 来说，AITER 的两条近期路径在本地实测下都**没有明显优势**（详见下文各节）：
  - GDN **decode** fast path：本地 decode GDN 已被 FLA packed recurrent 优化到占官方
    16-32K 负载的 **0.28%**，AITER decode 要砍的 launch overhead 已被 FLA packed 砍过，
    边际收益 << 上游宣称的 5-8%（那是相对 Qwen3-Next generic 路径的数）。
  - GDN **prefill**：是真热点（prefill-heavy 单请求下 GDN chunk 系列合计 ~44%），但 AITER
    prefill 上游只是研究方向（issue #2354），无可直接回迁的实现证据。
  - AITER **GEMM**：官方负载 GEMM 已是 autotune 的 rocBLAS/Tensile（BF16 dense，非
    FP8/MoE），AITER GEMM 无结构性优势。
- 因此 AITER 方向当前**不作为投入优先项**；若要动 GDN，应转向 prefill 拆解，不是 decode。


## 代码证据

比赛容器内实现文件：

```text
/usr/local/lib/python3.10/dist-packages/aiter/ops/triton/unified_attention.py
```

vLLM wrapper `RocmAiterUnifiedAttentionImpl` 传入的是 paged KV cache、`block_table` 和
`seqused_k`：

```python
self.unified_attention(
    q=query[:num_actual_tokens],
    k=key_cache,
    v=value_cache,
    out=output[:num_actual_tokens],
    cu_seqlens_q=cu_seqlens_q,
    seqused_k=seqused_k,
    block_table=block_table,
    ...
)
```

AITER 2D kernel 在循环外初始化全局 softmax 状态：

```python
M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
```

随后按 `block_table` 遍历所有 KV block/page：

```python
num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)
for j in range(0, num_blocks):
    physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)
    ...
```

每个 block 内更新同一组 online softmax 状态：

```python
m_j = tl.maximum(M, tl.max(S, axis=1))
P = tl.exp(S - m_j[:, None])
l_j = tl.sum(P, axis=1)
alpha = tl.exp(M - m_j)

acc = acc * alpha[:, None]
L = L * alpha + l_j
M = m_j
acc += tl.dot(P.to(V.dtype), V)
```

遍历完所有 block/page 后才统一归一化：

```python
acc = acc / L[:, None]
```

这说明 2D 路径是跨 page/block 的同一个 softmax。

3D 路径先在每个 segment 内做 online softmax，并保存 segment output、max、expsum：

```python
tl.store(segm_output_ptr + ..., acc)
tl.store(segm_max_ptr + ..., M)
tl.store(segm_expsum_ptr + ..., L)
```

`reduce_segments` 再用 LSE 形式合并 segment：

```python
overall_max = tl.max(segm_max)

segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
overall_expsum = tl.sum(segm_expsum)

segm_output *= tl.exp(segm_max - overall_max)[:, None]
acc_sum = tl.sum(segm_output, axis=0)
acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)
```

因此 3D 路径也不是错误的独立 chunk softmax。

## `%16` 路径

外部 KV cache page/block size 可以显式设为 `16`。AITER 内部会读取：

```python
cache_block_size = v.shape[1]
block_size = cache_block_size
```

如果内部 tile 太大，会选择一个能整除 `cache_block_size` 的 `BLOCK_SIZE`，再通过
`start_n // CACHE_BLOCK_SIZE` 和 `% CACHE_BLOCK_SIZE` 映射回 paged KV cache：

```python
physical_block_idx = tl.load(
    block_tables_ptr + block_table_offset + (start_n // CACHE_BLOCK_SIZE)
)
...
((start_n + offs_n) % CACHE_BLOCK_SIZE)
```

这条映射仍是跨 cache page 的 attention，不是 page 内独立 attention。

## 验证要求

C 组可以作为候选路径测试，但不能只看吞吐：

```text
C: AITER unified attention + --block-size 16
```

需要同时确认：

- 输出一致性：短样本、跨 page 长样本、prefill/decode 都要和基线路径比对。
- profile：确认实际命中 `kernel_unified_attention_2d` 还是
  `kernel_unified_attention_3d` + `reduce_segments`。
- 服务日志：确认 backend 选择为 `ROCM_AITER_UNIFIED_ATTN`，而不是回退到其他 attention。

之前 AITER unified 启动失败主要是未显式指定 block size 时被自动改成 `64`，触发 hybrid
page-size unification 失败。下一轮测试应显式使用 `--block-size 16`。

## wrapper 兼容性 gate

技术方案 PDF 允许结合 Qwen3.5-27B、长上下文和 DCU 特性做 KV cache、Attention
kernel、执行路径等优化，但禁止修改模型语义、跳层、token pruning、投机解码、测试集
特化 if-else 等行为。因此 AITER wrapper 不应写 `if Qwen3.5` 这种模型名特判。

当前容器里的 AITER `unified_attention` 签名不接受 `sinks` 和 `output_scale`。C3
实测中 vLLM 选择了 `ROCM_AITER_UNIFIED_ATTN`，并且 `has_sink=False`，但 wrapper
仍无条件传入：

```python
sinks=self.sinks,
output_scale=output_scale,
```

导致启动阶段报错：

```text
TypeError: unified_attention() got an unexpected keyword argument 'sinks'
```

正确修法是按 backend capability 和实际语义需求 gate：

```python
if self.sinks is not None and not self.supports_sinks:
    raise NotImplementedError(
        "AITER unified attention does not support attention sinks"
    )

if output_scale is not None and not self.supports_output_scale:
    raise NotImplementedError(
        "AITER unified attention does not support output_scale"
    )
```

然后只在后端签名支持时传对应 keyword。`supports_sinks` 和
`supports_output_scale` 应在 `__init__` 中用 `inspect.signature()` 计算一次，不要放在
`forward()` 热路径里重复反射。

这个 gate 的语义是：当前请求不需要 `sinks` / `output_scale` 时跳过不支持的可选参数；
如果模型或路径实际需要这些语义而 AITER 不支持，则显式失败，避免静默算错。这样是
backend 适配，不是模型特化。

## Qwen3.5 GDN fast path

本地实测修正了这条线的优先级：**AITER GDN decode fast path 在当前负载下没有明显优势**，
真热点在 prefill 不在 decode。

实测证据（`misc/profile/profile_runs/hipprof_stats_bm64_16_32k_official1_20260623_185033`，
官方 16-32K 多 prompt serving，BLOCK_M=64）：

```text
decode GDN: fused_recurrent_gated_delta_rule_packed_decode_kernel
            calls=4224 total=176ms avg=41.8µs pct=0.28%
prefill GDN (chunk 系列合计):
  chunk_fwd_kernel_o                        6.6%
  chunk_scaled_dot_kkt_fwd_kernel           2.2%
  chunk_gated_delta_rule_fwd                1.9%
  merge_16x16_to_64x64_inverse              1.7%
  recompute_w_u_fwd                         1.5%
  (prefill-heavy 单请求下 GDN 合计可达 ~44%)
```

三层理由判定 AITER GDN decode 无明显优势：

```text
1. decode GDN 本身只占 0.28%（176ms/63.1s）。就算 AITER 优化到 0，端到端也只省 0.28%。
   上游宣称的 5-8% 是相对 Qwen3-Next TPOT 的数，那个基数里 GDN decode 占比远高于 0.28%。
2. 当前 decode 已走 FLA packed recurrent（fused_recurrent_gated_delta_rule_packed_decode_kernel，
   单 launch fused，grid=(NV, B*HV)），AITER 要砍的“每层 20-26us launch overhead”已被
   FLA packed 砍过。AITER 相对 FLA packed 的边际收益 << AITER 相对 generic 的收益。
3. GDN 真热点在 prefill（prefill-heavy 44%），AITER GDN Decode 不碰 prefill。
```

接入成本也高：本地无 AITER GDN 接入（grep 零命中），要从上游 `gdn_linear_attn.py` /
`qwen_gdn_linear_attn.py` 最小回迁，处理 Qwen3.5 flat-layout（`#42880` guard）、`#3251`
layout 参数、sinks/output_scale 兼容性 gate。**高接入成本换 <1% 收益，不投入。**

若一定要做 GDN 方向，应转向 **prefill**：先用 profile 拆 `ChunkGatedDeltaRule` 内部
（`chunk_fwd_kernel_o` / `chunk_scaled_dot_kkt_fwd` / `chunk_gated_delta_rule` 哪个是主成本），
再评估是否有数学等价的 tile/fusion 优化空间。AITER prefill 上游只是研究方向，不可直接回迁。

本地源码现状：

- 当前本地 vLLM 还没有最新上游的
  `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` 拆分，也没有
  `rocm_aiter_ops.are_gdn_triton_kernels_available()` /
  `gdn_aiter_fused_*` 接入口。
- Qwen3.5 的 GDN 实现在 `vllm/model_executor/models/qwen3_5.py`，继承
  `Qwen3NextGatedDeltaNet`，并把投影布局改成 flat：
  `[q_all|k_all|v_all|z_all]` 和 `[b_all|a_all]`。
- 当前 prefill 走 `ChunkGatedDeltaRule`，后端是 FlashInfer 或 Triton/FLA；在 DCU/ROCm
  上 FlashInfer 条件不满足，实际应按 Triton/FLA 路径看。
- 当前 decode 已有本地 FLA packed recurrent decode fast path，由
  `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1` 默认启用；这不是 AITER，但可作为 A/B 对照。

上游证据链：

- ROCm/aiter#2423：AITER 增加 Qwen3-Next GDN decode Triton kernels，包括
  `fused_reshape_causal_conv1d_update_single_token`、
  `fused_rearrange_sigmoid_gated_delta_rule` 和 gated FP8 quant 等。该 PR 主要服务
  Qwen3-Next 的 interleaved layout。
- vllm-project/vllm#40711：vLLM 接入 AITER GDN decode fast path。PR 描述里给出的
  目标是减少每层约 20-26us launch overhead，整体 TPOT/throughput 有 5-8% 级别收益；
  但当时验证重点仍是 Qwen3-Next。
- vllm-project/vllm#42880：Qwen3.5 被临时 guard 回 generic GDN path。原因是
  Qwen3.5 使用 non-interleaved / flat projection layout：`[q_all|k_all|v_all|z_all]`
  和 `[b_all|a_all]`。直接把这种 packed tensor 送进 Qwen3-Next 的 fused AITER
  kernel 会读错列，GSM8K 几乎崩掉。这个 PR 只能证明 layout-aware GDN decode 的必要性，
  不能单独支撑 AITER GDN prefill 优化。
- ROCm/aiter#3251：AITER 后续给
  `fused_reshape_causal_conv1d_update_single_token` 增加
  `gqa_interleaved_layout` 或等价 layout 参数，明确支持 Qwen3.5 flat layout。PR 里提到
  vLLM 侧去掉 Qwen3.5 guard 并传 layout 后，GSM8K 与 generic path 基本一致。
- ROCm/aiter#2354：AITER 仓库还有 Qwen3.5 GDN prefill 相关 issue，说明上游也关注这条线；
  但它不是可直接回迁的实现证据，不能替代本地 profile 和 correctness gate。
- vllm-project/vllm#44700：GDN mixed prefill+decode split 有明显性能收益记录，但也有
  长上下文 mixed batch 下越界/page fault 的后续风险信号。它可以作为研究方向，不应
  直接无 gate 搬进比赛路径。

因此下一次上卡优先验证 C 组，但 C 组的定义应收窄为 decode-first：

```text
C: AITER GDN flat-layout decode fast path
```

最小验证步骤：

```text
1. 检查已安装 AITER 是否包含 ROCm/aiter#3251：
   fused_reshape_causal_conv1d_update_single_token 是否有 gqa_interleaved_layout /
   qkvz_layout / flat-layout 等价参数。
2. 检查本地 vLLM 是否已经有 AITER GDN 接入口；当前源码大概率没有，需要从上游
   gdn_linear_attn / qwen_gdn_linear_attn 相关改动中最小回迁。
3. 只在 decode-only、无 spec、无 prefill、AITER GDN kernel importable 时启用；
   Qwen3.5 flat layout 要显式传给 AITER，不要写模型名特判。
4. 先跑短 accuracy sanity 和长上下文 sanity，再看 throughput。
5. profile 确认实际命中 AITER GDN fused kernel，而不是仍在本地 FLA/generic GDN path。
6. prefill 暂不改实现，只 profile；确认它是主热点后再评估 AITER prefill。
```

预期命中的关键符号/路径：

```text
aiter.ops.triton.causal_conv1d_update_single_token
fused_reshape_causal_conv1d_update_single_token(..., gqa_interleaved_layout=False)
fused_rearrange_sigmoid_gated_delta_rule
vllm/model_executor/layers/mamba/gdn_linear_attn.py
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
```

风险边界：

- 这条线是 AITER GDN，不是 `_rocm_C`，不要写成 native ROCm C++ 结果。
- 它不改变模型语义，只能做 layout-aware backend dispatch 和数学等价 kernel fusion。
- decode 收益不能自动外推到 prefill；prefill 要单独 profile、单独 accuracy gate。
- 任何 mixed prefill+decode split、in-place SSM state、unchecked index 路径都必须先过
  accuracy 和长上下文稳定性 sanity。
- 如果只看到吞吐提升但 accuracy 或输出稳定性不过，不能作为有效优化。

## Full attention 后续方向

Qwen3.5 的 full attention 仍可能是 16K-32K 的大热点，但当前社区证据不支持直接把
AITER full attention 当成稳定主线：

- Qwen3.5 的 ROCm attention/cache update 有非标准 block/page size，历史上需要走
  Triton fallback 才能避免 broken output。
- AITER attention 在 Qwen3.5 上有 corrupted responses 相关记录，必须严控 correctness。
- `ROCM_AITER_UNIFIED_ATTN` 可以作为小 sanity/profile 组，但不应直接跑长 baseline。

更合理的 full attention 路线是先围绕 vLLM 当前实际命中的 Triton fallback 做 profile，
再考虑 custom Triton kernel 或针对热点 shape 的 Triton patch。候选动作：

```text
1. 确认 16K-32K full attention 热点 kernel 名称、shape、耗时占比。
2. 对比 clean Triton fallback 与 AITER unified/FA 的 correctness 和 backend log。
3. 如果 AITER full attention 不稳定或不命中，转向 Triton custom kernel/tuning。
4. custom kernel 只能做数学等价优化，不改变 block table、KV cache 语义或输出口径。
```
