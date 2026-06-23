# FP8 KV Cache

## 结论

FP8 KV Cache 不会让 Qwen3.5 当前主 attention backend 从 Triton 切到 AITER，也不会直接触发 AITER GEMM。它影响的是 KV cache 的存储 dtype、cache 写入、attention kernel 读取 K/V 时的 scale/dequant，以及 attention 内部 QK/PV 计算的数据流。

这条线仍然重要：AITER 官方增强项主要是 GEMM、MoE、MLA、MHA；当前 Qwen3.5 路径的 MLA/MHA 不命中，MoE 也不是主要入口，AITER 可能只覆盖 GEMM/边缘 op。FP8 KV Cache 则给 decode attention 的 KV 访存压力提供独立优化入口。两者组合可能在 decode 端叠加收益：

```text
AITER GEMM/边缘 op 覆盖
+
FP8 KV Cache 降低 attention 读 KV cache 的带宽/容量压力
=
decode 端可能有可测收益
```

但收益不能归因为 AITER attention。

## 源码依据

`--kv-cache-dtype fp8` 进入 cache config，而不是 linear quant config。`vllm/config/cache.py` 对 FP8 KV 的说明是减少 KV cache 显存占用并可能提升性能，同时有精度风险：

```python
if cache_dtype.startswith("fp8"):
    logger.info(
        "Using fp8 data type to store kv cache. It reduces the GPU "
        "memory footprint and boosts the performance. "
        "Meanwhile, it may cause accuracy drop without a proper "
        "scaling factor."
    )
```

普通 BF16 权重的 Linear/GEMM 路径仍在 `vllm/model_executor/layers/linear.py`：

```python
return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
```

FP8 weight quantized linear 才会走 `vllm/model_executor/layers/quantization/fp8.py` 中的 FP8 linear method，并检查 AITER linear 是否启用：

```python
self.use_aiter_and_is_supported = rocm_aiter_ops.is_linear_fp8_enabled()
```

因此，FP8 KV Cache 不会自动把 BF16 权重 GEMM 改成 AITER FP8 GEMM。AITER GEMM 和 FP8 KV Cache 是两条独立收益来源。

## 当前 Triton FP8 KV 路径

`vllm/v1/attention/ops/triton_decode_attention.py` 中 decode attention 读取 FP8 K/V 后做 scale/dequant，再进入 attention 内部点积：

```python
k = tl.load(K_Buffer + offs_buf_k, ...)
if k.dtype.is_fp8():
    k = (k.to(tl.float32) * ks).to(q.dtype)
qk = tl.dot(q, k.to(q.dtype))
```

V 也是同样模式：

```python
v = tl.load(V_Buffer + offs_buf_v, ...)
if v.dtype.is_fp8():
    v = (v.to(tl.float32) * vs).to(q.dtype)
acc += tl.dot(p.to(v.dtype), v)
```

`vllm/v1/attention/ops/prefix_prefill.py` 也会在 FP8 KV 时把 cache view 成目标 FP8 dtype，并向 Triton kernel 传入 `k_scale` / `v_scale`：

```python
if "fp8" in kv_cache_dtype:
    k_cache = k_cache.view(target_dtype)
    v_cache = v_cache.view(target_dtype)
```

这些 `tl.dot` 是 attention kernel 内部的 QK/PV 计算，不是 vLLM Linear layer 的 GEMM dispatch，也不是 `_aiter_ops.gemm_a8w8` 等 AITER GEMM wrapper。

## FP8 load/dequant 方向

这反而提示了一个更值得披露和研究的点：FP8 KV Cache 暴露了 attention kernel 内部的 FP8 KV load/dequant 热点。如果 Triton 当前只是普通：

```text
tl.load(fp8) -> fp32 scale/dequant -> cast -> tl.dot
```

那么可优化的是 load/dequant 这段数据通路（FP8 load/convert 指令是否被充分利用），不是
MFMA 计算能力——Triton 的 `tl.dot` 已经发射 MFMA，与 DUMMA 共享同一硬件单元，不存在
"Triton 不会用矩阵核心"的问题。

本卡约束：DUMMA 的 FP8 MMA 仅 gfx938，本卡 gfx936 不可用。所以 FP8 GEMM 对照应走
hipBLASLt（有完整 FP8/MXFP8 scaling 接口），FP8 load/convert 探测应查 DTK 的 `fp8.h`
device header 或 LLVM intrinsic（如 `__builtin_amdgcn_cvt_*_fp8`），而不是 DUMMA。

优先级不是重写整个 attention，而是先回答三个问题：

1. 当前 Triton FP8 KV kernel 生成的 IR/ISA 是否已经利用专门的 FP8 load/convert 指令。
2. hipBLASLt FP8 GEMM 或独立 FP8 load/convert microkernel 是否比 Triton 当前路径更快。
3. FP8 KV Cache 在 16-32K 上是否能带来真实吞吐/TPOT/TTFT 收益且精度可接受。

如果 1 和 2 证明存在低层空间，再考虑接入 vLLM。

## 实验矩阵

先不要把 FP8 KV 和 AITER 混在一起归因，按以下顺序测：

```text
A. clean + auto KV
   已有官方口径 baseline。

B. aiter + auto KV
   验证 AITER GEMM/边缘覆盖单独收益。

C. clean + fp8 KV
   验证 FP8 KV 本身是否改善 decode/长尾，并检查 accuracy。

D. aiter + fp8 KV
   只有 C 有收益且精度可接受时再跑。
   用来验证 GEMM/边缘覆盖和 FP8 KV 是否能叠加。
```

关键指标：

```text
output_throughput
total_token_throughput
mean_ttft_ms / p99_ttft_ms
mean_tpot_ms / p99_tpot_ms
duration
```

判断建议：

- `output_throughput` 小于 `+1%` 基本按噪声处理。
- 如果 TPOT 降而 TTFT 基本不变，更像 decode KV 访存收益。
- 如果 TTFT 也降，需要结合请求级明细和 profile 看 prefill/long-tail 是否受益。
- 如果 throughput 有收益但 accuracy 下降，这条不能作为提交方向。

## 精度和合规风险

FP8 KV Cache 是运行时 KV cache 低精度路径，不是权重量化，但仍然改变了 attention 计算中的历史 K/V 表示。它必须通过 accuracy gate，尤其是长上下文检索/聚合任务：

```text
retrieval_multi_point
aggregation_keyword_aggregation
gov_report
hotpotqa
```

快速 sanity 可先跑：

```bash
bash ./run_throughput.sh 16-32K
bash ./run_accuracy.sh retrieval_multi_point
bash ./run_accuracy.sh aggregation_keyword_aggregation
```

完整结论必须跑官方 `run_accuracy.sh all`。

## 可能的接入方式

如果 FP8 load/convert microbench 证明有明显收益，接入方式按风险从低到高：

1. 自定义 HIP op 做 gather/dequant 到临时 buffer，再复用现有 Triton attention。
   - 工程风险低。
   - 可能被额外写临时 buffer 抵消收益。

2. 替换 `triton_decode_attention.py` / `prefix_prefill.py` 中 FP8 KV load/dequant 局部。
   - 需要 Triton inline asm 或等价能力。
   - 能保留现有 attention 结构。

3. 写完整 DCU/HIP attention 子路径。
   - 潜在收益最大。
   - 工程和精度风险最高。

当前建议先做 micro probe，不直接改 vLLM 主路径。
