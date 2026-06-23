# flash_fwd_kernel 第二条 attention 路径

## 结论

- Qwen3.5 的 full attention 在 DCU 上**不止 `kernel_unified_attention_2d` 一条路径**。
  官方 16-32K 多 prompt serving 的 hipprof stats 里，出现一个独立的 cutlass FlashAttention
  kernel `flash_fwd_kernel_16x64_prefetch`，它与 `unified_attention_2d` 并存，是另一条
  attention 实现。
- 调 BLOCK_M 只优化了 `unified_attention_2d`（paged KV 路径），**动不到这条 flash
  attention 路径**。attention 总体优化必须两条路径都覆盖。
- 在确认它的来源（哪个 backend / 哪个 .so / Qwen3.5 哪一层）前，不写任何 patch。

## 证据

来源：`misc/profile/profile_runs/hipprof_stats_bm64_16_32k_official1_20260623_185033/
hipprof/hip_stats.hipkernel.csv`（BLOCK_M=64，16-32K official 多 prompt serving）。

```text
#2 kernel_unified_attention_2d        calls=480   total=12.30s  avg=25.6ms  pct=19.5%
#5 flash_fwd_kernel_16x64_prefetch<   calls=27    total=3.86s   avg=143ms   pct=6.1%
      Flash_fwd_kernel_16x64_prefetch_traits_dim96<96,128,64,4,bfloat16_t,3,96,...>>
```

对比单请求 12K prefill slice（同 BLOCK_M=64）：

```text
单请求 12K prefill:   unified_attention_2d 9.2%   flash_fwd_kernel 未出现
官方 16-32K 多 prompt: unified_attention_2d 19.5%  flash_fwd_kernel 6.1%
```

观察：

- `flash_fwd_kernel_16x64_prefetch` 在单请求 prefill slice 里**没出现**，只在官方多 prompt
  长上下文跑里出现。说明它对应某种特定 shape 或请求模式（可能是更长的 KV、特定 head_dim
  投影，或 prefill 全长度 attention）。
- 27 calls 但 3.86s，**单次 143ms**，是 `unified_attention_2d` 单次（25.6ms）的 5.6 倍。
  少量调用就吃 6.1%，单次极重。
- 模板参数 `dim96<96,128,64,4,...>`：head_dim 或投影维度 96，tile 128×64，BF16，3 个
  attention 维度参数。这不是 Qwen3.5 full attention 的 head_dim=256，所以**它很可能是
  Qwen3.5 里某个非主 full-attention 的 attention 子结构**（如 ViT/multimodal encoder
  attention、或 GDN 内部的某次 dot），而非主 LLM full attention。需确认。

## 来源归属（待确认）

本地 vLLM 源码 grep `flash_fwd_kernel` / `Flash_fwd_kernel_traits` **零命中**——这个 kernel
不在本仓库源码里，来自外部预编译库（composable_kernel / CK flash attention，或 AMD
flash-attention 包），以 .so 形式注入。本地只有 backend wrapper：

```text
vllm/v1/attention/backends/rocm_aiter_fa.py
vllm/v1/attention/backends/flash_attn.py
vllm/v1/attention/backends/rocm_attn.py
```

Qwen3.5 服务日志已确认主 attention backend 是 `TRITON_ATTN`（`unified_attention_2d`）。
所以 `flash_fwd_kernel` 不是主 backend 选的，而是**某条子路径单独调用了外部 flash
attention 库**。候选：

```text
1. multimodal encoder（ViT）attention：Qwen3.5 是多模态，ViT encoder 可能用 cutlass
   flash attention。日志里见过 "Using Torch SDPA backend for ViT model"，但 SDPA 在 ROCm
   上可能落到 flash_fwd_kernel。dim96 可能对不上 ViT（ViT 通常 head_dim 64/128），存疑。
2. GDN / linear attention 内部的某次标准 attention 子步骤。
3. 某个 profiling/warmup 阶段的固定 shape attention。
```

## 需要回答的问题

1. `flash_fwd_kernel_16x64_prefetch` 来自哪个 .so？用 hipprof `--kernel-stack` 取主机端
   调用栈，定位它从 vLLM 哪个 Python/C++ 路径发起。
2. 它对应 Qwen3.5 的哪一层 / 哪种请求？27 calls 是否对应固定的 27 个 attention 层，还是
   随 prompt 数变化？多跑一组不同 prompt 数的 official，看 calls 是否线性。
3. `dim96` 的 96 到底是 head_dim 还是投影维度？查 Qwen3.5 配置里是否有 head_dim=96 或
   96 维中间投影的 attention 子结构。
4. 它和 `unified_attention_2d` 是互斥分流还是叠加？同一请求里两者都出现，还是按 shape
   分流？看 per-call 时间戳和 max_seqlen。
5. 单次 143ms 为什么这么重？是 shape 大（长 KV × 多 head）还是 kernel 本身 tile/config
   不优？对比 `unified_attention_2d` 在等价 shape 下的耗时。

## 验证方法

先不改代码，先定位：

```text
A. hipprof --hip-trace --kernel-stack 对 flash_fwd_kernel 取主机端调用栈，定位发起路径。
   这是零代码改动的直接手段。
B. 多 prompt 数 official 跑（如 prompt=10/50/100），看 flash_fwd_kernel calls 是否随
   prompt 数线性 → 判断是 per-layer 还是 per-request。
C. 对照 decode-heavy slice（4K-8K input, 512-1024 output）的 stats，看 flash_fwd_kernel
   是否主要出现在 prefill（若是 decode 也大量出现，则不是 ViT 这种一次性 encoder）。
D. 查 Qwen3.5 config.json / modeling 代码里 head_dim、num_attention_heads、是否有 96 维
   中间结构，匹配 dim96。
```

## 风险边界

- 这个 kernel 来自外部预编译库，**本仓库改不了它的实现**。能做的只有：
  (a) 改 vLLM 侧的 dispatch，让它走 `unified_attention_2d`（若数学等价且更快）；
  (b) 或换外部库版本（属环境/构建改动，需重新构建 wheel）。
- 任何 dispatch 改动必须保持数学等价，不改 mask/scale/输出口径/模型语义。
- 在 A 步定位来源前，不写 patch，不假设它是 ViT 还是 GDN。

## 当前状态

```text
status: source_unconfirmed
owner: unassigned
next: hipprof --kernel-stack 定位 flash_fwd_kernel 主机端来源；多 prompt 数对照看 calls 线性
```
