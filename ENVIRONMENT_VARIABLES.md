# 自定义环境变量说明

本参赛方案新增了下面的环境变量：

```bash
export TRITON_UNIFIED_ATTN_BLOCK_M=64
export VLLM_PREFILL_ATTN_TILE_SIZE=16
export VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS=256
export VLLM_TRITON_FA_PREFILL=1
export VLLM_TRITON_FA_PREFILL_MIRROR_CAPACITY=32768
export VLLM_TRITON_FA_PREFILL_FULL_KV_THRESHOLD=30720
export VLLM_ROCM_STRIDED_GEMV=True
```

目前的代码会默认取这些值。

## 变量作用与配置原因

### `TRITON_UNIFIED_ATTN_BLOCK_M=64`

- **作用：** 设置 Triton Unified Attention kernel 在 query/head 维度上的 `BLOCK_M`。
- **配置原因：** 在 Unified Attention 路径中，GQA 的 query heads 会按 KV head 分组映射到 `BLOCK_M`，该参数决定一个 Triton program 同时展开的 query/head 行数。调高 BLOCK_M 能在 query 较长时让一个 program 覆盖更多 query 行，减少 q-block program 数，并在更多行之间摊薄 paged-KV 地址计算和 block-table 查询等逐开销，而代价是更大的 query/FP32 accumulator、较高的寄存器压力以及序列尾部更多的掩码计算。经测试，将 `64` 作为 BW1000/gfx936 上纯 Triton Unified Attention 路径的默认值。

### `VLLM_PREFILL_ATTN_TILE_SIZE=16`

- **作用：** 设置 Triton Unified Attention 在 prefill 阶段使用的 `TILE_SIZE`。
- **配置原因：** Qwen3.5-27B full-attention 的 head dimension 为 256，Unified Attention 需要在片上保留 query、K/V tile、softmax 统计量和 FP32 accumulator。在 BW1000/gfx936 上，`TILE_SIZE=32` 会提高每个 program 的 LDS 和寄存器占用，实际只能保持较低的并发 wave 数，难以隐藏 paged KV 的 HBM 访存延迟。将 tile 降为 `16` 后，单个 program 的资源需求降低，可同时存在更多 wave，对 `HEAD_SIZE=256` 的路径整体更快。

### `VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS=256`

- **作用：** 设置 3D decode Unified Attention kernel 中并行 softmax 的分段数，并决定相应中间 workspace 的形状。
- **配置原因：** Qwen3.5-27B 每层的 KV heads 数较少，每个 KV head 只由少量 program 串行扫描 4K–32K paged KV时，难以在 BW1000 的大量计算单元上展开足够的并行工作，也难以充分利用 HBM 带宽。3D kernel 将序列沿 KV 长度分段，分别计算局部 output/max/expsum，再由 reduce kernel 合并；经实测，`256`能很好平衡 reduce 成本和 workspace 开销，通过 grid 的 segment 维增加可调度 program 数，以隐藏长上下文下的 HBM 访存延迟。

### `VLLM_TRITON_FA_PREFILL=1`

- **作用：** 启用 ROCm 上的 FlashAttention prefill 快速路径。
- **配置原因：**针对长 query 输入，若直接使用 vLLM Unified Attention 在不连续的 paged KV 缓存上计算，会因频繁解析 block-table 和非连续寻址产生高昂的硬件开销。该方案在 prefill 阶段选择先将碎片化的 K/V 缓存整理为连续的内存视图，进而调用 FlashAttention 以充分利用 AMD GPU（gfx936/BW1000）的 MFMA 计算单元与 HBM 合并访存性能。

### `VLLM_TRITON_FA_PREFILL_MIRROR_CAPACITY=32768`

- **作用：** 设置单请求 FlashAttention prefill 路径中，每层连续 K/V 镜像可保存的最大 token 数。
- **配置原因：** vLLM 的 chunked prefill 会分多次处理一个长 prompt。如果每个 chunk 都从 paged KV cache 重新 gather 全部历史 K/V，已处理的前缀会被反复遍历，block-table 查询和 HBM 读取量会随 chunk 数累积。Qwen3.5-27B 的 full-attention GQA 只有少量 KV heads，连续镜像的显存代价可控。`32768` 与设置的最大模型上下文一致，可在整个评测长度内将新 K/V 顺序追加到镜像。
- **PS:** 当普通部署情况下， `max_model_len` 设为大于 32768 且实际序列超出镜像容量时，代码会在写入前检查 `seq_len > capacity`，禁用该请求的镜像快速路径，并回退到 paged KV gather、FlashAttention 和 LSE merge 的通用路径。

### `VLLM_TRITON_FA_PREFILL_FULL_KV_THRESHOLD=30720`

- **作用：** 设置单请求连续KV镜像路径使用“一次 full-KV causal FlashAttention”的最大序列长度。超过该阈值后，改用当前 chunk 的 causal FlashAttention、历史 KV 的 non-causal FlashAttention 及 LSE merge。
- **配置原因：** 在单请求 chunked prefill 中，有两种数学等价的组织方式：1.对“历史+当前 chunk”执行一次带 offset 的 causal FlashAttention，2.对历史执行 non-causal attention、对当前 chunk 执行 causal attention 后再合并 LSE--前者只有一次 kernel 调度，没有局部 output/LSE 的额外 HBM 写回与 merge，在 30K 以内更能发挥 BW1000 的计算吞吐。接近 32K 时，full-KV causal kernel 的 K/V 工作集、causal mask 范围和单次 kernel 持续时间进一步增大。经部分实测，`30720` 是在节省一次 launch/merge 与避免超长 full-KV 路径负收益之间的一个转换点。

### `VLLM_ROCM_STRIDED_GEMV=True`

- **作用：** 对符合 shape、dtype 的 ROCm decode `N=1` Linear，启用 strided-K GEMV kernel。
- **配置原因：** 并发为 1 时，decode Linear 是主要受 HBM 带宽限制的 GEMV。原 LLMM1 的线程数会随 K 增长，无法处理 MLP down projection 的 `K=17408`；strided-K 使用固定线程数循环遍历 K，既支持该形状，又通过连续读取权重提高了 BW1000 的 HBM 利用率，提高了 GEMV 的处理速度。
