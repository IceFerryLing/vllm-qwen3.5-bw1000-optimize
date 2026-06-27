# Unified attention page-local fastpath

## 硬件依据

`pra26/doc-dcu.md` 里对 DCU kernel 的建议有两点直接适用：

- 全局内存访问要尽量连续、对齐，避免大 stride、随机 gather 和非合并访问。
- vLLM 类 workload 里的 paged KV gather、block table indirection、mask 边界和不同请求长度
  会破坏规整访存，优化前应先确认瓶颈来自 global load、LDS、计算还是 launch/API。

在 Qwen3.5 full attention 的 Triton 2D 路径里，KV cache block size 记录为 `784`。
常见 `TILE_SIZE=32` 时，大多数 KV tile 都完全落在同一个 KV page 内，只有接近 page 边界的
少量 tile 会跨 page。

原路径每个 tile 都对 `seq_offset` 向量做：

```text
seq_offset // BLOCK_SIZE
seq_offset % BLOCK_SIZE
block_tables[seq_offset // BLOCK_SIZE]
```

这会为同一个 page 内的 32 个 token 重复计算 page index、重复加载相同 block table 项，
并在热循环里保留向量除法/取模和 paged gather 形态。

## 代码改动

`vllm/v1/attention/ops/triton_unified_attention.py` 增加了默认关闭的：

```bash
QWEN_UA_PAGE_LOCAL_FASTPATH=1
```

该开关只作用于 `kernel_unified_attention_2d`。当一个 tile 完全落在同一个 KV page 内时：

- 只加载一次 `block_tables[block_table_idx]`；
- 用标量 `block_offset + offs_t` 生成 page 内 offset；
- 避免对整个 `seq_offset` 向量重复做 `// BLOCK_SIZE` 和 `% BLOCK_SIZE`；
- 跨 page 的边界 tile 仍回退到原始向量路径。

默认不开启时，行为保持 baseline 不变。

## 风险与验证

这是地址计算 fastpath，不改变 attention 数学、mask、scale、KV cache 内容或输出口径。
但它会改变 Triton JIT 生成的 IR/ISA，必须在 DCU 容器内验证：

```bash
export QWEN_UA_PAGE_LOCAL_FASTPATH=1
export QWEN_UA_PROFILE_LOG=1
```

验证指标：

- `kernel_unified_attention_2d` kernel time；
- L2 read bytes / request 数；
- shared memory bank conflict 和 LDS/L1 stall，确认没有副作用；
- VGPR / SGPR / scratch，确认没有 spill；
- throughput output tokens/s；
- TTFT P99 / TPOT P99；
- accuracy gate。

若 Triton JIT 不接受该分支形态，或资源占用上升导致 kernel time 变慢，应保持默认关闭并回退。
