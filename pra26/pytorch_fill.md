# PyTorch fill / metadata 开销待查

## 结论

- `FillFunctor<int>` 是独立于 prefill 计算 kernel 的 PyTorch 侧 fill / metadata / buffer
  初始化开销，不能再归到 prefill 本体里。
- 12K prefill-heavy 单请求 hipprof trace 里，排名第二的 kernel 是 PyTorch int32 fill
  elementwise kernel，占 18.5% kernel 时间，17308 次调用。这个 workload 触发了大量
  fill，但 fill 本身应单独拆出来分析。
- 当前证据显示它是真实请求路径开销，不是 profiling 噪声，也不随 `BLOCK_M` 变化。
- 在确认具体来源前，不要直接假设它是 padding / zero 初始化而改掉；必须保持
  block table、slot mapping、KV cache metadata 和 mask 语义等价。

## 证据

来源：`misc/profile/profile_runs/hipprof_trace_exp_prof_bm16_20260623_140326/`

```text
workload: Qwen3.5-27B, 12K prefill-heavy 单请求
         prompt_tokens=12008, completion_tokens=16
backend:  TRITON_ATTN, unified_attention_2d, BLOCK_M=16
wheel:    exp/prof commit 022c747
```

kernel_summary 里 top kernels：

```text
1. kernel_unified_attention_2d                          6.95s  33.8%  (48 calls)
2. PyTorch fill elementwise (FillFunctor<int>)          3.80s  18.5%  (17308 calls)
3. chunk_fwd_kernel_o (GDN/FLA)                         3.20s  15.6%  (4944 calls)
4. chunk_scaled_dot_kkt_fwd_kernel                      1.21s   5.9%
...
```

那个 fill kernel 全名：

```text
void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>,
  std::array<char*, 1ul>>(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)
```

注意：

- 模板参数是 `FillFunctor<int>`，即 32-bit 整数 fill，不是 BF16 fill。BF16 fill 是另一个
  单独的小 kernel（`FillFunctor<c10::BFloat16>`，仅 3.1ms）。
- 17308 次调用、平均每次约 0.22ms。次数远高于 attention（48 次）和 GDN chunk
  （4944 次）。
- 该 workload 是 prefill-heavy，但 `FillFunctor<int>` 本身不等价于 prefill 计算。

## 官方负载下的强化证据

来源：`misc/profile/profile_runs/hipprof_stats_bm64_16_32k_official1_20260623_185033/
hipprof/hip_stats.hipkernel.csv`（BLOCK_M=64，16-32K 桶 official 多 prompt serving 跑，
Total kernel time 63.1s，326995 次 kernel 调用）。

```text
fill(int)  calls=81983  total=17.96s  avg=219us  pct=28.4%   #1 第一
```

对比 12K prefill-heavy 单请求 slice（同一 fill kernel）：

```text
12K 单请求:          calls=17182  total=3.77s   pct=18.4%   (BLOCK_M=16)
12K 单请求:          calls=17198  total=3.77s   pct=23.2%   (BLOCK_M=32)
12K 单请求:          calls=17189  total=3.77s   pct=25.3%   (BLOCK_M=64)
官方 16-32K 多 prompt: calls=81983  total=17.96s  pct=28.4%   (BLOCK_M=64)
```

判断：

- fill 次数从 17189 到 81983，约 4.8 倍，随 prompt / request 数近似线性增长，强烈指向
  请求路径上的 metadata / buffer 初始化开销。
- 占比从 18.4% 升到 28.4%，在官方真实负载下成为第一热点候选。
- 在 `BLOCK_M=64` 已把 `unified_attention_2d` 压下去后，fill 仍占 28.4%，说明这条线
  和 `BLOCK_M` 优化正交。

这条线应从 prefill 文档里拆出，作为 **PyTorch fill / metadata 开销** 单独跟踪。

## 已确认的 fill 行为特征

本地源码扫描 + 三组实测对比，已收窄 fill 的行为特征：

```text
1. fill 随 prompt / request 数近似线性增长：
   12K 单请求 17189 -> 官方多 prompt 81983，约 4.8 倍。
   这不是启动开销，也不像 profiling 假象。

2. fill 不随 BLOCK_M 变：
   BM16=17182 / BM32=17198 / BM64=17189，三组差 <0.1%。
   attention 的 total_q_blocks 会随 BLOCK_M 变化，但 fill calls 基本不动。

3. fill 是密集小 kernel：
   12K 单请求里 attention launch 48 次，fill 约 17.2K 次。
   这不能简单解释成 attention tile 内部成本。
```

## kernel-stack 结果

来源：`misc/profile/fillfunctor_kernel_stack_vllm_direct_20260624_211624/`

汇总结果：

```text
FillFunctor<int> calls=58816 total=12.925848709s avg=219767ns pct=58.876%
```

主机侧调用栈显示这批 kernel 来自 PyTorch fill 路径：

```text
torch.zeros_like / zero_ / fill__Scalar
  -> fill_kernel_cuda
  -> gpu_kernel_impl_nocast<FillFunctor<int>>
  -> hipLaunchKernel
```

这说明：

- 它不是 Triton attention kernel 内部 LDS bank conflict 一类问题。
- 它不是 `BLOCK_M` tile 选择直接造成的问题。
- 它更像 PyTorch 层 metadata / buffer 初始化或复用不足导致的大量 int32 fill kernel。

限制：

- `hipprof --kernel-stack` 只展开到 `EngineCore` 的 Python C API 边界，没有直接给出 Python
  源码行号。
- 仍需要靠源码候选和调用次数对齐，才能把具体来源钉死。

## 本地源码扫描收窄的候选来源

`FillFunctor<int>` 是对 int32 tensor 的显式 fill / zero，不是 BF16 计算 kernel。源码扫描后
候选收窄到：

```text
候选 A（zeros_like 栈形状最吻合）:
  vllm/v1/worker/gpu/block_table.py
  - input_block_tables = [torch.zeros_like(b.gpu) for b in self.block_tables]
  这个和 kernel-stack 的 zeros_like -> zero_ -> fill__Scalar 最吻合。
  但代码位置看起来偏初始化 / 持久 buffer，是否能解释请求期 58K calls 还要验证。

候选 B（GDN / causal conv metadata 路径，Qwen3.5 下可疑）:
  vllm/v1/attention/backends/utils.py: compute_causal_conv1d_metadata
  - batch_ptr = torch.full((MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device)
  - token_chunk_offset_ptr = torch.full(...)
  - resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)
  这是 int32 metadata fill，和当前 kernel dtype 对得上。

候选 C（每步 attention metadata 构造）:
  vllm/v1/worker/gpu_model_runner.py: _build_attention_metadata
  - block table padding fill_(-1)
  - dcp_local_seq_lens padding fill_(0)
  这些是 per-step 路径，但单独看调用次数可能不足以解释全部 fill。

候选 D（slot mapping padding）:
  vllm/v1/worker/gpu_model_runner.py: _get_slot_mappings
  - slot_mapping = torch.zeros(..., dtype=torch.int64)
  - slot_mapping[...].fill_(-1)
  这里更可能对应 FillFunctor<long>，不是当前 FillFunctor<int> 主热点。
```

已排除或优先级较低：

```text
- triton_attn.py: seq_lens.fill_(1)：主要是 graph capture 路径，不应解释每次请求大头。
- triton_attn.py: output.fill_(0)：空请求早退路径。
- block_table.py: clear() / request-end 清理：不是主要请求计算窗口。
```

## 当前缺口

现在已经能确认 fill 是独立热点，但还没完全确认具体 Python 来源：

```text
已确认:
  - dtype 是 int32。
  - 调用次数很高。
  - 平均约 219us。
  - 不随 BLOCK_M 变化。
  - kernel-stack 指向 PyTorch zeros_like / zero_ / fill__Scalar。

未确认:
  - 58K / 17K / 82K 次调用分别由哪些源码位置贡献。
  - 是否主要来自 block table / causal conv metadata / attention metadata。
  - 是否存在一次请求内反复重新分配和重新 fill 可复用 buffer 的情况。
```

## 下一步验证

不要再把这条线当作 prefill 本体优化。建议单独做 attribution：

```text
A. 在 block_table.py、attention/backends/utils.py、gpu_model_runner.py 的候选 fill 点
   做极小范围计数插桩，只记录调用次数和 shape，不改语义。

B. 用同一个 12K prefill-heavy 单请求重跑短 profile，对齐：
   - Python 候选点调用次数
   - hipprof FillFunctor<int> calls
   - shape 是否都是 int32 GPU tensor

C. 若定位到固定大小 buffer 反复 torch.full / zeros_like：
   优化方向是预分配 + 复用 + 只填有效/无效边界，而不是每次重新分配并全量 fill。

D. 若定位到必须每步重填的 metadata：
   需要考虑融合 metadata fill/copy，或改为自定义小 kernel 一次完成多个 int32 buffer。
```

## 风险边界

- 这不是 attention / GEMM / GDN 计算路径，改它属于 host 侧 / metadata / buffer 管理优化。
- 必须保持数学等价，不能改变 block table、KV cache、slot mapping、scheduler 或 mask 语义。
- `PAD_SLOT_ID`、`-1`、`0` 这类填充值可能参与无效槽位屏蔽，不能因为看似 padding 就省掉。
- 在 attribution 证明来源和量级前，不写性能 patch。

## 当前状态

```text
status: source_unconfirmed
category: pytorch_fill_metadata_overhead
not_category: prefill_compute
next: 对候选 fill 点做最小计数插桩，确认 17K/58K/82K calls 的具体来源
```
