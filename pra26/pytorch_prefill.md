# PyTorch prefill 侧开销待查

## 结论

- 12K prefill 单请求 hipprof trace 里，排名第二的 kernel 是一个 PyTorch fill
  elementwise kernel，占 18.5% kernel 时间，17308 次调用。这个大头在原
  hotpath_conclusion 里被一带而过，需要单独确认它是真实 prefill 开销还是
  profiling 噪声。
- 在确认性质前，不要把它当优化目标，也不要假设它是 padding/zero 初始化而直接改。

## 证据

来源：`misc/profile/profile_runs/hipprof_trace_exp_prof_bm16_20260623_140326/`

```text
workload: Qwen3.5-27B, 12K prefill 单请求 (prompt_tokens=12008, completion_tokens=16)
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
- 17308 次调用、平均每次 0.22ms。次数远高于 attention（48 次）和 GDN chunk（4944 次）。

## 官方负载下的强化证据

来源：`misc/profile/profile_runs/hipprof_stats_bm64_16_32k_official1_20260623_185033/
hipprof/hip_stats.hipkernel.csv`（BLOCK_M=64，16-32K 桶 official 多 prompt serving 跑，
Total kernel time 63.1s，326995 次 kernel 调用）。

```text
fill(int)  calls=81983  total=17.96s  avg=219µs  pct=28.4%   #1 第一
```

对比单请求 12K prefill slice（同一 fill kernel）：

```text
单请求 12K prefill:   calls=17182  total=3.77s  pct=18.4%   (BLOCK_M=16)
单请求 12K prefill:   calls=17189  total=3.77s  pct=25.3%   (BLOCK_M=64)
官方 16-32K 多 prompt: calls=81983  total=17.96s pct=28.4%   (BLOCK_M=64)
```

判断：

- fill 次数从 17189 → 81983，约 4.8 倍，随 prompt 数近似线性增长 → 强烈指向
  **per-request 级初始化开销**，不是 profiling 采样假象，也不是一次性 capture 开销。
- 占比从 18.4% 升到 28.4%，在官方真实负载下是**第一瓶颈**，比 attention 还高。
- 在 BLOCK_M=64 已把 unified_attention_2d 压到 19.5% 后，fill 仍占 28.4% 浮成第一 →
  这条线和 BLOCK_M 优化正交，BLOCK_M 动不了它，必须单独查来源并单独优化。

这条线优先级从"待查"升为**真实负载第一瓶颈候选**。

## 已确认的 fill 行为特征

本地源码扫描 + 三组实测对比，已收窄 fill 的行为特征：

```text
1. fill 随 prompt 线性增长（单请求 17189 → 官方 3 请求 81983，4.8 倍）→ per-request 路径，
   不是启动开销、不是 profiling 假象（对照 flash_fwd_kernel 是启动开销，单请求无）。
2. fill 不随 BLOCK_M 变：BM16=17182 / BM32=17198 / BM64=17189，三组差 <0.1%。
   而 attention 的 total_q_blocks 三组差 5 倍（2049/820/410）→ fill 不是 per-q-block，
   调 BLOCK_M 救不了它，与 BLOCK_M 优化完全正交。
3. fill 是 per-attention-layer 的密集小 fill：单请求 48 次 attention launch，
   fill 17189 次 → 每次 attention launch 约 358 次 fill。358 ≈ 24(num_query_heads)×15，
   但精确分解本地源码扫不出来。
```

## 本地源码扫描收窄的候选来源

`FillFunctor<int>` 是对 int32 tensor 的显式 fill（非 empty zero-init）。源码扫描后候选收窄到：

```text
候选 A（per-step 密集，最可疑）:
  vllm/v1/worker/gpu_model_runner.py 每步构建 int buffer
  - L3499/3510: slot_mapping = torch.zeros(...) + slot_mapping[...].fill_(-1)
  - L1985: dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
  - L1846/1849: num_draft_tokens / num_decode_draft_tokens (np → torch)
  这些是 per-step 的，但每步十几次，量级对不上 358/层。

候选 B（per-attention-call）:
  vllm/v1/attention/backends/utils.py:755-776
  MAX_NUM_PROGRAMS = max(1024, mlist_len)*2（至少 2048）
  batch_ptr / token_chunk_offset_ptr 的 torch.full 或 resize+fill(PAD_SLOT_ID, int32)
  每次 attention 调用约 2 次 fill。

已排除:
  - triton_attn.py:199 seq_lens.fill_(1)：只在 graph capture 触发，非每步。
  - triton_attn.py:445 output.fill_(0)：空请求早退。
  - block_table.py:200 block_table.fill_(0)：clear() 请求结束才调。
```

**关键缺口**：候选 A+B 每步十几次 fill，对不上 358 次/attention-layer。说明 fill 不全是 vLLM 显式发的，可能含：
- PyTorch / ROCm allocator 在 tensor 分配时的隐式 zero-init；
- 或 attention kernel 内部 Triton 编译插入的辅助 fill。

**本地源码扫描到此为止，无法精确分解 358 的来源，必须靠 hipprof `--kernel-stack` 实测。**

## 验证方法（已更新）

先不改代码，先定位来源：

```text
A. hipprof --kernel-stack 对 FillFunctor<int> 取主机端调用栈（优先级最高）。
   用 session 控制只采一个 prefill 请求窗口，从 hiptrace CSV 里看每次 fill 的 host stack，
   直接对号到候选 A/B 的具体行号，或发现是 PyTorch 隐式 init。这是零代码改动的一锤定音。
B. 问题 2 已有答案：fill 随 prompt 线性、不随 BLOCK_M 变、~358 次/attention-layer。
   若要进一步确认 358 的分解，跑不同 num_seqs 的 prefill，看 fill calls 是否随 num_seqs
   或 num_query_heads 变化。
C. 对照 decode-heavy slice 的 stats，看 fill 是否主要在 prefill（decode 也有 attention
   layer，若 decode fill 占比和 prefill 相似则 fill 与 prefill/decode 无关，纯 per-layer）。
D. kernel-stack 定位后，若指向固定 buffer 反复 resize+fill（如候选 B 的 MAX_NUM_PROGRAMS），
   优化方向是预分配+复用而非每次 fill；若指向 PyTorch 隐式 init，则看能否减少临时 tensor
   分配。任何改动必须数学等价（PAD_SLOT_ID / fill(-1) 等是 mask 无效槽位，不能漏填）。
```

## 风险边界

- 这个 fill 不是 attention / GEMM / GDN 计算路径，改它属于“host 侧 / metadata / buffer
  管理优化”，必须保持数学等价，不能改变 block table、KV cache 语义、scheduler 行为或
  输出口径。
- 在 A/B 步骤确认来源和量级前，不写 patch。
- 如果 kernel-stack 证明它是 profiling 采样假象（如 capture 期间重复 fill），则不是真实
  瓶颈，记录结论后关闭这条线。

## 当前状态

```text
status: source_unconfirmed
owner: unassigned
next: hipprof --kernel-stack 定位 fill kernel 主机端来源；短 slice 重测看次数随长度变化
```
