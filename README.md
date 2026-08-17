# PRA2026-BH408 对手代码学习与优化分析

## 分支用途

本分支用于独立保存和研究 `pra2026-bh408` 的完整比赛源码。分支基线为
对手公开提交 `d6676b0`，除本分析文件外，其他文件均来自该提交。

- 学习分支：`study/pra2026-bh408`
- 对手原始历史：`vendor/pra2026-bh408-source`
- 我们的正常提交分支：`submit/v3.0`
- 隔离原则：本分支只研究对手代码，不向 `submit/v3.0` 自动合并任何代码

vLLM 上游项目介绍保留在 [README_UPSTREAM.md](README_UPSTREAM.md)。对手的
构建、环境、结果和合规说明分别位于 [BUILD.md](BUILD.md)、
[ENVIRONMENT.md](ENVIRONMENT.md) 和 [docs/cscc](docs/cscc)。

## 总体结论

对手的核心思路不是做大范围通用优化，而是针对比赛唯一硬件和模型 shape，
通过严格 gate 建立多个窄而深的专用快路径：

1. **H11.5 wide-causal GQA6 prefill**：优化长上下文首 token 时间（TTFT）。
2. **H10.8 gfx936 strided LLMM1**：优化 decode 阶段大投影，降低 TPOT。
3. **H10-only M=4096 TunableOp profile**：固定 prefill GEMM 的 rocBLAS solution。
4. **完整的实验淘汰机制**：只有性能、输出、accuracy 和 SLA 同时闭环的候选
   才进入最终栈，负收益或数值不稳定方案会明确回滚。

公开证据中，H11.5 + H10.8 的三轮 full 综合分均值为 `88.5484555`，accuracy
系数为 `K=1.0`。后续 H10-only 在 all3 窗口得到约 `+0.939469%` 的加权提升，
但没有重跑同口径 full x3，因此不能直接把该增量加到最终综合分上。

## 为什么 GQA6 有效

### 1. 修复通用 kernel 对 GQA6 的非整除映射

目标模型每个 KV head 对应 6 个 query heads。原 AITER 2D attention 使用：

```text
BLOCK_M = 16
BLOCK_Q = 16 // 6 = 2
```

一个 CTA 的 16 行只能完整容纳 `2 tokens x 6 heads = 12 rows`。剩余 4 行会
映射到下一个 query token，从而与相邻 CTA 的工作发生重叠。即使 mask 能维持
结果语义，这些重复行仍会产生额外的加载和计算。

H11.3 将 6 个 query heads 拆成 3 组，每组 2 heads：

```text
HEADS_PER_CTA = 2
GQA_SPLITS = 3
BLOCK_M = 16
BLOCK_Q = 8
16 rows = 8 tokens x 2 heads
```

这样每个 CTA 都只覆盖完整的 token/head 矩形，不再跨入相邻 query block。
新增的第三个 grid 维度仅表示三个 GQA head group，并不是 segmented decode，
不需要临时 segment buffer 或额外归约 kernel。

这一阶段相对 H10.4 的小样本加权吞吐提高 `7.6875%`，三档 mean TTFT 分别
下降约 `11.45% / 14.45% / 16.75%`。收益主要来自 prefill，而非 decode。

### 2. 用编译参数提高 gfx936 占用率

H11.4 针对长 prefill 使用：

```text
num_warps = 2
num_stages = 1
matrix_instr_nonkdim = 16
kpack = 2
```

实验记录显示，kernel 资源从约 `253 VGPR / 16896 B LDS` 降到
`175 VGPR / 8192 B LDS`。这能让 gfx936 上同时驻留更多 wave，减少长序列
attention 因寄存器压力造成的 occupancy 限制。

### 3. wide-causal 减少长 prefill 的重复扫描

H11.5 将 query tile 扩大为：

```text
BLOCK_M = 64
HEADS_PER_CTA = 2
BLOCK_Q = 32
```

原路径中的每个 query CTA 会扫描到整个 query chunk 的末尾，再依靠 causal mask
丢弃未来 token。wide-causal 路径按照当前 CTA 最后一个 query 位置收紧 K/V
循环上界，较早的 CTA 不再加载和计算注定被 mask 的未来 K/V。

长上下文 causal prefill 的冗余随序列长度增长，因此三档输入越长，TTFT 改善
越明显。相对 H11.4，三档 mean TTFT 分别下降约
`15.887% / 20.472% / 24.652%`，而 TPOT 基本不变。

### 4. 逻辑 tile 与物理 page 同时对齐

比赛配置的 KV cache page size 为 `784`，满足：

```text
784 = 14 x 56
```

H11.5 使用 56-token 的逻辑 K/V tile，因此一次 tile 不会跨越 page；同时将其
padding 到 64 列，以适配 AMD MFMA 的矩阵布局。这样兼顾了 paged KV 地址计算
和矩阵指令形状，避免为了 64-token tile 处理跨页边界。

最终 kernel 的记录资源约为 `216 VGPR / 32768 B LDS / 0 spill`，并命中
`v_mmac_f32_16x16x16_bf16`。raw kernel 相对 H11.4 的记录加速范围为
`1.6755x` 到 `3.1083x`。

### 5. 有效的根本原因

GQA6 本身并不会自动带来这些收益。真正有效的是同时利用了以下固定条件：

- GQA ratio 恰好为 6，可以稳定拆成 `3 x 2 heads`；
- head size 固定为 256，能够冻结 MFMA 和寄存器布局；
- page size 固定为 784，可使用不跨页的 56-token tile；
- workload 是单序列长 causal prefill，收紧 causal 上界收益显著；
- gfx936、BF16 和固定 Triton/AITER 版本使编译参数可重复验证。

因此它是一个高度特化的 kernel，不应被误解为适合所有 GQA 模型的通用配置。

## GQA6 路径的安全边界

入口位于
[rocm_aiter_unified_attn.py](vllm/v1/attention/backends/rocm_aiter_unified_attn.py)，
核心 kernel 位于
[rocm_aiter_unified_attention_gqa6.py](vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py)。

构造期和调用期 gate 共同限制以下条件：

- `gfx936`、BF16、head size 256；
- 24 个 query heads、4 个 KV heads，即 GQA6；
- cache block size 784；
- decoder self-attention、单序列长 causal prefill；
- 不启用 ALiBi、sliding window、logits soft cap 或 sinks；
- decode、短 query、多序列及不匹配 shape 回退到原 AITER 路径。

这种 fail-closed 设计值得学习：特化路径只处理已证明有效的输入域，其余情况
保留成熟实现，减少优化对正确性和其他模型的影响。

## 其他主要优化

### H10.8：gfx936 strided LLMM1

该路径优化 decode 阶段 `n=1` 的大投影，主要目标 shape 为：

```text
dtype = BF16
k = 5120
m in {14336, 16384, 34816}
bias = None
```

三个大投影使用 `LLMM1Strided(4, 640)` wave-pair reduction，`m=96` 保留原
LLMM1，其他 shape 回退常规 GEMM。H10.8 相对 H11.5 的小样本三档 TPOT
改善约 `5.233% / 5.033% / 4.906%`。

值得借鉴的不是直接复制参数，而是先从 profiler 找出 decode 中重复出现的固定
GEMV/skinny GEMM，再按设备、dtype、连续性和精确 shape 做窄 gate。

### H10-only：M=4096 rocBLAS TunableOp

对手没有在线搜索 solution，而是将经过独立长时间验证的 rocBLAS 结果冻结成
profile：Attention QKV、GDN QKVZ、MLP gate/up 使用 `Gemm_Rocblas_20981`，
MLP down 使用 `Gemm_Rocblas_20979`。loader 在设备初始化后加载，并在 graph
capture 前复核状态。

该方法的价值在于消除运行时调优开销和不确定性；风险在于 solution 与硬件、
ROCm/rocBLAS 版本、矩阵 shape 强绑定，环境发生变化时必须拒绝命中，而不能
静默复用。

## 与我们当前方案的对比

| 维度 | 我们的 `submit/v3.0` | 对手方案 |
| --- | --- | --- |
| Prefill attention | KV mirror + non-paged FlashAttention | 直接读取 paged KV 的专用 GQA6 Triton kernel |
| 显存取舍 | mirror 约增加 2 GiB 显存占用 | 不需要同类 mirror，KV 容量更有优势 |
| 通用性 | FlashAttention 路径更成熟、覆盖面更广 | 只覆盖 gfx936/BF16/head256/GQA6/特定 page size |
| 主要收益 | 依赖 FA kernel 与镜像布局 | 消除 GQA6 行重叠并裁剪 causal K/V 扫描 |
| Decode | 我们已有 3D decode 固定配置 | H10.8 专用 LLMM1 改善大投影 TPOT |
| 风险 | mirror 搬运和额外显存 | Triton 编译资源、固定 shape 和版本耦合 |

两条 prefill 路径是互斥实现，不能把对手 GQA6 kernel 直接叠加在我们的 KV
mirror + FlashAttention 上并假定收益相加。公开分数的测试时间和代码基线也不
完全相同，在没有同机、同 wheel、同评测脚本的 full x3 结果前，不能只凭分数
判断哪条路径更快。

## 建议学习顺序

1. **先修正通用 fallback 的 GQA6 映射。** 检查我们的 Triton unified
   attention 是否同样存在 `BLOCK_M` 不能被 6 整除造成的跨 block 行映射。
2. **将 paged GQA6 作为显式实验路径。** 保留默认 FA 路径，通过严格开关在
   BW1000 上与 KV mirror 方案 A/B，不直接替换生产路径。
3. **分别度量 TTFT 和 TPOT。** GQA6 主要改善 prefill；LLMM1 主要改善 decode，
   必须避免只看总吞吐而误判贡献来源。
4. **同时记录显存与 KV 容量。** paged kernel 即使速度接近，也可能因省去
   mirror 而在长上下文容量上更有价值。
5. **保持 fail-closed gate。** 设备、dtype、head 数、head size、page size、
   attention 类型和序列形态任一不匹配都应回退。
6. **按候选逐项闭环。** 每项依次通过输出 hash、accuracy、TTFT/TPOT、SLA、
   显存和 full x3，禁止一次合入多个无法单独归因的变化。

## 参考证据

- [优化方案](docs/cscc/OPTIMIZATION.md)
- [完整阶段实验记录](docs/vllm_cscc_stage_experiment_results.md)
- [最终结果](docs/cscc/RESULTS.md)
- [合规说明](docs/cscc/COMPLIANCE.md)
- [源码哈希清单](evidence/manifests/repo_source.sha256)

本 README 是对公开代码和实验记录的学习总结，不改变对手源码所声明的测试
结果，也不代表这些参数已经在我们的 BW1000 环境中复现。
