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

## GQA6 优化原理详解

### 1. 先明确 GQA6 的计算形状

目标 full-attention 层有 24 个 query heads 和 4 个 KV heads，因此：

```text
GQA ratio = num_query_heads / num_kv_heads = 24 / 4 = 6
```

同一个 KV head 被 6 个 query heads 共享。kernel 将“query token”和“该 KV
head 下的 query head”展平成矩阵的 M 维，一行代表一个
`(query_position, query_head)`。这种布局能让 6 个 query heads 复用同一份 K/V，
但前提是 CTA 的行数能够被每组 head 数整除。

GQA6 的特殊之处在于 6 不是 2 的幂，而 Triton 通常希望
`tl.arange(0, BLOCK_M)` 使用 16、32、64 等 2 的幂。对手的优化就是围绕
“GQA6 与 power-of-two tile 不整除”这一矛盾逐步展开。

### 2. H11.3：消除 `BLOCK_M=16` 的跨 block 重叠

通用 AITER 映射使用：

```text
BLOCK_M = 16
num_queries_per_kv = 6
BLOCK_Q = BLOCK_M // num_queries_per_kv = 2

query_pos  = q_block * BLOCK_Q + row // 6
query_head = kv_head * 6 + row % 6
```

问题在于 `BLOCK_Q=2` 只声明当前 CTA 处理 2 个 query tokens，但 16 行实际会
触及 3 个 tokens：

| CTA 内行号 | 映射 query token | 映射局部 head | 是否属于声明的 2-token block |
| --- | --- | --- | --- |
| `0..5` | `q[2b]` | `h[0..5]` | 是 |
| `6..11` | `q[2b+1]` | `h[0..5]` | 是 |
| `12..15` | `q[2b+2]` | `h[0..3]` | 否，越过 block 边界 |

下一个 CTA 从 `q[2b+2]` 开始，又会计算其 `h[0..5]`。因此前一个 CTA 的最后
4 行与后一个 CTA 的前 4 行重叠。mask 可以挡住序列尾部越界，却不能消除序列
中间这些合法但重复的 QK、softmax、PV 和 output store。

H11.3 不把 `BLOCK_M` 强行改成非 2 次幂，而是将 6 heads 分成 3 组，每组
2 heads：

```text
HEADS_PER_CTA = 2
GQA_SPLITS = 3
BLOCK_M = 16
BLOCK_Q = BLOCK_M // HEADS_PER_CTA = 8

16 rows = 8 query tokens x 2 query heads
```

新映射为：

```text
query_pos  = q_block * 8 + row // 2
query_head = kv_head * 6 + head_group * 2 + row % 2
grid       = (query_blocks, 4 kv_heads, 3 head_groups)
```

每个 CTA 现在恰好覆盖一个完整的 `8 x 2` 矩形，没有尾部行落入下一 query
block。第三个 grid 轴只是 GQA head group，不是 segmented 3D decode，所以
不引入 segment workspace、第二个 reduction kernel 或额外 launch。

#### CTA 数量为什么会下降

忽略最后一个不完整 tile，设 query 长度为 `L`、KV head 数为 4：

```text
通用映射 CTA ~= 4 x L/2
H11.3 CTA    ~= 4 x 3 x L/8
H11.3 / 通用 ~= 3/4
```

所以 H11.3 长序列下约少发射 25% 的 CTA。等价地看，通用映射为每个 token/KV
head 计算约 8 行，其中只有 6 行唯一；H11.3 接近只计算所需的 6 行。这里减少
的是重复工作，不是删除有效 attention 计算。

实验上，H11.3 相对直接基线 H10.4 的 all3 小样本加权吞吐提高
`7.6875%`，且 9 条生成文本与 H10.4 逐请求相同。这个端到端收益小于理论
25%，因为 attention 只是完整推理链的一部分，而且有效行内部的 QK/PV 工作量
没有减少。

### 3. H11.4：先解决寄存器和 LDS 压力

H11.3 修正了映射，但长 prefill 编译结果仍受 VGPR/LDS 限制。H11.4 在
`max_seqlen_q >= 128` 时切换为：

```text
num_warps = 2
num_stages = 1
waves_per_eu = 1
matrix_instr_nonkdim = 16
kpack = 2
```

短 prefill 仍保留 H11.3 的 `4 warps / 2 stages`，因为短请求 CTA 数少，过度
减少 warps 可能无法填满设备。长 prefill 的编译资源记录变化为：

| 配置 | VGPR | LDS | 目的 |
| --- | ---: | ---: | --- |
| H11.3 长 prefill 原配置 | 约 253 | 16896 B | 4 warps、2 stages |
| H11.4 | 约 175 | 8192 B | 降低资源占用，提高可驻留 wave 机会 |

`matrix_instr_nonkdim=16 + kpack=2` 让 Triton 选择适合 gfx936 BF16 的原生
矩阵布局。这里没有改变 attention 数学、grid 或输入输出，只改变编译布局，
所以它属于硬件资源调优。

### 4. H11.5：将 8-token query tile 扩为 32 tokens

H11.5 在精确目标 workload 上进一步设置：

```text
BLOCK_M = 64
HEADS_PER_CTA = 2
BLOCK_Q = 32
GQA_SPLITS = 3
```

一个 CTA 从 H11.4 的 `8 tokens x 2 heads` 扩为 `32 tokens x 2 heads`。
对同一 KV head/head group 而言，query CTA 数约变为原来的四分之一：

```text
H11.4 CTA ~= 4 x 3 x ceil(L / 8)
H11.5 CTA ~= 4 x 3 x ceil(L / 32)
```

有效 QK/PV 数学量并未凭空消失，但以下固定开销被摊薄：

- block-table 索引和 paged KV 地址计算；
- K/V tile 的全局加载以及进入矩阵操作数的准备；
- online-softmax 循环控制、状态维护和 CTA 调度；
- 同一 K/V tile 在不同小 query CTAs 中的重复读取。

同一份 K/V tile 在 H11.4 中服务 16 个 query/head rows，在 H11.5 中服务
64 行，CTA 内 K/V 复用粒度提高 4 倍。这是 wide-query tile 的主要价值。

### 5. H11.5：按 CTA 收紧 causal K/V 上界

仅扩大 query tile 还不够。causal attention 中，第 `j` 个 query CTA 最多只需要
看到该 CTA 最后一个 query token。H11.5 使用：

```text
causal_query_stop = min((q_block + 1) * BLOCK_Q, query_len)
num_kv_tiles = ceil((context_len + causal_query_stop) / TOKENS_PER_BLOCK)
```

设当前 chunk 的 query 长度为 `L`、已有 context 为 `C`、query tile 为 `Bq`、
K/V tile 为 `Bk`。若所有 CTA 都扫描到 chunk 末尾，tile 访问量近似：

```text
Nq * ceil((C + L) / Bk),  Nq = ceil(L / Bq)
```

收紧后变成：

```text
sum(j=0..Nq-1) ceil((C + min((j+1)*Bq, L)) / Bk)
```

当 `C=0` 且 `L` 足够大时，后者接近前者的一半，也就是只计算 causal 下三角
需要的 tile，而不是先做近似完整方阵再 mask 掉未来位置。当 `C` 很大时，历史
context 仍必须读取，节省主要来自当前 chunk 内未来 token 的部分。

kernel 仍在最后一个 K/V tile 内保留逐行 causal mask，因此缩短循环上界不会
放宽注意力范围。online softmax 的 `running_max`、`running_sum` 和 FP32
accumulator 也保持原算法，不需要保存完整 score matrix。

### 6. 为什么逻辑 K/V tile 是 56，而物理矩阵宽度是 64

比赛的 KV cache page size 为 784：

```text
784 = 14 x 56
784 % 64 = 16
```

如果直接使用 64-token 逻辑 tile，部分 tile 会跨越两个不一定连续的物理 cache
pages，需要读取两次 block table 并拼接地址。H11.5 选择 56 个有效 tokens，
保证每个逻辑 tile 完整落在一个 784-token page 中：

```text
logical TOKENS_PER_BLOCK = 56
physical BLOCK_SIZE      = 64
valid columns            = 0..55
padded and masked        = 56..63
```

64 列的物理形状又能与 16x16x16 BF16 MMAC 对齐。head dimension 256、
`BLOCK_M=64` 和 K/V padded width 64 都是 16 的倍数，最终 code object 确认
使用 `v_mmac_f32_16x16x16_bf16`。这是一种“逻辑分页对齐”和“物理矩阵对齐”
分离的设计。

56 相比 64 有 12.5% 的列 padding，但避免了跨页控制流。后续实验曾尝试显式
支持 logical-64 跨页，即使 VGPR 从 216 降到 198，端到端加权性能仍为负；
另一些 padded-32/16 方案也未降低 32 KiB LDS，并增加了 online-softmax 循环
次数。这些反例说明 tile 不能只看 padding 比例或 VGPR 单项指标。

### 7. H11.5 的 kernel 数据流

目标请求实际执行以下流程：

1. grid axis 0 定位 32-token query block，axis 1 定位 KV head，axis 2 定位
   2-head group。
2. 加载 `64 x 256` 的 Q tile，对应 32 tokens 和 2 query heads。
3. 根据当前 query block 的 causal 上界，只遍历需要的 56-token K/V tiles。
4. 每个逻辑 tile 从一个 paged-cache block 取数，并 mask 64 列中的后 8 列。
5. 执行 `Q @ K`、scale、causal mask 和 online softmax 更新。
6. 执行 `P @ V`，在 FP32 accumulator 中累积，最后归一化并写回 BF16 output。

最终 H11.5 code object 的记录为 `216 VGPR / 32768 B LDS / 0 spill`。虽然资源
高于 H11.4，但 wide tile 的 K/V 复用、四分之一 query CTA 和 causal 剪枝带来的
收益更大；零 spill 也避免了寄存器溢出到显存。

### 8. 性能证据如何证明收益来自 prefill

H11.5 相对 H11.4 的 standalone same-input raw kernel 加速为
`1.6755x–3.1083x`。固定 all3 小样本相对 R24 的结果为：

| 输入档位 | mean TTFT 变化 | mean TPOT 变化 |
| --- | ---: | ---: |
| 4-8K | `-15.887%` | `+0.273%` |
| 8-16K | `-20.472%` | `+0.184%` |
| 16-32K | `-24.652%` | `+0.052%` |

TTFT 明显下降且输入越长收益越大，符合“减少长 causal prefill 重复 work”的
机制；TPOT 基本不变，因为 `max_seqlen_q=1` 的 decode 明确回退原 AITER 路径。
这比只观察总 output tok/s 更能证明 GQA6 kernel 的贡献位置。

比赛使用单卡、单请求并发，单个请求的近似耗时为：

```text
request_time ~= TTFT + (output_tokens - 1) x TPOT
output_throughput ~= output_tokens / request_time
```

因此 prefill 不会被其他并发请求隐藏。GQA6 即使不改变单 token decode 速度，
只要降低每个长请求的 TTFT，也会直接缩短串行评测总时间并提高 output tok/s。
输入越长，attention prefill 占比越高，H11.5 的收益也越容易体现在吞吐和 SLA。

需要注意，最终 full x3 使用的是 H11.5 + H10.8 组合。组合相对 R24 的三档
吞吐提升为 `+6.744% / +9.540% / +13.724%`，其中还包含 H10.8 对 decode
TPOT 的贡献，不能全部归因于 GQA6。

正确性证据也必须分层理解：H11.5 standalone same-input 对 H11.4 的最坏
`max_abs` 为 `4.8828125e-4`；all3 小样本每档各有 1/3 请求发生生成文本或长度
变化，因此该窗口中的 raw output tok/s 只用于候选筛选。最终可计分结论来自
H11.5 + H10.8 的 full x3、450/450 请求、SLA 和固定 accuracy `K=1.0`，而
不是用一次微基准替代端到端正确性。

### 9. 为什么它有效，但不能泛化成“所有 GQA6 都更快”

真正产生收益的是一组条件同时成立：

- GQA ratio 恰好为 6，可拆成 `3 x 2 heads`，同时保留 2 次幂 M tile；
- 24/4 heads 和 head size 256 固定，矩阵形状与 MMAC 布局可冻结；
- page size 784 恰好被 56 整除，逻辑 tile 不跨物理 page；
- workload 是单序列长 causal prefill，wide tile 有足够并行工作且 causal
  剪枝空间大；
- gfx936、BF16、Triton/AITER 版本固定，编译资源和指令选择可复验；
- 不匹配的短请求、多序列和 decode 有可靠回退路径。

因此，GQA6 的效果不是来自一个孤立参数，而是来自映射正确性、CTA 粒度、K/V
复用、causal 算法剪枝、分页布局和硬件矩阵指令的共同配合。

## GQA6 路径的安全边界

入口位于
[rocm_aiter_unified_attn.py](vllm/v1/attention/backends/rocm_aiter_unified_attn.py)，
核心 kernel 位于
[rocm_aiter_unified_attention_gqa6.py](vllm/v1/attention/ops/rocm_aiter_unified_attention_gqa6.py)。

实际 dispatch 分三层：

| 路径 | 额外条件 | 配置 |
| --- | --- | --- |
| H11.5 wide-causal | 单序列、cache block 784、`max_seqlen_q >= 128` | BM64/BQ32、逻辑 K/V 56、物理 64 |
| H11.4 long prefill | prefill 且 `max_seqlen_q >= 128`，但不满足 H11.5 | BM16/BQ8、2 warps/1 stage |
| H11.3 short prefill | `1 < max_seqlen_q < 128` | BM16/BQ8、4 warps/2 stages |
| 原 AITER | decode 或任一核心 gate 不匹配 | 通用 fallback |

进入 H11.3/H11.4/H11.5 体系前，构造期和调用期还共同限制：

- `gfx936`、BF16、head size 256；
- 24 个 query heads、4 个 KV heads，即 GQA6；
- `kv_cache_dtype=auto`、decoder self-attention；
- 不启用 ALiBi、sliding window、logits soft cap 或 sinks；
- Q 尾部 shape 为 `(24, 256)`，K/V cache 尾部 shape 为 `(4, 256)`。

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
