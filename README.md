# Qwen3.5 vLLM Contest Fork

本仓库用于 2026 先导杯「基于国产加速卡的千问大模型推理服务优化」。基础代码来自
SourceFind/OpenDAS vLLM v0.18.1，目标环境是国产 DCU/BW1000（`gfx936`）上的
Qwen3.5-27B BF16 单卡在线推理。

本项目的分支信息如下：

- `develop` 是日常研究分支。
- `submit/vN` 是比赛提交线。

因此，`develop` 上出现过的代码不一定进入 `submit/vx.x`，`submit/vx.x` 的后续优化也不一定
回合并到 `develop`。
在gitlab上，`develop`分支被更名为`workspace`。

## 优化思路简述
调研技术方案后，团队认为vllm 0.18.1 - 是一个非常关键的信息，这个版本号非常新，以至于团队相信，
几乎所有通用性优化vllm社区都应该已经制作。
早期调研发现，Qwen 3.5 的 hybrid KV cache 会对 attention page 与 GDN state page
进行大小和 chunk 对齐，在目标配置下会将 attention 的 `BLOCK_SIZE` 计算为 784。另一方面，
在未启用 AITER unified attention、AITER MHA 或 prefill/decode split backend 时，ROCm 平台级
的 backend 优先级会默认回退到 Triton unified attention。这两个机制相互独立：前者产生
非典型的大 page 形状，后者决定 full-attention 层的默认 backend。对初赛纯文本的
full-attention prefill，实测发现在满足语义和 shape 约束时，将 K/V 转为连续布局后使用
non-paged FlashAttention 比通用的 paged unified attention 更快。此外，线性注意力区有各种
小 kernel，进一步拖累了整个模型的计算。
最初团队认为，可能这个vllm在DCU上只是“能跑”的水准，但是团队拿到真实算力平台后，采样Triton JIT
IR 发现，Triton编译器甚至能编译并使用builtin MMA能力，利用上DCU自身的matrix core，外加长期
优化不顺，我们最终判定，哪怕是通用优化，triton编译器在这个平台上也已经做的足够好──以至于不能盲目
手写custom kernel。事实上JIT技术也能拿到更多平台运行时信息，手写算子也更像黑盒，更不能搬迁NVIDIA
经验。
所以关键点是：1.怎么让Triton kernel跑的更快，在这个模型和平台上表现更好？2.能不能通过
custom kernel实现自定义快速算子（后来证明难度很高且投入产出比低）；4.能不能想办法开启因为各种原因最
终没有被选中的快速算子，从而绕开某些算子过于通用而不快的问题？
对于1，我们通过利用AI工作流调整参数解决，比如调参BLOCK_M、TILE_SIZE，以及通过微调相关算子代码解决
，比如线性注意力区投影的各种GEMV算子代码微调；对于4，我们先是通过将gfx936加入编译白名单，从而让其能
够命中 LLGemm 算子，然后在 Triton backend 内为满足约束的 full-attention prefill 增加
non-paged FlashAttention 快路径：当前 chunk 执行 causal FlashAttention，历史 K/V 从 paged
cache gather 到连续 workspace 后执行 prefix attention，各段结果再通过 LSE merge 合并；
decode 仍保持 Triton paged attention 路径。这减少了实际代码量，同时带来了巨大的吞吐提升，
因为这个系统本来就拥有由于模型形状和平台检查而未被选中的快速路径。
所有优化都和profile证据息息相关，尽管对hipprof工具的错误使用也产生了错误决策，但我们团队依然坚持
以实际证据为主。事实上，我们团队最终能摸清到底哪些路径主导了整个流程，正是依靠AI profile工作流传回
来的相关信息。

## AI 辅助优化工作流

`develop` 的主要成果不是一组可以直接提交的 kernel，而是把 AI 辅助优化固化成可审计、可回滚、
可复现的工程闭环：

```text
读取 AGENTS.md、目标 config.json 和比赛规则
                    |
                    v
建立干净 baseline，记录源码、wheel、环境和三档结果
                    |
                    v
只抓服务期的短 profile，区分 prefill / decode 与实际 backend
                    |
                    v
锁定热点 kernel、真实 shape、数据类型和 fallback
                    |
                    v
实现数学等价、带精确 gate 和回退路径的小 patch
                    |
                    v
干净源码构建 wheel -> 安装 -> import/JIT/kernel correctness
                    |
                    v
官方 run_throughput.sh 三档 A/B + SLA + accuracy
             /                              \
            v                                v
  收益不稳定或风险过高：回滚          收益稳定：合规审查和计分
                                             |
                                             v
                                  重放到 submit/vN 专门提交到测评机
```

项目专用 skills 对应这个闭环的四个阶段：

| Skill | 职责 | 主要产物 |
|---|---|---|
| [`scnet-vllm-baseline`](.codex/skills/scnet-vllm-baseline/SKILL.md) | 建立、恢复和验证干净 baseline | source/wheel 记录、三档 `result.json`、accuracy、canonical baseline index |
| [`dcu-hipprof-profile`](.codex/skills/dcu-hipprof-profile/SKILL.md) | 在 DCU 容器抓取短服务期 profile | 热 kernel、prefill/decode 分类、Triton/AITER/`_rocm_C`/fallback 证据 |
| [`vllm-diff-compliance`](.codex/skills/vllm-diff-compliance/SKILL.md) | 按比赛技术方案审查 patch | BLOCKER/HIGH RISK/NEEDS EVIDENCE/OK 结论 |
| [`pra26-score`](.codex/skills/pra26-score/SKILL.md) | 计算三档吞吐、SLA 和四类准确率对总分的影响 | 确定性得分与优化优先级 |

全局边界由 [`AGENTS.md`](AGENTS.md) 约束，包括分支与 remote 语义、wheel 构建契约、远端
工作区清洁检查、模型与 cache 位置、SCNet 容器操作、profile 有效性、官方 benchmark 边界、
合规红线和提交前证据。

代码通常也由AI生成后人工review，降低代码迭代成本。

## 统一演进时间线

### 2026-06-20：从通用 fork 变成比赛工程

- **`develop` · `be4fe9f`**：重写 `AGENTS.md`，首次明确仓库身份、baseline/main/develop/
  competition-ci 分支语义、remote 规则、wheel 构建路径、DCU 才能作为性能结论、比赛合规红线
  和“小 patch、可解释、可回退”的 AI 开发原则。
- **`develop` · `3d49cac`**：建立 `Source Smoke`。公共 CI 只检查必要文件和 Python 源码语法，
  不把无 PyTorch、无 DTK/HIP/DCU 的 runner 结果包装成 wheel 构建或性能结论。

### 2026-06-22：把远端 baseline 和合规流程做成可复用规则

- **`develop` · `1490e35`**：将 SCNet/Slurm 作业定位、动态 host key、队伍 home 持久化、
  `baseline_index`、localhost 代理清理、run 目录和 gfx936 实验边界写入 `AGENTS.md`。
- **`develop` · `8c3e18c`**：撤销“把模型复制到容器 `/root`”的早期建议。模型、wheel、构建
  cache 和大日志统一放在 `/public/home/xdzs2026_c118`，避免浪费容器层与共享资源。
- **`develop` · `590a0bc`**：新增 clean baseline、hipprof profile 和 diff compliance 三个
  项目 skills；长执行流程从聊天和单个文档中拆成可重复调用的操作手册。
- **`develop` · `cb46f3b` 起**：建立 `pra26/` 研究知识库，逐步记录 AITER、BLOCK_M、DUMMA、
  FlashAttention、FP8 KV、LDS、on-card profiling、PyTorch fill 和 ROCm C++ 等方向。

### 2026-06-23：从“猜热点”转向“先探针、后 patch”

- **`develop` · `022c747`**：为 Triton unified attention 增加 block/probe controls，用于确认
  实际执行路径、tile 和热点。
- **`develop` · `34e5fd1`**：根据长 prefill 中的 fill 开销，实验性避免 GDN/linear-attention
  层对即将被完整覆写的 output buffer 预清零；保留 profile/padding 场景的条件性清零。
  **说明**： fill相关开销后来被证明无效。核心原因是：一、在正确的profile指导下，FillFunctor不属于核心
  热点，占比很小，哪怕真的有优化也会被热点淹没； 二、原本的设计经过通过查询vllm社区issues/PR，可以观察到
  是有意为之──避免外部调用者默认导致错误调用。
  但是在错误的profile请求（比如将vllm整个生命周期混入）中，FillFunctor<int>会被错误识别为全局热点，
  因为vllm实现中，warmup阶段本身会创建一个64MiB buffer，用于warmup。
  这浪费了团队的时间，但也提醒团队，到底应该关注什么，不应该关注什么。

### 2026-06-24：工作流开始围绕比赛得分闭环

- **`develop` · `f35053e`**：新增 `pra26-score` skill 和确定性脚本。三档权重为 4-8K 20 分、
  8-16K 50 分、16-32K 30 分，同时纳入 TTFT/TPOT SLA 和 QA、摘要、检索、聚合四类准确率
  系数。优化排序从“单个 kernel 加速比”转向“最终比赛得分贡献”。
- **`submit/v1` · `9097a14`**：把 unified attention 的 `BLOCK_M` 变成 wheel 构建期常量；
  `build_vllm.sh` 注入 `TRITON_UNIFIED_ATTN_BLOCK_M=64`。源码默认仍是 16，因此只有按提交
  构建脚本生成的 wheel 固化为 BM64。
  **说明：`BLOCK_M` 是 Triton unified attention 的 query-row tile 参数。** kernel 会把同一
  KV head 对应的“query token × GQA query head”展平成 M 维；一个 Triton program 一次处理
  `BLOCK_M` 行，并通过 `BLOCK_Q = BLOCK_M // num_queries_per_kv` 换算出该 program 覆盖的
  query token 数。它不是 KV cache 的物理 `BLOCK_SIZE`，也不是 K/V 序列方向每轮加载多少
  token 的 `TILE_SIZE`。

  对实际 Qwen3.5-27B，`num_queries_per_kv=24/4=6`：BM16 时 `BLOCK_Q=2`，每个 program
  向前推进 2 个完整 query token，对应 12 个不重叠的 query-head rows；BM64 时 `BLOCK_Q=10`，
  向前推进 10 个完整 query token，对应 60 个不重叠的 rows。当 `BLOCK_M` 不能被
  `num_queries_per_kv` 整除时，当前 Triton kernel 没有额外的 active-row mask；因此在非末尾
  q-block 中，多出的 4 行会与下一个 q-block 的前 4 个 query-head rows 重叠计算，而不是被
  mask 屏蔽。长 prefill 下，BM64 能减少 query-block/program 数量，将这部分重叠计算的
  行占比从 `4/16` 降到 `4/64`，并让更多 query rows 复用同一轮 K/V tile、block-table
  寻址和 softmax 循环开销。自定义 HIP UA2D kernel 另外实现了 active-row mask，才会屏蔽
  这 4 个尾行。

  更大的 `BLOCK_M` 也会增加 Q、softmax 状态和 output accumulator 的寄存器压力，可能降低
  occupancy、引发 spill，或者在短 query/decode 中产生更多无效行。因此 BM64 不是通用常数，
  必须针对 `head_dim=256`、Q/KV head 比、prefill/decode 路由和 gfx936 实测。该参数作为 Triton
  `constexpr` 参与 kernel specialization；比赛平台又不能依赖启动时注入自定义环境变量，所以
  v1 通过 `build_vllm.sh` 在 wheel 构建阶段把默认值固化为 64。它只改变并行分块与资源使用，
  不改变 attention 的因果 mask、KV 内容或模型数学语义。

  在吞吐测试中，AI工作流将过长的E2E等待时间错误判定为熔断证据，但是在实际算力平台上，几十秒的E2E、
  几十分钟的编译时间是家常便饭。应该避免AI工作流靠猜来认为服务异常。

### 2026-06-25：访存实验与首次“保留有效部分、回滚噪声部分”

- **`develop` · `33a7f78`**：为 Triton UA2D 增加默认关闭的 page-local KV 地址快路径。
  当 32-token KV tile 完全位于 `BLOCK_SIZE=784` 的同一 page 时，只读一次 block table 并使用
  标量 page offset；跨 page 继续走原向量寻址。提交明确标注仍需 DCU JIT、吞吐、accuracy、
  cache、寄存器和 kernel time 证据。
  这个提交最终被证实无用。因为attention kernel真正的瓶颈是stall时间。
- **`develop` · `9ee3296`**：保存 BM64 和 metadata fill/copy 收窄实验。4-8K 10 请求记录为
  output throughput `13.23 tok/s`、TTFT P99 `2248.73 ms`、TPOT P99 `68.89 ms`，快速
  accuracy 通过。

### 2026-06-26～27：回滚无稳定收益实验，并修正 profile 方法

- **`develop` · `7116743` · revert**：metadata fill staging 的收益没有超过噪声阈值，撤销
  padding/copy 收窄和相应测试，只保留 BM64 构建期配置。
- **`develop` · `bbfc3c4` · revert**：撤销 06-23 的 GDN output 预清零收窄实验。失败或证据
  不充分的尝试留在历史中，但不包装成最终贡献。
- **`develop` · `AGENTS.md` 后续记录**：逐步确认 profile 必须只覆盖服务期；hipprof 数据库
  必须由 profiler 正常退出后提交，不能把含 journal/WAL、异常退出或混入启动期的数据当成
  有效热点证据。

### 2026-06-30：启用 gfx936 GEMV 基础路径与 ROCm 扩展

- **`develop` · `0079f69` / `submit` 同源提交 `f93a2de`**：把 `gfx936` 纳入 gfx9 skinny
  GEMM 判断，同时限制依赖 MI3XX 指令的 `wvSplitK` 只在 gfx942/gfx950 使用；单 token、
  `K<=8192` 的合适 shape 可以进入 LLMM1。
- **`develop` · `cad6390` / `submit/v1.1` · `ab94c4b`**：恢复 HIP 构建中的 `_rocm_C`
  extension。该提交属于后续自定义 ROCm kernel 能随 wheel 交付的构建前提。
- **`submit/v1.2` · `f7a8de6`**：尝试改写 LLMM1，让 Qwen MLP `down_proj` 也走该路径。
  这项尝试后来在 07-06 被完整回滚。
  **说明**：这项工作主要和decode阶段有关。Qwen的线性注意力投影阶段会命中大量的小GEMV，这些GEMV是
  通用算子，性能受限，在I/O上频繁等待。通过切换到LLGemm1，这是skinny GEMM的一部分，它在N=1上有特
  化的效果，最终通过吞吐数据证明它能比通用GEMM本身有更好的效果。
  尝试过在调用层融合in_proj_qkvz和in_proj_ba两个算子，失败（性能倒退）。可能是因为这种融合破坏了
  `16384`这种规整的矩阵宽，导致尾部开销>融合收益。
  当然，这些算子本身。事实上后面也试过将n=1的tensor给padding到n=16，空算十五倍看看能不能缓解I/O难题；
  也试过padding 其他n看看能不能选到更好的kernel，microbench阶段确实有成绩，但是实际vllm未命中（矩
  阵形状错配，后来因为时间不充裕也没有继续尝试）

### 2026-07-04～05：develop 上完成一次自定义 UA2D 的完整实验链

- **`develop` · `276bd7d`**：microbench 指出 Q@K 的 LDS bank conflict。实验将 K 转置存储为
  `[TILE_SIZE, HEAD_SIZE]` 并使用 `KT_STRIDE=260`，记录到 `BANK_CF 32768 -> 8192`、
  LDS 指令 `20480 -> 6144`、Q@K `97.6 -> 59.1 us`。Triton `tl.trans` 未生成预期 LDS
  布局，因而转为自定义 HIP/DUMMA kernel。
- **`develop` · `a8fe6dd`**：将完整 causal attention、online softmax、P@V 和 GQA packing
  接入 `_rocm_C`，新增 `--use-custom-ua2d` 显式开关，默认不替换 Triton。
- **`develop` · `6c821e0`、`d1cc39e`**：增加 gfx936 构建保护并使用本地可编译的 DUMMA
  适配头，区分 backend enablement/build fix 与真正 kernel 优化。
- **`develop` · `6202a6c`**：记录 custom UA2D 未命中的具体原因，避免“开了开关就假定生效”。
- **`develop` · `6500ce1`**：允许目标模型的 `num_queries_per_kv=6`，并处理 BM64 下只有
  前 60 行有效、尾部 4 行需要 mask 的情况。
- **`develop` · `a57eda9`、`1affc02`**：修复 tile loop 中重复加载 Q、`BLOCK_M` mask、循环
  和路由细节；custom kernel 仍严格限制 ROCm、BF16、head_dim=256、Q/KV=6、block_size=784
  且无 FP8/ALiBi/sink/sliding-window 等语义扩展的场景。
- **`develop` · `60105f1`**：按 Qwen decode projection shape 微调 LLMM1
  `rows_per_block`。这是 develop HEAD 上最后一个性能提交。

  **说明**：自定义UA2D的尝试无疑是失败的。首先是试图仿照triton unified attention来写，但是triton
  本身就已经能编译到相对底层，而且后者真正卡住的地方是频繁的stall时间。通过切换到Flash Attention，
  在定向的纯文本请求下去使用计算开销更小、数据等待时间更短从而更快的算子，带来的收益要远大于盲目custom
  kernel。

### 2026-07-06：submit 线转向 Strided-K

  **说明**：由于期末考试压力巨大且时间紧张，这一阶段几乎不再使用develop和submit双轨，有优化点会
  制作成一版submit并直接提交到评测机尝试。

- **`develop` · `74c992a`、`a9760fe`**：补齐官方 `run_throughput.sh` 边界、run 目录提前
  告知、wheel 路径记录和远端操作约束。`a9760fe` 是本文采用的 develop 工作流快照。
- **`submit` · `61ab7c8` · revert**：完整撤销 v1.2 的 LLMM1 `down_proj` 强制路由。
- **`submit` · `87eae58`**：新增 Strided-K 单 token GEMV。每个 block 负责若干输出行，线程
  沿 K 做 grid-stride，改善合并访存和 occupancy，并支持 `K>8192`。提交记录目标 projection
  带宽相对 LLMM1 提升约 7%～20%，由 `VLLM_ROCM_STRIDED_GEMV` 控制且默认开启。
- **`submit` · `9b1adfa`**：保留 LLMM1 fallback，并按 Qwen shape 选择
  `rows_per_block`。
- **`submit` · `d10bbd2`**：3D decode attention 的 softmax 并行分段由 16 调为 32，目标是
  改善 24K～32K 长上下文 decode 的并行度。
- **`submit/v2.0` · `0a27664`**：Triton unified attention prefill 的 `TILE_SIZE` 默认从
  32 调为 16，并允许 `VLLM_PREFILL_ATTN_TILE_SIZE` 覆盖。gfx936/head_dim=256 下用于降低 LDS
  压力、提高 occupancy。

### 2026-07-07：full-attention prefill 改走连续 FlashAttention

- **`submit/v2.1` · `0ff997e`**：Qwen full-attention prefill 默认尝试非 paged
  `flash_attn_varlen_func`。当前 chunk 做 causal FA；paged cache 中的历史 K/V gather 到连续
  workspace 后做 prefix FA；各段通过 LSE 合并。纯 decode 仍走 Triton paged attention。
- 提交标题写“适配 AITER flashattention”，但最终实现明确使用普通 `flash_attn`，不是 AITER，
  也不是 `vllm.vllm_flash_attn`。不可导入或不满足 dtype/head/attention 语义时回退 Triton。

### 2026-07-08：旁支 MLP padding 与主线 Strided-K 调参

- **`submit/v2.2` · `4045de9` · side branch**：从 v2.1 单独分叉，尝试把小 N 的旧 Qwen
  MLP GEMM padding 到 2/4/8 行后调用 rocBLAS。它不是 v2.9 的祖先，而且 shape 是旧的
  `[24576,4096]` / `[4096,12288]`，不会命中实际 27B 的 `[34816,5120]` /
  `[5120,17408]`。不能算作 v2.9 优化。
- **`submit` · `ee1e066`**：Strided-K 扩展到大 K，并继续按 K/M shape 调整
  `rows_per_block`。

### 2026-07-10：修复 BF16 GEMV 累加精度

- **`submit` · `2f151cc`**：gfx936 不适合用模拟 packed BF16 FMA 做长 reduction。新路径将
  BF16 pair 转 FP32，使用 FP32 `fmaf` 累加，最终再转 BF16；同时增加 Qwen projection 和
  `down_proj` accuracy 测试。该改动既减少 BF16 模拟开销，也让结果更接近 rocBLAS。

  **说明**：这一阶段的优化围绕继续优化decode展开。profile证据显示prefill本身（TTFT指标）随着上下
  文数量呈现线性变化，而最后吞吐依旧由decode去主导。decode还是卡在线性注意力各种投影调用产生的小GEMM
  而不是triton unified attention 3d。所以我们利用AI工作流，在期末复习的闲暇，继续深入研究这些小
  GEMM到底应该怎么调整，才能充分利用带宽。

### 2026-07-11：decode MLP 融合与 4096-token prefill 专用解

- **`submit` · `d0b1ce1`**：将 3D decode softmax segments 改成环境变量
  `VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS`，最终默认 256，要求为正的 2 次幂。提交标题
  写 `MIN_LAUNCH_GRID_SIZE_2D=128`，但该常量原本就是 128；实际变化是 segments。
- **`submit/v2.4` · `be6637e`**：`K=5120` 的 Strided-K 使用 640 线程，使 640 个
  16-byte chunk 基本一线程一个。提交标题写 `NUM_THREADS=5120`，实际代码并非启动 5120
  个线程。
- **`submit` · `13639b4`**：融合 Qwen3.5 单 token `gate_up` GEMV 和 SwiGLU。直接输出
  `[1,17408]`，避免物化 `[1,34816]` 的完整 gate/up tensor 和单独 activation kernel；仅在
  TP=1、BF16、无量化、无 LoRA、目标 shape 下启用，测试要求与未融合路径逐 bit 一致。
- **`submit` · `7cee783`**：首次加入 fused K/V gather，试图一次收集 paged K 和 V。
- **`submit` · `da496ba`**：为 gfx936 BF16 prefill `down_proj` 精确 shape
  `[4096,17408] x [17408,5120]` 固定 rocBLAS solution 20980，并增加逐 bit 一致性测试。
- **`submit` · `4490eaf` · revert**：完整撤销首次 fused K/V gather，包括 kernel、binding、
  测试和路由；后续改用连续 KV 镜像解决重复 gather。

### 2026-07-12：decode 专用 shape 与连续 KV 镜像

- **`submit` · `4d411c5`**：为 BF16 decode `down_proj [5120,17408]` 增加 rows=1、1024
  线程专用 Strided-K kernel，提高长 K 下的独立 HBM 请求数。
- **`submit` · `eb40282`**：把 Qwen3.5 LM head `[248320,5120]` 作为 gfx936 BF16 特例
  路由到 Strided-K rows=2；其他无关的大词表 shape 仍回退原实现。
- **`submit` · `f958b91`**：为单请求 Qwen3.5 full-attention prefill 的每一层维护连续 K/V
  镜像，默认容量 32K、full-KV FA 阈值 30K。30K 内对完整连续 KV 做一次 causal FA；30K～32K
  从镜像读取历史，继续 prefix/suffix FA + LSE merge；不满足条件时回退通用 gather 路径。

### 2026-07-13：补齐 gate_up GEMM 与 48 层 GDN prefill

- **`submit` · `3a2d060`**：为 gfx936 BF16 prefill `gate_up` 精确 shape
  `[4096,5120] x [5120,34816]` 固定 rocBLAS solution 20981，并与默认 linear 做逐 bit
  一致性测试。
- **`submit/v2.9` · `97869b0`**：针对实际 GDN shape `(H,K,V,BT)=(48,128,128,64)` 固定
  gfx936 Triton 配置，优化 `chunk_fwd_h`、`chunk_fwd_o` 和 `recompute_w_u`；支持直接写入
  预分配 `core_attn_out`，消除一次临时 output 与 copy；其他 shape/平台继续走原 autotune，
  避免短序列 warmup 污染长上下文配置。

  **说明：这项优化针对 Qwen3.5-27B 占比最高的 GDN prefill 主路径。** 模型共有 64 个
  decoder layer，其中 48 层是 linear-attention/GDN，只有 16 层是 full attention。因此，
  即使单次 GDN kernel 的收益不如大 GEMM 显眼，相关开销也会在一次 forward 中重复 48 次。
  需要区分两个容易混淆的“48”：模型有 48 个 GDN layer；调优条件中的 `H=48` 则表示每一层
  GDN kernel 的 48 个 value heads，并不是层数。`K=128`、`V=128` 分别是 key/value head
  dimension，`BT=64` 是 GDN chunk 的 token tile 大小。

  一次 native Triton/FLA GDN prefill 并不是单个 attention kernel，而是依次执行
  `chunk_local_cumsum -> chunk_scaled_dot_kkt_fwd -> solve_tril -> recompute_w_u_fwd ->`
  `chunk_gated_delta_rule_fwd_h -> chunk_fwd_o`。其中 `recompute_w_u_fwd` 生成 WY 表示所需的
  `w/u`，`chunk_fwd_h` 按 chunk 推进 recurrent state 并产生 `v_new`，最后 `chunk_fwd_o`
  将 query、chunk 内 attention 和历史 state 合成为本层输出。v2.9 主要优化后三个在 profile
  中反复出现、且能够保持算法不变的 Triton kernel。

  原实现对这些 kernel 使用 autotune，但 autotune key 主要包含 `H/K/V/BT`，没有把完整序列
  长度 `T` 作为调参维度。短序列 warmup 和正式 16K～32K prefill 的 `H/K/V/BT` 完全相同，
  因而短序列选出的配置可能被缓存并复用于长上下文；同时每个 fresh cache 都需要承担候选配置
  试跑成本。v2.9 在且仅在 `gfx936 + (H,K,V,BT)=(48,128,128,64)` 时绕过 autotune wrapper，
  直接调用底层 Triton kernel，并固定为针对目标 shape 选出的配置：

    - `chunk_fwd_h`：`BV=32`、`num_warps=8`、`num_stages=1`；把 128-wide value dimension
      分成 4 个 value tiles，以更多 program/warps 展开 state 更新，同时避免过深 pipeline
      增加寄存器和 shared-memory 压力。
    - `chunk_fwd_o`：`BK=32`、`BV=128`、`num_warps=2`、`num_stages=1`；`BV=128` 一次覆盖
      完整 value dimension，使 grid 的 value 方向只有一个 program，减少重复的 query/state
      读取和边界处理。
    - `recompute_w_u_fwd`：`BK=64`、`BV=128`、`num_warps=2`、`num_stages=2`；一次覆盖完整
      value dimension，避免原 `BV=64` 下同一 chunk/value 工作被拆成两份。

  另一项收益来自 output 直写。原路径先在 `chunk_fwd_o` 中执行 `torch.empty_like(v)` 创建临时
  `o`，48 个 GDN 层各自产生一个中间 tensor，随后再执行
  `core_attn_out[:num_actual_tokens] = o.squeeze(0)`。v2.9 将已经由 model runner 预分配的
  `core_attn_out` 以可选 `out` 参数一路传入 GDN custom op 和 `chunk_fwd_o`，让 Triton kernel
  直接写最终地址，从而消除临时 output 的分配和一次 device-to-device copy。该直写只在普通
  non-spec prefill、无需按 speculative mask 合并输出时启用；存在 spec/mixed sequence 时仍走
  原来的临时 tensor 和 merge 路径。

- **`submit/v3.0` · `8bf1196`、`f916d70`、`442994b`**：按 NT、head 数和 K/V 维度泛化 gfx936
  长上下文 GDN prefill 配置选择，复用 chunk indices，并预热长上下文 specialization，
  避免首个长请求触发 Triton JIT 而导致 SLA 熔断。
