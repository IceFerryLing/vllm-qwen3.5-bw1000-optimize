# Qwen3.5 vLLM Contest Fork

本仓库用于 2026 先导杯「基于国产加速卡的千问大模型推理服务优化」。基础代码来自
SourceFind/OpenDAS vLLM v0.18.1，目标环境是国产 DCU/BW1000（`gfx936`）上的
Qwen3.5-27B BF16 单卡在线推理。

本项目的演进分为两条相互关联、但不能混为一谈的线路：

- `develop` 是日常研究与 AI 辅助优化工作流的孵化线。它保存 profiling 探针、实验 patch、
  失败回滚、比赛研究笔记、`AGENTS.md` 约束和项目专用 Codex skills。
- `submit/vN` 是比赛提交线。从 baseline 重放已经筛选过的最小优化 patch；其中
  `submit/v2.9` 是本文记录的当前优化版本。

因此，`develop` 上出现过的代码不一定进入 `submit/v2.9`，`submit/v2.9` 的后续优化也不一定
回合并到 `develop`。下面的时间线同时展示“工作流如何形成”和“最终性能 patch 如何演进”，并
明确标注回滚与旁支。

## 版本基准

| 角色 | Git ref | Commit | 说明 |
|---|---|---:|---|
| 官方代码基线 | `target/submit/baseline` / `v0.18.1` | `01d6aad` | SourceFind/OpenDAS v0.18.1 比赛快照 |
| AI 工作流孵化线 | `origin/develop` | `a9760fe` | 截至 2026-07-06 的工作流、研究材料和实验集成线 |
| 当前提交优化线 | `target/submit/v2.9` / `submit/v2.9` | `97869b0` | 截至 2026-07-13，从 baseline 线性演进的比赛优化版本 |

`submit/v2.9` 相对 baseline 有 23 个提交，净修改 20 个代码/测试/构建文件，约
`+1629/-106`。`develop` 相对 baseline 的提交更多，但其中大量内容是文档、skills、研究记录、
实验 kernel 和已回滚尝试，不能用提交数量衡量最终性能贡献。

## 实际目标模型

性能分析与 shape gate 必须以比赛 checkpoint 的 `config.json` 为准，不能用配置类的旧默认值
代替实际模型。当前目标模型的文本侧关键配置是：

- `hidden_size=5120`
- `num_hidden_layers=64`
- `num_attention_heads=24`
- `num_key_value_heads=4`
- `head_dim=256`
- `intermediate_size=17408`
- `full_attention_interval=4`

Qwen3.5-27B 是 hybrid decoder：64 层中有 16 个 full-attention 层和 48 个
linear-attention/Gated Delta Net（GDN）层。TP=1 时：

- full attention 是 GQA，`num_queries_per_kv=6`；packed `qkv_proj` 输出宽度是
  `q + gate + k + v = 14336`；
- GDN `qkvz` 输出宽度是 `16384`，`out_proj` 是 `6144 -> 5120`；
- dense MLP 是 `gate_up: 5120 -> 34816`、`down: 17408 -> 5120`；
- LM head 是 `[248320, 5120]`；
- 比赛 attention/KV page 形状可能使用 `BLOCK_SIZE=784`。

这些事实决定了后文所有 attention、GEMV、GEMM 和 GDN 专用路径的触发条件。

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
                                从 baseline 重放到 submit/vN
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
  实际执行路径、tile 和热点，而不是只凭源码静态判断。
- **`develop` · `34e5fd1`**：根据长 prefill 中的 fill 开销，实验性避免 GDN/linear-attention
  层对即将被完整覆写的 output buffer 预清零；保留 profile/padding 场景的条件性清零。

### 2026-06-24：工作流开始围绕比赛得分闭环

- **`develop` · `f35053e`**：新增 `pra26-score` skill 和确定性脚本。三档权重为 4-8K 20 分、
  8-16K 50 分、16-32K 30 分，同时纳入 TTFT/TPOT SLA 和 QA、摘要、检索、聚合四类准确率
  系数。优化排序从“单个 kernel 加速比”转向“最终比赛得分贡献”。
- **`submit/v1` · `9097a14`**：把 unified attention 的 `BLOCK_M` 变成 wheel 构建期常量；
  `build_vllm.sh` 注入 `TRITON_UNIFIED_ATTN_BLOCK_M=64`。源码默认仍是 16，因此只有按提交
  构建脚本生成的 wheel 固化为 BM64。

### 2026-06-25：访存实验与首次“保留有效部分、回滚噪声部分”

- **`develop` · `33a7f78`**：为 Triton UA2D 增加默认关闭的 page-local KV 地址快路径。
  当 32-token KV tile 完全位于 `BLOCK_SIZE=784` 的同一 page 时，只读一次 block table 并使用
  标量 page offset；跨 page 继续走原向量寻址。提交明确标注仍需 DCU JIT、吞吐、accuracy、
  cache、寄存器和 kernel time 证据。
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

### 2026-07-06：develop 阶段收束，submit 线转向 Strided-K

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

## submit/v2.9 最终生效的优化

| 路径 | 最终优化 | 主要触发条件/回退 |
|---|---|---|
| Wheel 构建 | unified attention `BLOCK_M=64` 构建期注入 | 必须使用 `build_vllm.sh` 或显式设置构建变量；普通源码默认是 16 |
| Full-attention prefill | 非 paged FlashAttention、历史分段 LSE merge | ROCm、BF16/FP16 KV、支持的 head_dim 和普通 causal decoder 语义；否则回退 Triton |
| 单请求 full-attention prefill | 每层连续 KV 镜像；30K 内 full-KV FA | Qwen 24Q/4KV/head_dim256，默认镜像容量 32K；失配、多请求或超容量回退 gather |
| Triton prefill fallback | `TILE_SIZE=16` | 可用 `VLLM_PREFILL_ATTN_TILE_SIZE` 恢复/调节 |
| Full-attention decode | 3D softmax 默认 256 segments | 可用环境变量调整；不改变 2D/3D 原路由语义 |
| 通用单 token projection | Strided-K GEMV + shape 化 rows/thread | ROCm gfx9、FP16/BF16、N=1、无 bias、K 对齐；否则回退 LLMM1/rocBLAS/linear |
| Dense MLP decode gate_up | GEMV + SwiGLU 融合 | Qwen3.5-27B、TP=1、BF16、无量化、无 LoRA、单 token；否则走原 MLP |
| Dense MLP decode down | `[5120,17408]` rows=1/1024-thread GEMV | gfx936 BF16 精确 shape；否则通用 Strided-K 或原 linear |
| Decode LM head | `[248320,5120]` Strided-K rows=2 | gfx936 BF16 精确 shape；其他大词表保持原路由 |
| 4096-token MLP prefill | rocBLAS solution 20981 gate_up、20980 down | gfx936 BF16、二维连续 tensor、无 bias、精确 4096-token shape；否则原 GEMM |
| 48 层 GDN prefill | 固定真实 shape 的 Triton 参数，output 直写 | gfx936 `(48,128,128,64)`；其他平台/shape 保留 autotune |

## 未进入 v2.9 的尝试

以下提交存在于历史，但不属于 v2.9 最终净优化：

- v1.2 强制 `down_proj` 走 LLMM1：已由 `61ab7c8` 回滚；最终改用 Strided-K 专用 kernel。
- GDN/linear-attention output 清零收窄：只在 develop 实验，已由 `bbfc3c4` 回滚。
- metadata fill/copy staging：只在 develop 实验，收益未超过噪声，已由 `7116743` 回滚。
- 首版 fused K/V gather：已由 `4490eaf` 回滚；最终用连续 KV 镜像降低重复 gather。
- `submit/v2.2` 小 N MLP padding：独立旁支，不是 v2.9 祖先，且使用旧模型 shape。
- develop 自定义 UA2D：保留为显式开关的实验 kernel，没有进入 `submit/v2.9`。

## 性能证据的解释边界

历史提交和研究记录中包含若干 kernel/microbench 数据，例如：

- Strided-K 对部分 decode projection 的带宽提升记录约为 7%～20%；
- `K=5120` 的 640-thread 调参对部分大 projection 的记录约为 9%～10%；
- Triton prefill TILE 32 -> 16 的 kernel 级记录约为 2.6～2.8 倍；
- develop UA2D 的 Q@K microbench 记录为 `97.6 -> 59.1 us`；
- 3D decode softmax 16 -> 32 的历史 kernel 记录在 24K～32K 约快 26%～30%。

这些数字来自不同阶段、不同 kernel 或短样本，不能相加，也不能直接等同于 v2.9 的官方端到端
吞吐提升。尤其最终 v2.9 使用 256 个 decode softmax segments，而不是中间实验的 32。

正式结论必须来自同一官方 workload 下的：

1. 干净 baseline 与候选 wheel；
2. `run_throughput.sh` 的 4-8K、8-16K、16-32K 三档原始 `result.json`；
3. TTFT P99、全局 TPOT P99 和完成率 SLA；
4. QA、摘要、检索和聚合四类 accuracy；
5. 明确的源码 commit、构建命令、wheel 路径、服务参数和环境记录。

另外，2.9 的连续 KV 镜像和 256-segment decode workspace 会使用额外显存。它们可能降低 paged
KV cache 容量或最大并发，必须结合服务启动日志中的 KV cache token 数和三档吞吐结果评估，
不能只看 attention kernel 时间。

## 构建与验证契约

提交版本必须在比赛容器中通过 wheel 路径交付，不依赖 `PYTHONPATH`、editable install 或手工
修改 `site-packages`：

```bash
set +u
source /opt/dtk/env.sh
set -u
export LD_LIBRARY_PATH=/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/dcc/lib:${LD_LIBRARY_PATH:-}

./build_vllm.sh
pip install --force-reinstall dist/vllm-*.whl --no-deps
```

随后至少完成：

1. import smoke 和目标 Triton/HIP JIT probe；
2. 新增 kernel 的 correctness/accuracy 测试；
3. 官方 `run_throughput.sh` 三档 A/B；
4. 官方 accuracy；
5. 对最终 diff 做比赛合规审查。

性能、构建和兼容性结论只来自官方 DCU 容器/算力节点。登录节点、本地开发机和公共 GitHub
Actions 只能做源码、Git、文档和语法层面的检查。

## 分支与提交边界

- `v0.18.1`：只跟踪 baseline，不直接开发。
- `develop`：接收实验 patch、诊断、研究材料和工作流改进；不等同于评测提交。
- `exp/*`：实验线，默认不作为正式提交代码。
- `main`：稳定集成线，只接受通过基本构建、正确性和合规检查的改动。
- `submit/vN`：从 baseline 重放最小提交 patch；除非任务明确要求，不直接修改。
- `target`：官方目标 GitLab remote，只接受 `submit/*`；Agent 不向该 remote push，由队员手动
  完成最终推送。

所有 AI 辅助生成或修改的代码都必须由提交者逐行 review，并能够解释命中路径、模型 shape、
数学语义、回退条件、构建方式、性能证据和合规风险。
