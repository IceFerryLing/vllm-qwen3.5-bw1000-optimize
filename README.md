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

调研技术方案后，团队认为vllm 0.18.1 是一个非常关键的信息，这个版本号非常新，以至于团队相信，
大部分通用性优化vllm社区都应该已经制作。

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
手写custom kernel。事实上JIT技术也能拿到更多平台运行时信息，手写算子也难以调参，更不能搬迁NVIDIA
经验。

所以关键点是：1.怎么让Triton kernel跑的更快，在这个模型和平台上表现更好？2.能不能通过
custom kernel实现自定义快速算子（后来证明难度很高且投入产出比低）；3.能不能想办法开启因为各种原因最
终没有被选中的快速算子，从而绕开某些算子过于通用而不快的问题？

对于1，我们通过利用AI工作流调整参数解决，比如调参BLOCK_M、TILE_SIZE，以及通过微调相关算子代码解决
，比如线性注意力区投影的各种GEMV算子代码微调；对于3，我们先是通过将gfx936加入编译白名单，从而让其能
够命中 LLGemm 算子，然后在 Triton backend 内为满足约束的 full-attention prefill 增加
non-paged FlashAttention 快路径：当前 chunk 执行 causal FlashAttention，历史 K/V 从 paged
cache gather 到连续 workspace 后执行 prefix attention，各段结果再通过 LSE merge 合并；
decode 仍保持 Triton paged attention 路径。这减少了实际代码量，同时带来了巨大的吞吐提升，
因为这个系统本来就拥有由于模型形状和平台检查而未被选中的快速路径。

所有优化都和profile证据息息相关，尽管对hipprof工具的错误使用也产生了错误决策，但我们团队依然坚持
以实际证据为主。事实上，我们团队最终能摸清到底哪些路径主导了整个流程，正是依靠AI profile工作流传回
来的相关信息。

## AI 辅助优化工作流 （ AI披露 ）

`develop` 主要把 AI 辅助优化固化成了一套流程：

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
| [`scnet-vllm-baseline`]| 建立、恢复和验证干净 baseline | source/wheel 记录、三档 `result.json`、accuracy、canonical baseline index |
| [`dcu-hipprof-profile`] | 在 DCU 容器抓取短服务期 profile | 热 kernel、prefill/decode 分类、Triton/AITER/`_rocm_C`/fallback 证据 |
| [`vllm-diff-compliance`] | 按比赛技术方案审查 patch | BLOCKER/HIGH RISK/NEEDS EVIDENCE/OK 结论 |
| [`pra26-score`]| 计算三档吞吐、SLA 和四类准确率对总分的影响 | 确定性得分与优化优先级 |

全局边界由 [`AGENTS.md`](AGENTS.md) 约束，包括分支与 remote 语义、wheel 构建契约、远端工作区清洁检查、模型与 cache 位置、SCNet 容器操作、profile 有效性、官方 benchmark 边界、合规红线和提交前证据。代码部分也由AI生成后人工review，降低代码迭代成本。

## 各项优化措施及其对性能的提升

### 软硬件约束与热点来源

Qwen 官方模型卡给出的 Qwen3.5-27B 语言模型结构是 64 层
`16 × (3 × Gated DeltaNet + 1 × Gated Attention)`：48 层是 GDN 线性注意力，
16 层是 full attention。GDN 使用 16 个 Q/K heads、48 个 V heads，head dimension
均为 128；full attention 使用 24 个 Q heads、4 个 KV heads，head dimension 为 256。
模型 hidden size 为 5120，MLP intermediate size 为 17408，padded vocabulary 为
248320。官方模型支持更长上下文，但本次比赛工作负载只覆盖到 32K。

这些形状把热点分成三组：48 次重复执行的 GDN chunk prefill；16 层在非典型
`BLOCK_SIZE=784` paged KV 上运行的 GQA full attention；以及每层 MLP、GDN 投影和
LM head 中的 GEMM/GEMV。Prefill 主要处理大矩阵和长序列 attention，decode 则反复执行
`N=1` 的带宽受限 GEMV。

因此，考虑方向为：

- 当前 kernel 按 wave64 组织 shuffle、reduction 和线程块，线程数必须考虑 64-lane wave 对齐。
- `N=1` GEMV 算术强度低，关键是形成合并的 16-byte 权重读取并产生足够多的独立 HBM 请求。
- Unified Attention 的 Q、K/V tile 和 FP32 accumulator 会共同占用寄存器与 LDS；tile 过大
  会降低驻留 wave 数，难以隐藏 paged-KV 访存停顿。
- 当前实现记录 gfx936 没有适合该路径的 packed BF16 dot/FMA；模拟 `__hfma2` 会增加指令并在
  每个 pair 后舍入，因此 BF16 GEMV 改为 FP32 `fmaf` 累加。
- Triton 已能在平台上生成矩阵计算代码，主要问题是针对真实shape 选择 block/warp/stage、避免短序
  列 autotune 污染，以及在首个真实请求前完成 JIT。


### Full-attention prefill

1. **Unified Attention `BLOCK_M=64`。**
   [`unified_attention`](vllm/v1/attention/ops/triton_unified_attention.py) 将同一 KV head 下的
   “query token × GQA query head”展平到 M 维，再通过
   `BLOCK_Q = BLOCK_M // num_queries_per_kv` 决定一个 program 覆盖的 query token 数。
   对目标 `24Q/4KV`，提高 `BLOCK_M` 能在长 query 下减少 q-block program 数，让更多 query
   rows 复用 block-table 寻址和 K/V tile 循环；代价是更大的 Q/softmax/accumulator 状态及
   更高寄存器压力。`d56a55b` 后该值在运行时读取，默认 64。gfx936 单卡微基准表明，这一选择
   明确面向长 query：q_len=4096 时，BM32→BM64 在 4K/16K/32K context 分别由
   **7.462→4.734 ms、48.890→30.440 ms、103.748→64.627 ms**，用时降低
   **36.6%～37.7%**；q_len=1 的纯 2D UA 中则是 BM32 最快，4K/16K/32K 分别为
   **0.514/2.021/4.031 ms**，BM64 为 **0.573/2.251/4.485 ms**。因此 BM64 是纯
   Triton 长 prefill 的 shape specialization，不是 decode 的通用最优值。满足条件的 Qwen3.5
   prefill 默认走后述 FlashAttention 路径，BM64 主要覆盖 Triton 回退和其他 UA prefill 场景。

2. **Prefill `TILE_SIZE=16`。**
   `_get_tile_size` 为非 Gemma 的 Unified Attention prefill 返回环境变量配置。源码记录
   gfx936、`head_dim=256` 下，32-token tile 会因 LDS 压力只保留约 1 wave/SIMD；降到 16 后
   可达到约 2 waves/SIMD，目标 kernel 实测约 **2.6～2.8×**。这项数据只代表纯 Triton
   Unified Attention kernel，不代表默认 FA prefill 的端到端收益。

3. **把 full-attention prefill 路由到连续 FlashAttention。**
   关键路径为
   `TritonAttentionImpl.forward -> _forward_fa_prefill -> flash_attn_varlen_func -> merge_attn_states`。
   当前 chunk 直接在连续 K/V 上做 causal FA；历史 K/V 从 paged cache 分块 gather 到连续
   workspace，做 non-causal prefix attention，再通过 LSE 合并。它既绕开 gfx936 上有数值问题的
   paged FA 路径，也避免让通用 Unified Attention 直接承担目标纯文本 prefill。该路由只在 ROCm、
   FP16/BF16、普通 decoder full attention、无 sliding-window/ALiBi/sink/softcap/MM prefix 等
   语义扩展时启用；依赖不可导入或条件不满足时回退 Triton。4096-token query chunk 下，history
   从 0 增至 32K 时，该路径相对 paged UA 的总延迟降幅稳定在 **60.1%～65.1%**。

4. **单请求连续 K/V 镜像与混合 FA 路由。**
   `_forward_fa_prefill_mirror` 为目标 `24Q/4KV/head_dim=256` 的每个 full-attention 层维护
   连续镜像，chunked prefill 只追加本轮 K/V，避免每轮重新 gather 全部前缀。序列不超过
   30K 时执行一次 full-KV causal FA；30K～32K 时执行当前 chunk causal FA、历史 non-causal FA
   和 LSE merge；超过容量或状态不连续时回退通用 gather 路径。BF16 下 32K 镜像约为
   **128 MiB/层**，按 16 个 full-attention 层约 **2 GiB/卡**，这是用显存换重复 gather 的减少。
   8K～32K 的稳态 chunked-prefill 微基准显示，镜像总延迟降低 **7.6%～15.6%**；追加当前
   4096-token K/V 只需约 **0.031 ms**，收益主要来自消除随 history 增长到
   **0.412～2.807 ms** 的 paged-cache gather。

### Decode attention

低并发 decode 的 2D grid 只有 `num_sequences × num_kv_heads` 量级，目标模型只有 4 个 KV heads，
长上下文时无法充分利用。3D kernel 将 KV 序列切成多个 segment，分别写出局部
output/max/expsum，再由 `reduce_segments` 合并。

实机 sweep 覆盖 segments=16/32/64/128/256。4K～16K 的 3D+reduce 总延迟都在
**259～264 µs** 左右；24K 时 16→256 由 **362.512→263.326 µs**，用时降低 **27.36%**；
32K 时由 **469.470→263.688 µs**，用时降低 **43.83%**。256 段的
output/max/expsum workspace 为 **193.5 MiB**，相对 16 段的 **12.094 MiB** 多用
181.406 MiB，以固定 workspace 换取长上下文并行度。32K 下 32/64/128 段分别为
348.634/479.021/288.647 µs，说明 segment 数与每段工作量、reduce 开销和 gfx936 驻留能力
共同决定结果，256 是目标 24Q/4KV/head_dim=256 shape 的实测最优点。

### Decode GEMV 与 MLP

`f93a2de` 先让 gfx936 进入 gfx9 skinny-GEMM 路由，`ab94c4b` 恢复 `_rocm_C` 扩展构建，
为后续 kernel 随 wheel 交付提供基础。这两项属于使能条件，不单独归因性能。

1. **Strided-K `N=1` GEMV。**
   路径为 `Qwen3.5 Linear -> rocm_unquantized_gemm -> shape gate -> LLMM_StridedK`。
   相比 LLMM1 按 K 增长线程数，Strided-K 让固定规模线程沿 K 的 8-element/16-byte chunk
   grid-stride，改善合并访存和 occupancy，并支持 `K>8192`。提交和源码记录在 gate_up、
   qkvz、qkv、LM head、o_proj 等目标投影上，相对 LLMM1 的 HBM 带宽提升约 **7%～20%**。
   `rows_per_block` 按 M/K shape 选择，未命中时仍回退 LLMM1 或通用 linear。早期测试还记录
   `K=17408` down projection 的 rocBLAS 路径已达到约 **91% 峰值带宽**，因此最初没有用通用
   Strided-K 强行替换；后续才为该精确 shape 增加 rows=1/1024-thread 专用实现。

2. **BF16 改用 FP32 累加。**
   gfx936 路径把 BF16 pair 转为 FP32 后逐项 `fmaf`，替代由 `__hip_bfloat162` 模拟的 packed
   BF16 FMA。这既减少模拟指令，也避免每两个乘积就舍入；测试用更严格阈值与 rocBLAS 结果比较。
   在 `[16384,5120]` 上，当前 Strided-K 相对 rocBLAS 的 rel-L2 为 **1.88e-5**，旧 LLMM1
   为 **3.47e-3**；在 `[8192,5120]` 上分别为 **3.05e-6** 与 **3.45e-3**，验证了长 reduction
   中 FP32 累加的精度价值。

3. **`K=5120` 的 640-thread 配置。**
   5120 个 BF16 元素恰好形成 640 个 16-byte chunk；在 wave64 上使用 640 线程可让每个线程
   读取一个连续 chunk，避免通用 128-thread 配置的五轮 grid-stride。源码记录 gate/qkv 大投影
   约有 **9%～10%** 提升。standalone kernel 在 M=16384/34816/248320 上相对 128 threads
   分别由 **142.245→128.453 µs、299.552→270.662 µs、2.103→1.910 ms**，用时降低
   **9.2%～9.7%**，输出逐 bit 相同。

4. **融合 gate_up GEMV 与 SwiGLU。**
   `Qwen3_5MLP.forward -> rocm_unquantized_silu_mul -> LLMM_SiluMul` 同时计算对应 gate/up 行，
   直接输出 `[1,17408]`，避免物化 `[1,34816]` gate_up tensor 和单独 activation kernel。
   仅在 TP=1、BF16、无量化、无 LoRA、目标 shape且连续布局下命中。目标 shape 上，未融合的
   Strided-K+SwiGLU 为 **261.489 µs**，融合后为**247.957 µs**，用时降低 **5.17%**，结果逐 bit 相同。

5. **Decode down projection 专用 kernel。**
   对 `[M,K]=[5120,17408]` 使用 rows=1、1024 threads，每个 block 负责一个输出行并遍历
   2176 个连续 16-byte chunk。它用更多独立 HBM 请求替代通用 rows=8/128-thread 配置，同时
   避免重复读权重。已构建 rows=8 参考实现为 **151.280 µs**，rocBLAS 为
   **150.309 µs**；当前源码将该精确 shape 收窄到 rows=1/1024-thread 实现，避免把这一策略
   扩散到其他 K>8192 shape。

6. **LM head 专用路由。**
   将 gfx936 BF16 `[248320,5120]` 从 LLMM1 rows=8 改为 Strided-K rows=2；其他大词表
   shape 继续走原路径，避免把单一模型的结论泛化。目标 LM head 上 rocBLAS、LLMM1 rows=8、
   Strided-K rows=2 分别为 **1.905/2.403/1.754 ms**；rows=2 相对 rocBLAS 用时降低 **7.93%**，
   相对 LLMM1 用时降低 **27.03%**。

### Prefill MLP GEMM

4096-token prefill 的矩阵已经足够大，手写 GEMV 不再合适，因此直接使用 rocBLAS 的目标
solution：down projection `[4096,17408] × [17408,5120]` 固定 solution 20980，gate_up
`[4096,5120] × [5120,34816]` 固定 solution 20981。路由严格限制到 gfx936、BF16、无 bias、
连续 tensor 和精确 shape，并与默认 `torch.nn.functional.linear` 做逐 bit 一致性测试。
solution 20980 将 down projection 从 **2.460 ms / 296.8 TFLOPS** 提升到
**2.374 ms / 307.6 TFLOPS**，用时降低 **3.51%**；solution 20981 将 gate_up 从
**4.375 ms / 333.8 TFLOPS** 提升到 **4.086 ms / 357.4 TFLOPS**，用时降低 **6.60%**。
两项输出均与默认 solution 逐 bit 相同。

### GDN prefill 与首次请求 JIT

Qwen3.5-27B 有 48 个 GDN 层，单次小收益会在每次 forward 中重复 48 次。Chunked GDN prefill
的关键链路是：

```text
Qwen3_5GatedDeltaNet
  -> chunk_gated_delta_rule
  -> recompute_w_u_fwd
  -> chunk_gated_delta_rule_fwd_h
  -> chunk_fwd_o
```

1. **按真实 shape/NT bucket 选择 Triton 配置。**
   对 gfx936、`BT=64`、目标 K/V 维度，根据 head 数和 chunk 数选择 `BK/BV`、warps 和 stages；
   Qwen3.5 主 shape 为 `(H,K,V,BT)=(48,128,128,64)`。这避免短序列 warmup 与 16K～32K
   正式负载共享不合适的 autotune 结果，同时保留其他平台和 shape 的原始 autotune 回退。
   T=4K～32K 上，固定配置相对原始 autotune 的用时降低范围为：recompute **5.1%～6.9%**、
   state-update h **30.8%～33.3%**、output o **16.3%～19.8%**，各组输出一致。

2. **输出直接写最终地址 与 chunk metadata 复用。**
   普通 non-spec prefill 将 model runner 已分配的 `core_attn_out` 传入 `chunk_fwd_o`，直接写最终
   地址，消除临时 output tensor 和一次 device-to-device copy；同一轮 GDN pipeline 只生成一次
   chunk indices，再交给 cumsum、recompute 和 chunk kernel 复用。4K 完整 pipeline 从临时
   output+copy 的 **1.709 ms** 降到直写的 **1.628 ms**，用时降低 **4.71%**。显式复用 indices
   与各 stage 经缓存取得 indices 分别为 **1.628/1.629 ms**；前者消除了 6 次
   `prepare_chunk_indices` 入口调用。

### 微基准测试结果

#### Attention

| 项目 | 工作负载 | 基线 median | 优化后 median | 用时降低 |
|---|---|---:|---:|---:|
| BLOCK_M | q_len=4096，context=4K，BM32→64 | 7.462 ms | 4.734 ms | 36.56% |
| BLOCK_M | q_len=4096，context=16K，BM32→64 | 48.890 ms | 30.440 ms | 37.74% |
| BLOCK_M | q_len=4096，context=32K，BM32→64 | 103.748 ms | 64.627 ms | 37.71% |
| 3D segments | decode 24K，16→256 | 362.512 µs | 263.326 µs | 27.36% |
| 3D segments | decode 32K，16→256 | 469.470 µs | 263.688 µs | 43.83% |

| History | paged UA | gather | suffix+prefix FA | LSE merge | gather+FA+merge | 用时降低 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.726 ms | 0 | 1.650 ms | 0 | 1.650 ms | 65.08% |
| 4K | 13.294 ms | 0.412 ms | 4.561 ms | 0.333 ms | 5.301 ms | 60.12% |
| 8K | 21.897 ms | 0.814 ms | 7.227 ms | 0.332 ms | 8.365 ms | 61.80% |
| 16K | 38.997 ms | 1.614 ms | 12.533 ms | 0.332 ms | 14.464 ms | 62.91% |
| 24K | 56.095 ms | 2.412 ms | 17.822 ms | 0.332 ms | 20.561 ms | 63.35% |
| 32K | 73.210 ms | 3.210 ms | 23.148 ms | 0.332 ms | 26.664 ms | 63.58% |

| Seq | 通用 gather 路径 | KV mirror | 用时降低 | mirror 路由 |
|---:|---:|---:|---:|---|
| 8K | 5.302 ms | 4.477 ms | 15.57% | full-KV causal FA |
| 16K | 11.412 ms | 9.956 ms | 12.76% | full-KV causal FA |
| 24K | 17.500 ms | 15.598 ms | 10.87% | full-KV causal FA |
| 30K | 22.081 ms | 20.397 ms | 7.63% | full-KV causal FA |
| 31K | 22.843 ms | 20.237 ms | 11.41% | split FA + merge |
| 32K | 23.601 ms | 20.900 ms | 11.45% | split FA + merge |

#### GEMV 与 MLP

| `(M,K)` | rocBLAS | LLMM1 | Strided-K | Strided-K GB/s | 相对 rocBLAS |
|---|---:|---:|---:|---:|---:|
| (16384,5120) | 239.714 µs | 171.886 µs | 116.867 µs | 1435.95 | -51.25% |
| (96,5120) | 25.748 µs | 11.076 µs | 10.985 µs | 90.44 | -57.34% |
| (8192,5120) | 69.509 µs | 91.890 µs | 59.399 µs | 1412.69 | -14.55% |
| (5120,6144) | 50.162 µs | 69.106 µs | 50.399 µs | 1248.77 | +0.47% |
| (34816,5120) | 499.329 µs | 348.952 µs | 246.080 µs | 1449.11 | -50.72% |
| (5120,17408) | 150.309 µs | — | 151.280 µs（rows=8参考） | 1178.63 | +0.65% |
| (248320,5120) | 1.905 ms | 2.403 ms | 1.754 ms | 1450.20 | -7.93% |

| 项目 | 基线 | 优化后 | 结果 |
|---|---:|---:|---|
| 640 threads，M=16384 | 142.245 µs | 128.453 µs | -9.70%，bitwise |
| 640 threads，M=34816 | 299.552 µs | 270.662 µs | -9.64%，bitwise |
| 640 threads，M=248320 | 2.103 ms | 1.910 ms | -9.19%，bitwise |
| gate_up+SwiGLU 融合 | 261.489 µs | 247.957 µs | -5.17%，bitwise |
| rocBLAS solution 20980 | 2.460 ms / 296.8 TFLOPS | 2.374 ms / 307.6 TFLOPS | -3.51%，bitwise |
| rocBLAS solution 20981 | 4.375 ms / 333.8 TFLOPS | 4.086 ms / 357.4 TFLOPS | -6.60%，bitwise |

#### GDN

| T | recompute autotune→fixed | h autotune→fixed | o autotune→fixed |
|---:|---:|---:|---:|
| 4K | 0.278→0.263 ms（-5.15%） | 0.965→0.644 ms（-33.30%） | 0.438→0.352 ms（-19.56%） |
| 8K | 0.601→0.559 ms（-6.91%） | 1.909→1.288 ms（-32.50%） | 1.067→0.855 ms（-19.83%） |
| 16K | 1.245→1.173 ms（-5.83%） | 3.795→2.593 ms（-31.67%） | 2.194→1.826 ms（-16.78%） |
| 32K | 2.473→2.328 ms（-5.87%） | 7.570→5.241 ms（-30.77%） | 4.401→3.683 ms（-16.31%） |



## 优化点汇总表

| 阶段 | 优化点 / 提交 | 目标与生效条件 | 关键代码路径 | 软硬件依据 | 性能数据 |
|---|---|---|---|---|---|
| Full-attention | `BLOCK_M=64`（`9097a14`、`d56a55b`） | Unified Attention 长 prefill；GQA ratio≤16 | `_get_block_m -> UA2D` | 减少长 query 的 q-block program，换取寄存器压力 | q_len=4096、4K～32K：BM32→64 用时降低 36.6%～37.7%；2D decode 由 BM32 胜出 |
| Full-attention | prefill `TILE_SIZE=16`（`0a27664`） | 非 Gemma、纯 Triton prefill | `_get_tile_size -> UA2D` | 降低 head_dim=256 的 LDS 压力 | 约 2.6～2.8× kernel 加速 |
| Decode attention | 3D softmax segments（`d10bbd2`、`d0b1ce1`） | 小 batch、长 context decode | `UA3D -> reduce_segments` | 以 segment 维扩展并行度 | 16→256：24K 用时降低 27.36%，32K 用时降低 43.83%；workspace 193.5 MiB |
| Full-attention | 连续 FlashAttention prefill（`0ff997e`） | ROCm BF16/FP16、普通 causal full attention | `_forward_fa_prefill -> FA -> LSE merge` | 避开 paged 寻址与有数值问题的 paged FA | history 0～32K 相对 paged UA 用时降低 60.1%～65.1% |
| Full-attention | 32K KV mirror + 30K hybrid route（`f958b91`） | 单请求、24Q/4KV/head=256 | `_forward_fa_prefill_mirror` | 用约 2 GiB 镜像换重复 gather | 8K～32K 用时降低 7.6%～15.6%；128 MiB/层 |
| Decode GEMV | 通用 Strided-K + shape rows（`87eae58`、`9b1adfa`、`ee1e066`） | BF16/FP16、N=1、无 bias、K%8=0 | `rocm_unquantized_gemm -> LLMM_StridedK` | 合并 16-byte 读取，固定线程 grid-stride K | 七组目标 shape 相对 rocBLAS 为 -57.3%～+0.7%；最高 1450 GB/s |
| Decode GEMV | BF16 FP32 `fmaf` 累加（`2f151cc`） | gfx936 BF16 reduction | `LLGemmStridedK_kernel` | 避免 packed BF16 模拟和逐 pair 舍入 | [16384,5120] rel-L2 1.88e-5，LLMM1 为 3.47e-3 |
| Decode GEMV | K=5120 使用 640 threads（`be6637e`） | K=5120 的目标投影 | `LLMM_StridedK` launch | 640 个 16-byte chunk 对应 10 个 wave64 | 约 +9%～10% |
| Decode MLP | gate_up GEMV + SwiGLU 融合（`13639b4`） | TP=1、BF16、无量化/LoRA、[34816,5120] | `Qwen3_5MLP -> LLMM_SiluMul` | 消除完整 gate_up 中间量和 activation launch | 261.489→247.957 µs，用时降低 5.17%，bitwise |
| Decode GEMV | down rows=1/1024 threads（`4d411c5`） | BF16 [5120,17408] | `LLGemmStridedKRows1Down_kernel` | 增加独立 HBM 请求、不复制权重读取 | rows=8 参考 151.280 µs，rocBLAS 150.309 µs；rows=1 为精确 shape 路由 |
| Decode GEMV | LM head rows=2（`eb40282`） | gfx936 BF16 [248320,5120] | `_use_strided_gemv -> LLMM_StridedK` | 仅放开目标大词表 shape | 1.905→1.754 ms，用时降低 7.93%；相对 LLMM1 用时降低 27.03% |
| Prefill MLP | down solution 20980（`da496ba`） | BF16 [4096,17408]×[17408,5120] | `rocblas_bf16_mlp_down_4096` | 使用目标 shape 的 rocBLAS 已选优实现 | 2.460→2.374 ms，用时降低 3.51%，bitwise |
| Prefill MLP | gate_up solution 20981（`3a2d060`） | BF16 [4096,5120]×[5120,34816] | `rocblas_bf16_mlp_gate_up_4096` | 使用目标 shape 的 rocBLAS 已选优实现 | 4.375→4.086 ms，用时降低 6.60%，bitwise |
| GDN prefill | gfx936 shape/NT 配置（`97869b0`、`8bf1196`） | BT=64、K/V=128/256、目标 NT bucket | `recompute_w_u/chunk_fwd_h/chunk_fwd_o` | 48 层重复热点；避免短序列 autotune 配置污染 | recompute/h/o 分别用时降低 5.1%～6.9%/30.8%～33.3%/16.3%～19.8% |
| GDN prefill | output 直写 + chunk indices 复用（`97869b0`、`8bf1196`） | 普通 non-spec prefill | `Qwen3NextGatedDeltaNet -> chunk pipeline` | 减少临时分配、D2D copy 和 metadata 入口 | 1.709→1.628 ms，用时降低 4.71%；消除 6 次 indices 入口调用 |
| GDN 冷启动 | 长上下文 specialization 预热（`f916d70`、`442994b`） | V1 profile、gfx936 命中配置 bucket | `_warmup_prefill_kernels` | 在 KV cache 分配前完成 JIT/autotune | 首请求 45.478→0.008 s；预热 200.385 s；reserved 662→442 MiB |

## 时间线（workspace分支）

### 2026-06-20～24：建立比赛工程和证据闭环

从 SourceFind/OpenDAS v0.18.1 建立 baseline，明确只认可 DCU 实机结果；完善服务期 profile、
wheel 来源、三档吞吐、SLA、accuracy 和合规检查流程。`9097a14` 开始尝试 BM64，使优化从通用
猜测转向真实 backend、shape 和得分贡献。

### 2026-06-30～07-06：打通 ROCm 自定义算子并转向带宽优化

`f93a2de`、`ab94c4b` 让 gfx936 命中 skinny GEMM 并恢复 `_rocm_C` 构建。早期强制 LLMM1
处理 down projection 的 `f7a8de6` 收益或适用性不足，由 `61ab7c8` 完整回退；随后改用可覆盖
大 K 的 Strided-K，并开始调优 3D decode segments 和 Unified Attention tile。

### 2026-07-07～10：形成 FA prefill 与 Strided-K 主线

`0ff997e` 将满足约束的 full-attention prefill 路由到连续 FlashAttention，保留 decode 的
Triton paged 路径。`87eae58`、`9b1adfa`、`ee1e066` 逐步完善 Strided-K 和 shape fallback，
`2f151cc` 用 FP32 累加同时修复 gfx936 BF16 精度与性能问题。

### 2026-07-11～12：融合、专用 shape 与连续 KV 镜像

这一阶段加入 K=5120/640-thread、gate_up+SwiGLU 融合、decode down rows1/1024、LM head rows2，
并为 4096-token MLP 选择 rocBLAS 20980/20981。`7cee783` 的 fused K/V gather 在 `4490eaf`
回退，最终由 `f958b91` 的逐层连续 KV 镜像解决 chunked prefill 重复 gather。

### 2026-07-13：优化 48 层 GDN

`97869b0`、`8bf1196` 按 gfx936 的真实 H/K/V/BT/NT 选择 GDN Triton 配置，增加 output 直写和
chunk indices 复用；`f916d70`、`442994b` 把长上下文 specialization 提前到 profile 阶段编译，
避免第一个正式请求承担 JIT/autotune 和额外峰值显存。`d56a55b` 删除 wheel 构建期文本注入，改为运行
时读取 `BLOCK_M`、默认 64。
