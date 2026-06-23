# LDS bank conflict 待验证问题

## 结论

- LDS bank conflict 目前只是可疑问题，不是已确认瓶颈。
- 不应在没有 profile / ISA / microbenchmark 证据前改主路径 kernel。
- 近期目标是先确认哪些 Qwen3.5 热路径实际大量使用 LDS，以及这些 LDS 访问是否产生
  可观测 bank conflict 或 LDS stall。
- 若证据成立，再考虑 padding、swizzle、lane-to-element mapping 或 tile layout 调整。

## 背景

在 AMD/DCU 上，LDS 是 workgroup 内共享存储。attention、skinny GEMM、归约、cache
写入和部分 fused kernel 都可能用 LDS 暂存数据。LDS 能降低全局访存压力，但访问模式不当
时可能出现 bank conflict，导致同一 wavefront 的 LDS load/store 被串行化。

这条线和当前比赛目标的关系是：

- Qwen3.5 长上下文下，decode / prefill 的瓶颈可能来自 attention KV 访存、metadata、
  GDN / linear attention、skinny GEMM 或边缘 fused op。
- LDS bank conflict 只可能解释其中“已命中 LDS 且 LDS stall 明显”的子路径。
- 如果当前 profile 显示瓶颈主要在 global memory、MFMA、launch overhead、Python
  调度或 KV metadata 构建，则优先级应低于更直接的路径。

## 本地候选代码

当前仓库里需要优先观察的 LDS 相关路径：

```text
csrc/rocm/skinny_gemms.cu
csrc/rocm/attention.cu
csrc/attention/attention_kernels.cuh
vllm/v1/attention/ops/triton_decode_attention.py
vllm/v1/attention/ops/prefix_prefill.py
```

其中 `csrc/rocm/skinny_gemms.cu` 明确把 activation matrix 搬到 LDS，并用
`__builtin_amdgcn_global_load_lds` 做 global-to-LDS 路径；这是最适合先做
microbenchmark / ISA 检查的候选点。

`csrc/rocm/attention.cu` 和通用 attention kernel 中有 shared memory / LDS 归约逻辑。
但上一轮 `_rocm_C` 记录显示服务实际 attention backend 是 `TRITON_ATTN`，因此这些 C++
attention kernel 是否在 Qwen3.5 主服务路径命中，必须先通过入口计数或 profile 确认。

Triton attention 路径也可能使用 LDS，但需要看编译后的 IR / ISA。不能只凭 Python 源码里
看不到显式 LDS 就排除，也不能只凭 kernel 名字推断 bank conflict。

## 需要回答的问题

1. 当前 Qwen3.5 4K-8K / 16K-32K throughput 下，哪些 kernel 是真实 hot path？
2. hot kernel 是否使用 LDS，使用规模是多少，是否接近 LDS 容量或影响 occupancy？
3. profile 中是否有 LDS bank conflict、LDS serialization、LDS stall 或等价指标升高？
4. 冲突发生在 LDS load、store、归约、tile staging，还是 global-to-LDS copy 后的读取？
5. 是否存在固定 stride / layout 导致 wavefront 内多个 lane 命中同一 bank？
6. padding 或 swizzle 能否减少冲突，同时不降低 occupancy 或增加寄存器压力？
7. 该路径是否在 `TRITON_ATTN`、AITER、`_rocm_C` overlay、FP8 KV Cache 等不同配置下都命中？

这些问题没有答案前，不做主路径 patch。

## 验证入口

先做 profile，不先改代码：

```text
A. clean baseline
B. clean + fp8 KV
C. AITER unified attention + --block-size 16
D. _rocm_C overlay
```

每组都记录：

```text
attention backend
top kernels by time
LDS/shared-memory related counters
occupancy / wavefront occupancy
global memory bandwidth counters
TTFT P99 / TPOT P99 / output_throughput
```

若使用 hipprof，应优先采短窗口，覆盖稳定 decode 段和至少一个长 prefill 段。profile
结论应写到 run 目录，并只引用具体 run 的 counter 和 kernel 名称。

## 可接受证据

可以推动 LDS patch 的最低证据：

- profiler 显示某个已命中的 kernel 是 top hot path；
- 同一 kernel 有明显 LDS bank conflict / LDS stall / serialization 指标；
- ISA 或源码能定位具体访问 pattern；
- microbenchmark 证明 padding / swizzle / layout 改动能降低 LDS 冲突；
- vLLM 级 A/B 在同容器、同参数、同 backend 下显示稳定收益；
- accuracy gate 不受影响。

不够的证据：

- 只看到代码用了 `__shared__` / LDS；
- 只看到吞吐波动，没有 kernel counter；
- 只在非主路径 microbenchmark 里变快；
- 只在 CUDA/NVIDIA bank conflict 经验上推断 DCU 行为；
- 只看一次短跑，没有同容器 A/B。

## 可能修法

证据已成立（见上文根因定位）。按风险从低到高、且都可环境变量 gate 回退考虑：

1. **PROBE_STAGE 隔离 QK vs PV（零代码改动，先做归因）**。
   已有 `QWEN_UA_PROBE_STAGE=qk`（只跑 QK dot+sum）和 `=softmax`（跑到 softmax 不做 PV）。
   分别跑 PMC 对比 `bank_conflict_per_lds_inst`：若 qk 显著高 → 偏 K tile；若 softmax 增量高
   → 偏 V tile。先定位冲突偏 K 还是 V。

2. **新增 `QWEN_UA_NUM_STAGES` gate 给 tl.dot 加 num_stages**。
   当前 L417/L489 的 `tl.dot` 无 num_stages。num_stages=2 会 double-buffer K/V tile，
   改变 LDS 分块相位，常能打散 stride-aligned 冲突。默认 1=baseline，可试 2/3。数学等价
   （BF16 matmul 不变），需确认 Triton ROCm 后端支持该 hint，不支持则 fallback。

3. **新增 `QWEN_UA_TILE_SIZE` gate 扫 TILE_SIZE**。
   当前 TILE_SIZE 由 `_get_tile_size` 固定（32/16），无 env override。试 16/64：改变 K
   `[256,T]`/V`[T,256]` 分块比例，验证"tile 越大冲突越多"假设。只改 K/V 循环分块粒度，
   不改 mask/scale/输出，等价。

4. **HEAD_SIZE_PADDED padding 验证根因**。
   当前 PADDED=256 无 padding。gate 设为 384（改变与 32 的倍数关系）或 512（仍是倍数）：
   若 384 降 conflict、512 不降 → 证实"leading stride=bank 倍数"是根因。padding 元素在
   `dim_mask`（L238 `offs_d < HEAD_SIZE`）处 mask 掉，数学等价。注意 PADDED 增大增加 LDS
   占用，512 时 K tile 翻倍，需确认不超 64KB。

5. **`QWEN_UA_BLOCK_M` gate（已存在）作对照**。
   试 BLOCK_M=32：Q 变 `[32,256]`，改变 MFMA M 维分块。若 conflict 随 BLOCK_M 变化 → 冲突
   也涉及 Q/acc 路径；若几乎不变 → 确认集中在 K/V。

6. （兜底）若 Triton hint 都不奏效，再考虑 inline asm 或自写 HIP attention 子路径做
   XOR swizzle（参考 AMD CK-Tile 的 XOR swizzle 方案），但工程量大、精度风险高，放最后。

所有 gate 都必须默认关闭（baseline 行为不变），只在显式 env 开启时生效，便于 A/B 和回退。
每步都用同一 PMC counter（`bank_conflict_per_lds_inst`/`wait_lds_per_lds_inst`/`lds`）对比，
并在有卡节点跑 accuracy gate（LongBench/RULER 子集）确认无精度回归。

任何修法都必须保持数学等价，不改变模型结构、tokenizer、输入输出口径、生成长度或
评测统计路径。


## 根因定位（PMC + 源码 + 文献）

PMC 确认冲突在 `kernel_unified_attention_2d`（Triton JIT kernel，不是 C++ kernel）。
该 kernel 在 `vllm/v1/attention/ops/triton_unified_attention.py`，2d 主体在 L152-514。

运行参数：`BLOCK_M=16, BLOCK_Q=2, TILE_SIZE=32, head_size=256, HEAD_SIZE_PADDED=256
（next_power_of_2(256)=256，无 padding）, block_size=784, num_kv_heads=4,
num_queries_per_kv=6, wave_size=64`。

tile 布局（2d kernel）：

```text
Q   [BLOCK_M, HEAD_SIZE_PADDED] = [16, 256]    (L243-247, 循环外 load 一次)
K   [HEAD_SIZE_PADDED, TILE_SIZE] = [256, 32]  (L344-356, 转置布局, d 维在行)
V   [TILE_SIZE, HEAD_SIZE_PADDED] = [32, 256]  (L337-371, d 维在列)
S/P [BLOCK_M, TILE_SIZE] = [16, 32]
acc [BLOCK_M, HEAD_SIZE_PADDED] = [16, 256]    (L261, 循环内反复读写)
QK: tl.dot(Q, K)   [16,256]×[256,32] = [16,32]  (L417)
PV: tl.dot(P, V)   [16,32]×[32,256] = [16,256]  (L489)
```

关键源码事实：核内**零 Triton hint**——无 `num_stages`、`num_warps`、`order=`、
`eviction_policy=`、`@triton.autotune/heuristics`。LDS 布局完全由 Triton/LLVM ROCm 后端
默认 pipeline 决定，无任何手控 swizzle。

根因（结合 PMC `bank_conflict_per_lds_inst=1.77`、`wait_lds_per_lds_inst=2.03`、
`lds=33792` 容量未饱和）：

```text
1. K/V tile 的 leading dimension = 256 = LDS bank 数(32) 的整数倍。
   K [256,32] 和 V [32,256] 的连续维度 stride = 256 × 2B(BF16) = 512B = 16×32B
   = 16 个 bank 的整数倍。wave_size=64 下同一 wavefront 的 lane 沿该维度步进时
   lane→bank 映射周期性撞同一 bank → 经典 stride-conflict。
   lds=33792 ≈ K(16KB)+V(16KB)+Q/P/acc 杂项，与"K+V 双 tile 占主体"一致。

2. 核内无 num_stages / order=，ROCm 后端默认对 [HEAD,TILE] 转置 tile 不插 XOR swizzle。
   lds=33792 远未到 64KB 却高 stall → 访问相位问题，不是容量问题（与 PMC 结论一致）。

3. stride_k_cache_3 = stride_v_cache_3 = 1（constexpr，KV cache shape
   [num_blks, 784, num_kv_heads, 256]），固化 head_size 维单位步长对齐，
   后端无法从 stride 推导出 swizzle 需求。
```

文献对照（anysearch，AMD 官方）：

- AMD ROCm 博客《Avoiding LDS Bank Conflicts on AMD GPUs Using CK-Tile》明确：MI 系列 LDS
  32 bank / 4 字节，wavefront=64 lane；bank conflict 最常见于"stride 是 bank 数整数倍"的
  访问模式；naive 布局对 `ds_read_b128` 产生 2-way bank conflict，XOR swizzle 可消除且不
  增 LDS。我们 PMC 的 1.77 conflict/inst ≈ 2-way，与该博客 naive 布局的 2-way 量级吻合。
- ROCm 官方《Optimizing Triton kernels》示明 Triton 在 ROCm 上 `tl.dot` 会把 blocked
  layout 经 `triton_gpu.convert_layout` → `#shared` → `tt.trans` → `#shared2` →
  `dot_op` 的路径塞进 LDS 做转置，正是冲突来源；可用 Triton GPU IR（.ttgir）确认实际
  LDS 布局。
- Triton issue #6446：AMD 上 blocked→dot_op 布局转换会整块塞进 shared memory，大 tile
  时甚至撑爆 LDS。佐证 K/V [256,32] 转置 tile 走的就是这条路径。
- arXiv 2511.11581《The Anatomy of a Triton Attention Kernel》：Triton paged attention
  kernel 在 AMD 上靠 autotuning 才达到 SOTA；tile 配置（num_stages/num_warps/BLOCK）对
  ROCm 性能影响极大，默认配置通常非最优。

结论：根因是 K/V tile leading dimension=256 对齐 bank 边界 + 无 swizzle + 无 num_stages。
修法方向不是增大 LDS，而是改变 LDS 访问相位（num_stages / tile padding / order / swizzle）。

## 当前状态

```text
status: confirmed_for_unified_attention_2d (request-level isolation still needed)
owner: unassigned
next: session-gated PMC 重测，归因到具体请求窗口；再评估 padding/stride 改写
```

已有硬证据（misc/profile/profile_runs/pmc_direct_ua2d_exp_prof_bm16_20260623_141638）：

```text
kernel: kernel_unified_attention_2d  (Qwen3.5 full attention, TRITON_ATTN)
lds=33792 字节/block (≈33KB，离 64KB 上限有余量 → 不是容量挤占 occupancy)
wave_size=64
PMC rows=240

SQ_LDS_BANK_CONFLICT  mean/row=3.67e8
SQ_WAIT_INST_LDS      mean/row=4.22e8
SQ_INSTS_LDS          mean/row=2.08e8
SQ_INSTS_VALU         mean/row=6.31e8

bank_conflict_per_lds_inst = 1.77   # 每条 LDS 指令平均 1.77 个 bank conflict
wait_lds_per_lds_inst      = 2.03   # 每条 LDS 指令平均 2.03 cycle 的 LDS stall
lds_inst_per_valu_inst     = 0.33   # LDS 指令占计算指令 1/3
```

判断：

- bank conflict 不再是“可疑”，是已确认存在且不轻（1.77 conflict/inst、2.03 wait/inst）。
- lds=33792 离 64KB 上限有余量，说明问题在访问模式，不是 LDS 容量挤占 occupancy。
- 因此可优化方向应是 padding / lane stride 改写 / tile layout（见上文“可能修法”），而不是
  增大 LDS。
- 按本文件“可接受证据”清单，目前满足“profiler 显示 hot path + 同一 kernel 有明显
  bank conflict/LDS stall 指标”；还差“microbench 证明改动能降低冲突”和“vLLM 级同容器 A/B”。

未完成的隔离：这次 PMC 不是 session-gated，240 行混了 warmup + 真实请求窗口的 kernel；
且 `--pmc-off --session` 在本 DTK hipprof build 不落盘。数字方向可信，精确归因到某次
请求窗口要重测（用 hipprof session start/stop 控制窗口，或 rocprofiler API 按 kernel 名
filter）。

在拿到 microbench + 同容器 A/B 前，本文件不作为实现依据，只作为已确认的问题记录。
