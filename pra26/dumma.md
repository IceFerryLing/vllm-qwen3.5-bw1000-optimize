# DUMMA 编译

DUMMA probe 使用 `dcc` 编译，不走 `hipcc`。

关键参数：

```bash
dcc -x hip <source>.cpp -O3 --offload-arch=gfx936 \
  -L/opt/dtk-26.04-DCC2602-0317/dcc/lib/clang/17.0.0/lib/linux \
  -o <output>
```

## BF16 模型与 FP8 路线状态

当前 Qwen3.5-27B 服务按全量 BF16 权重路径处理。默认 BF16 baseline 不会因为模型本身
自动使用 FP8 KV 或 FP8 GEMM；FP8 只会在显式开启运行时路径时出现，例如：

```text
--kv-cache-dtype fp8
```

因此 FP8 KV Cache 是额外实验路径，不是 BF16 baseline 的默认组成部分，也不能把 FP8
相关收益归因到普通 BF16 full attention。

已有 `../before/day6.md` 记录显示，默认 `--kv-cache-dtype fp8` 在 Qwen3.5 上不理想：

```text
baseline:
  kv_cache_dtype: auto
  attention backend: TRITON_ATTN
  attention block size: 784

FP8 KV:
  kv_cache_dtype: fp8
  attention backend: TRITON_ATTN
  attention block size: 1568
```

FP8 KV 虽然让 KV cache token 容量约翻倍，但也改变了 attention block size，从 `784`
变成 `1568`。这不再是单纯的 KV dtype 变化，会影响 Triton attention 的 tile、cache 和
metadata 行为。历史实测中 4K-8K、8K-16K 的 TTFT/吞吐都不优，因此：

```text
默认 FP8 KV Cache 不作为近期主线。
FP8 load/dequant 只保留为低优先级研究项。
```

## DUMMA 与 full attention 接入判断

当前已确认两点：

```text
1. Triton JIT 产物已经用上 MFMA。
2. FP8 KV 路径没有用上专门的 FP8 load / convert 指令，但该路径已降级为低优先级。
```

这说明当前更精确的问题不是“能不能让 attention 用矩阵核心”。对近期主线来说，更应该
围绕 BF16 full attention 热点看：

```text
现有 Triton full attention 已经能用 MFMA；
真正需要改的是 Qwen3.5 head_dim=256 / block_size=784 / GQA=6 的 paged attention 结构、
tile 选择、寄存器压力、block table 访问和 online softmax 开销。
```

### DUMMA 不适合直接替换 full attention 主路径

先校正一个事实口径：DUMMA 是 device-side warp-level fragment API（`du_mma_sync` 在
`__global__` kernel 内调用，类 WMMA），不是 host-launched GEMM 库；hipBLASLt/rocBLAS
才是 host-launched 库。但这一区别不改变下面的结论。

关于 MFMA 能力归属：Triton JIT 机器码已经包含 MFMA 指令，`tl.dot` 和 DUMMA 的
`du_mma_sync` 底层发射的是同一硬件 MFMA 单元。因此 DUMMA 提供的不是"Triton 缺失的
矩阵核心能力"，而是同一 MFMA 硬件下的另一种 fragment 封装。DUMMA 对照项的目的只能是
诊断 MFMA 利用充分度（寄存器布局 / scheduling / tile 是否更充分），不能预设 DUMMA 更优；
也可能结论是 Triton 已经用得不错、DUMMA 无优势，由对照项 C 的证据决定。

阻挡 DUMMA 直接替换 full attention 的真正理由是它的 tile 与接口形态，而不是 host-launch：

```text
DUMMA fragment tile 固定（BF16 为 16x16x16），不感知 paged/block table，
不自带 online softmax / mask / GQA head mapping。
```

而 Qwen3.5 full attention 热点不是一个单独 GEMM，而是：

```text
paged KV block table lookup
QK
online softmax
PV
causal / prefix masking
GQA head mapping
KV cache dtype / scale
长上下文分块
```

如果直接用 DUMMA 替换 full attention，通常会变成：

```text
gather paged K/V -> 临时连续 buffer
DUMMA QK GEMM
softmax kernel
DUMMA PV GEMM
scatter/write output
```

如果直接用 DUMMA 替换 full attention，由于 DUMMA 是 device-side fragment API，技术上
可以在一个自定义 HIP kernel 内 inline 调用 QK/PV，不必然引入多次 host launch。但要在
这个 kernel 里同时手写：gather paged K/V、DUMMA QK、online softmax、DUMMA PV、causal/
prefix mask、GQA head mapping、KV scale、长上下文分块、输出写回。这等于从零重写当前
Triton 已经 fused 好的 attention，且不会自动获得 paged 感知和 online softmax。它很可能
破坏当前 Triton fused attention 的优势，工程量也接近重写 attention。因此：

```text
DUMMA full attention replacement 不作为 P0。
```

### DUMMA 也不是 FP8 KV 的直接答案

先校正一个本卡约束：DUMMA 的 FP8（e4m3/e5m2）MMA 在类型表里只列了 gfx938，当前比赛卡
按 gfx936 处理，因此 DUMMA FP8 MMA 在本卡不可用。如果将来要做 FP8 GEMM roofline 对照，
应走 hipBLASLt（它有完整的 FP8/MXFP8 scaling 接口：A/B/C/D_SCALE_POINTER +
SCALE_MODE + AMAX_D_POINTER + Ext scaleAlphaVec），而不是 DUMMA FP8。

即使后续重新研究 FP8 KV，问题也在 Triton attention kernel 内部：

```text
tl.load(fp8) -> fp32 scale/dequant -> cast -> dot/MFMA
```

这段 dot 已经发射 MFMA。DUMMA（在支持 FP8 的卡上）或 hipBLASLt 能提供 FP8 GEMM 能力，
不等于能在现有 fused Triton kernel 里替换这段 `tl.load(fp8)`。如果为了调用它们先把
paged K/V gather 成连续矩阵，额外内存流量和 launch 开销可能抵消收益。

因此 DUMMA/hipBLASLt 更适合作为能力上限或 roofline 对照，而不是 FP8 KV 主路径接入方式。
考虑到默认 FP8 KV Cache 已经不理想，这条线暂时不作为近期主线。

### 如果以后重开 FP8

只有在能解释或控制 `784 -> 1568` block size 变化，并且 16K-32K 上出现明确收益信号时，
才值得重开 FP8 load/dequant。届时优先级应是：

```text
P0: 保存当前 Triton FP8 路径反汇编，定位 load/dequant 指令序列。
P1: 查 DTK/DCU 是否有 FP8 load/convert builtin、LLVM intrinsic 或 inline asm 形式。
P2: 在独立 Triton microbench 中尝试 inline asm 替代 fp8 load/dequant。
P3: 如果 Triton inline asm 不可行，写独立 HIP microkernel 做 FP8 paged load/dequant microbench。
P4: 只有 microbench 明显赢，且 FP8 KV 端到端不再变差，再接入 vLLM attention 局部。
```

接入 vLLM 时应尽量保留现有 attention 结构，只局部替换 FP8 K/V load/dequant：

```text
if kv_cache_dtype.startswith("fp8")
   and head_size == 256
   and block_size in (784, 1568):
    use optimized fp8 load/dequant path
else:
    use existing Triton path
```

这个 gate 只能做数学等价的 load/dequant 优化，不能改变 scale、mask、block table、
softmax、输出口径或模型语义。

### DUMMA 可保留的实验价值

DUMMA 仍然值得做 microbench，用来回答：

```text
1. DCU 上 BF16/FP8 GEMM 的理论上限是多少。
2. DUMMA/hipBLASLt 能否作为独立 GEMM roofline 对照，解释 Triton MFMA 产物离峰值多远。
3. 当前 full attention 瓶颈到底是 MFMA 计算、softmax、paging、block table 访问、
   register pressure，还是 kernel launch / metadata。
```

建议的对照：

```text
A. Triton BF16 QK/PV tile microbench
B. DUMMA / hipBLASLt BF16 GEMM microbench
C. Triton full attention kernel 的 VGPR / spill / occupancy / kernel-time 对照
D. 低优先级：Triton fp8 load -> descale -> dot 与 hipBLASLt FP8 GEMM 对照
   （DUMMA FP8 MMA 仅 gfx938，本卡 gfx936 不可用，改用 hipBLASLt）
```

如果 DUMMA GEMM 明显强，只能说明硬件和厂商库有矩阵计算能力；真正要接 vLLM，仍然要
解决 paged attention fused kernel 内部的 block table、mask、softmax、KV layout 和
state 管理问题。

### GEMM 实测与 roofline 优先判断

官方 16-32K 多 prompt 负载实测（`hipprof_stats_bm64_16_32k_official1_20260623_185033`）：

```text
Cijk_Alik_Bljk_BBH_MT..._ISA936 系列（rocBLAS/Tensile，BF16）合计 ≈ 26%+
triton_poi_fused_*_rocm_unquantized_gemm_silu_* （已 fused SiLU/RMSNorm）≈ 1.4%
GEMM 类合计 ≈ 27-28%，是官方负载第一大 kernel 类（与 fill 28.4% 相当）
```

关键事实：

```text
1. 这些 Cijk 是 rocBLAS/Tensile 已 autotune 的 GEMM：20+ 种 MT tile 配置
   （256x256x16 WGM1/4/8、64x32x32、32x16x4、128x64x32 …），每个 shape 选不同 tile。
   即 GEMM 库层面已经过厂商高度优化，不是朴素 GEMM。
2. 全部 BF16，无 FP8（本卡 BF16 权重路径）；Qwen3.5 非 MoE，无 grouped GEMM 场景。
```

判定：**AITER GEMM 和 DUMMA GEMM 在当前负载下都吃不到这 27%，不应投入替换**。

```text
- AITER GEMM：优势场景是 FP8（gemm_a8w8）/ MoE grouped；BF16 dense 无结构性优势，
  且 GEMM 已是 autotune rocBLAS。换 AITER 要改 linear.py dispatch，接入成本不低，收益不确定。
- DUMMA GEMM：device-side fragment（du_mma_sync，tile 固定 16x16x16 BF16），替 Linear GEMM
  等于重写 GEMM 库，平均打不过已 autotune 的 rocBLAS。DUMMA 价值是 roofline 对照，不是生产替换。
- hipBLASLt GEMM：与 rocBLAS 同源（Tensile），平均不会更快；但有 epilogue fusion
  （GELU/RELU/SIGMOID+Bias，无 SiLU）可探索，以及自带 hipblaslt-bench 做 roofline。
```

GEMM 的真优化空间不在“换库”，而在：① 减少 shape 碎片化（vLLM 侧，20+ tile 说明 shape
高度分散）；② epilogue 融合（triton fused 已在做，hipBLASLt 部分支持但无 SiLU）；
③ **先用 hipBLASLt `hipblaslt-bench` 跑 Qwen3.5 真实 M/N/K 的 BF16 GEMM 峰值，对照 Cijk
实测，判断这 27% 是硬件极限还是有优化空间**。这一步成本最低（自带 bench），应优先做——
若离峰值很近，GEMM 这 27% 基本是硬件极限，不该花时间；若离峰值远，再查是否 shape 碎片
导致次优 tile。在 roofline 结论出来前，不改 GEMM 库选择。

