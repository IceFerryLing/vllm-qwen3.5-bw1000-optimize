# 明日上卡计划

日期：2026-06-24

## 硬性决策

本次上卡冻结 FP8。

不要把时间花在：

- `exp/fp8`
- `--kv-cache-dtype fp8`
- DUMMA FP8 load/dequant patch
- FP8 精度或 profile 后续

本次目标是验证 Day 6 已经确定的 full-attention 方向，并产出覆盖整条计算链路的
profiling 证据。

本次上卡按单卡处理。除非服务意外启动了多卡路径，否则不要花时间分析
RCCL/NCCL 或多卡归因。

DTK PDF 还给出一条硬边界：按 DTK 26.04 DUMMA 类型表，DUMMA FP8 MMA
不是 gfx936 路径。文档中 FP8 e4m3/e5m2 MMA 只列了 GFX938，而当前比赛目标按
gfx936 处理。因此 DUMMA FP8 跟 FP8 KV 一起冻结。DUMMA 后续仍可作为
BF16/FP16 MMA roofline probe，但不是明天的 FP8 工作。

## P0：快速验证 BLOCK_M

根据 `../before/day6.md`，full-attention Triton prefill 是可信热点。先只做一次快速
`BLOCK_M` 对比：

```text
BLOCK_M = 16
BLOCK_M = 32
BLOCK_M = 64
```

本地 `triton_unified_attention.py` 对 Qwen3.5 GQA 的默认值实际等价于
`BLOCK_M=16`，因为：

```python
BLOCK_M = 16 if num_queries_per_kv <= 16 else next_power_of_2(num_queries_per_kv)
```

所以 `32/64` 需要临时实验 patch 或环境变量 gate，位置是：

```text
vllm/v1/attention/ops/triton_unified_attention.py
```

patch 只作为诊断实验，必须容易回退。在 profile 和 serving benchmark 都支持之前，
不要把它包装成最终优化。

最低有效形状：

```text
input_len=24576, output_len=16-64, concurrency=1
```

如果卡时间很紧，优先跑一个代表性的 16K-32K prefill-heavy slice，再考虑小桶。
`4K-8K` 只作为服务稳定性和正确性 smoke。

每个 `BLOCK_M` 记录：

```text
commit / patch id
server command
env vars
input_len / output_len / num_prompts / concurrency
TTFT p50/p90/p95/p99
TPOT p50/p90/p95/p99
ITL p50/p90/p95/p99
output tok/s
failed requests
server log path
result json path
```

决策规则：

```text
如果 32/64 没有明显改善长 prefill 的 TTFT 或 output tok/s，停止继续 sweep。
如果只改善 4K-8K、不改善 16K-32K，按 shape noise 处理。
如果 mean 改善但 p99 恶化到可能触发 SLA gate，不保留。
```

## P1：整链路 Profile

快速 `BLOCK_M` 检查之后切到 profiling。profile 必须回答整条推理链路中到底谁是
主要瓶颈：

```text
full attention
GDN / linear attention
GEMM / projection
RMSNorm / RoPE / elementwise
KV cache write/read
host API / launch / synchronize / allocator overhead
single-card stream / event / implicit synchronization overhead
```

只跑短 slice，不跑完整长任务：

```text
Prefill-heavy: 16K-32K input, 16-64 output
Decode-heavy: 4K-8K input, 512-1024 output
```

artifact 根目录：

```text
<TEAM_HOME>
```

至少保存：

```text
README.md
env.log
vllm_server.log
profile_command.txt
hip_stats*
hip_trace*
kernel_summary.txt
hotpath_conclusion.md
```

### 短窗口采集方法学

vLLM 是长驻 server，hipprof 默认要进程结束 / 缓存满 / 5s 才落盘，直接 `hipprof
./vllm serve ...` 拿不到指定的 decode/prefill 窗口。必须用 session 动态控制：

```text
1. 启动服务时挂 session：
   hipprof --session <key> ./vllm serve ...

2. benchmark 压测到稳定 decode（或一个长 prefill）时，另一终端开窗口：
   hipprof --session-client <key> --start

3. 采 1-2 秒后关窗口并落盘：
   hipprof --session-client <key> --stop --flush
```

这样得到的就是纯净的 decode / prefill 窗口，不必重启服务，也不把 prefill 段污染进
decode 采样。node 被 pam_slurm 回收时，落盘参数加 `--buffer-size 10000
--flush-interval 1000 --exit-cleanup` 防丢数据。

先用 `hipprof --stats`。只有 stats 找到有价值的短窗口后，再收 timeline。

本次不要只做普通 timeline，要更充分使用 DTK profiling 工具。最低 profile 阶梯：

```text
1. Stats:
   hipprof --hip-trace --stats --show-pid --devices 0 ...

2. 对一个短请求窗口收 timeline:
   hipprof --hip-trace --barrier --group-stream --trace-args \
     --show-pid --devices 0 --output-type 2 ...

3. 对 top attention/GDN kernel 收 PMC 拿资源证据:
   hipprof --pmc --pmc-type 3 --kernel-name <hot_kernel> ...

   注意 PMC 三组（--pmc / --pmc-read / --pmc-write）硬件计数器互斥，不能同一次 run
   拿全。要拿全需分三次独立 run：
     hipprof --pmc       --pmc-type 3 --kernel-name <hot_kernel>   # kernel-time / VGPR / SGPR / scratch(spill) / LDS size / workgroup / bank conflict
     hipprof --pmc-read  --pmc-type 3 --kernel-name <hot_kernel>   # L2 读字节 / 请求数（≈ global read bandwidth）
     hipprof --pmc-write --pmc-type 3 --kernel-name <hot_kernel>   # L2 写字节 / 请求数（≈ global write bandwidth）
   trace run 要加 --pmc-off 保证 timeline 准确，不要和 PMC 同开。

   这一步一次就输出每条 kernel 的 kernel-time / vgpr / sgpr / scratch(非零即 spill) /
   shared memory size / workgroup size / shared memory bank conflict(时间占比%) /
   shared memory operation(LDS 次数) / L1 cache unit is stalled(LDS stall 近似)。
   dumma.md 对照项 C 要的 VGPR / spill / occupancy / kernel-time 原料这里一次拿齐
   （occupancy 用 VGPR + LDS + workgroup 自行算）；LDS bank conflict 判据就是
   shared memory bank conflict 这一列。不需要先抽 code object。

4. Triton kernel 的 ISA / code object 提取（hot path 是 Triton unified attention，JIT
   无落盘 ELF）：
   - _rocm_C.abi3.so 等已落盘产物可直接：
       dccobjdump --inputs=_rocm_C.abi3.so --show-resource-usage --show-sass \
         --show-instruction-encoding --architecture=gfx936
   - Triton JIT 产物先落盘再反汇编：
       export MLIR_TRITON_DUMP_PATH=<TEAM_HOME>
       # 或 TRITON_KERNEL_DUMP / TRITON_DEBUG=1，按容器内 Triton 版本可用项选
       跑一次服务后从 dump 取 .llir / .asm 喂 dccobjdump。
     这是 dumma.md P0「保存 Triton 反汇编」和 lds.md「看编译后 IR/ISA」能落地的关键。

5. 归因：确认 hot kernel 从 _rocm_C / Triton launcher / _aiter_ops 哪个发起，用 kernel-stack
   不改代码：
     hipprof --hip-trace --kernel-stack --show-pid --devices 0 ...
   输出每个 kernel launch 的主机端调用栈 CSV，比插桩计数低侵入。
```

profile 结论不能只写“哪个 kernel 热”。必须记录证据更像指向哪一类瓶颈：

```text
MFMA compute
global memory / KV reads
online-softmax math
register pressure
LDS usage
spill
launch overhead
implicit synchronization
```

## P1 强制要求：full attention 内部插桩

profile 不能停在 “Triton unified attention 很热”。至少必须有一个 probe，把 full
attention 内部工作拆开看。

目标文件：

```text
vllm/v1/attention/ops/triton_unified_attention.py
```

Triton fused kernel 不能用 host timer 得到可靠的逐行耗时。这里使用诊断 probe
variants。最低可接受拆解：

```text
QK path:
  Q/K load, K descale if enabled, tl.dot(Q, K), scale/mask/softcap/alibi/bias

Online softmax path:
  max, exp, sum, alpha, running M/L update

PV path:
  V load/descale if enabled, sliding-window mask, tl.dot(P, V), accumulator update

Epilogue:
  acc / L, output scale, store
```

优先使用低风险方式：

```text
1. 增加环境变量 gate 的诊断模式，默认关闭。
2. 编译独立 probe variants，尽量保持相同 shape 和相同访存；
   分别在 QK 后停止、QK+softmax 后停止，或运行完整 QK+softmax+PV。
3. 对比各 variant 的 hipprof kernel time，估算 attention 内部成本。
4. 每个 variant 尽量保存资源证据：VGPR、SGPR、LDS usage、spill count、
   kernel descriptor，以及简短 SASS/MFMA/global-load 摘要。
5. probe mode 不能用于 accuracy 或最终 throughput 宣称。
```

probe label 要写进日志或 artifact：

```text
backend = TRITON_ATTN
kernel = unified_attention_2d or unified_attention_3d
BLOCK_M
BLOCK_Q
TILE_SIZE
head_size
block_size
num_query_heads
num_kv_heads
num_queries_per_kv
max_seqlen_q
max_seqlen_k
num_seqs
VGPR / SGPR / LDS usage, if available
spill count, if available
SASS evidence for MFMA / global load / LDS / stores, if available
```

可选的 ROCTX/HIPTX range 适合做 host/layer/backend 级关联，但不能替代
full-attention 内部 probe variants。内部拆解必须来自诊断 kernel variants 和
kernel-level profiling。

注意：AITER attention 和 native ROCm attention 不是本任务的有效替代路径。对
Qwen3.5，本次调查目标是当前可靠的 Triton full-attention 路径。

## P2：Benchmark Gate

使用 `vllm bench serve` 或等价逐 token gate。不要用旧的
`samples=1, max_tokens=8` 数字做决策。

本机 benchmark 客户端先清代理：

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

baseline 风格客户端形状：

```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --model <TEAM_HOME> \
  --dataset-name random \
  --input-len 24576 \
  --output-len 64 \
  --num-prompts 3 \
  --max-concurrency 1 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --save-result \
  --save-detailed
```

`num_prompts=3` 只算 smoke。正式宣称 p99 前必须提高样本数。

## 收工产物

本次 session 只有留下以下产物，才算成功：

```text
1. BLOCK_M=16/32/64 在长 prefill slice 上的对比表。
2. hipprof stats，说明 whole-chain top kernels/API costs。
3. 至少一组 full-attention internal probe 结果。
4. 简短 hotpath conclusion，明确下一个优化目标。
```

## 本次（2026-06-23）实际产出与缺口

已拿到（见 `misc/profile/profile_runs/`）：

```text
- hipprof trace + kernel_summary + hotpath_conclusion (BLOCK_M=16, 12K prefill)：
  full attention 33.8% 单一最大 kernel bucket；GDN/FLA chunk 系列合计约 24%；
  backend 确认 TRITON_ATTN / unified_attention_2d。
- kernel_unified_attention_2d 的 PMC bank conflict 证据：bank_conflict_per_lds_inst=1.77、
  wait_lds_per_lds_inst=2.03、lds=33792。LDS bank conflict 已从“可疑”升为“确认”。
  详见 pra26/lds.md。
```

注：20260622_1250_abc_profile 那批 A/B/C（default / roc_attn / aiter）吞吐对照**不可信**，
不作为任何结论依据：C 组是 sigpatch（rocm_aiter_unified_attn.py 去掉 sinks/output_scale
kwargs），且各分组非同容器紧邻 A/B。仅保留 run 目录原文件供回溯，不引用其数字。

缺口（下次上卡补）：

```text
1. BLOCK_M=32/64 对比表未拿到。
   - bm32 (hipprof_trace_exp_prof_bm32_20260623_143233) 刚起服务 1 分钟就被
     “remaining walltime < 1h”杀掉；bm32_64_latest.txt 为空。
   - 下次优先用更短 slice 跑 32/64，不要等长 prefill。

2. full-attention internal probe 只打了 shape 标签，没拆内部耗时。
   - root_exp_prof_start_20260623_131600/full_attention_probe_evidence.txt 里
     QWEN_UA_PROFILE 全是 probe_stage=0，只有 backend/kernel/BLOCK_M/shape 信息，
     没有 QK / softmax / PV 各阶段成本。
   - 下次要真正用上 probe_stage 开关或独立 probe variants 拆内部成本。

3. PyTorch fill elementwise kernel 占 18.5% (17308 calls) 被低估，来源未确认。
   - 详见 pra26/pytorch_prefill.md。下次用 hipprof --kernel-stack 定位其主机端来源。

4. HIP sync 三件套占 API 时间 90%+ (hipEventSync 31.6% / hipMemcpyAsync 31.4% /
   hipDeviceSynchronize 27.2%) 没被列为瓶颈。
   - 这是“implicit synchronization overhead”的直接证据，下次要单独追 hipDeviceSynchronize
     7.16s 来自 vLLM 哪条路径。

5. PMC 没有 session-gated。
   - pmc_direct_ua2d 混了 warmup + 请求窗口的 kernel（240 行），且 --pmc-off --session
     在本 DTK hipprof build 不落盘。下次用 hipprof session start/stop 控制窗口，或
     rocprofiler API 按 kernel 名 filter，做请求级隔离。

6. ABC 吞吐对照整批不可信，已废弃。
   - 20260622_1250_abc_profile 的 A/B/C 组：C 组是 sigpatch 改了 wrapper 行为，且各组非
     同容器紧邻 A/B。不引用其任何数字。
   - 若将来要测 AITER unified，必须做干净 AITER（非 sigpatch）+ 同容器紧邻 A/B + 同
     backend 日志确认，否则不算数。
```
