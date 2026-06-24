# BLOCK_M 默认值与 gate 设计

## 现状

`vllm/v1/attention/ops/triton_unified_attention.py` 的 `_get_block_m`（L25-40）：

```python
def _get_block_m(num_queries_per_kv: int) -> int:
    block_m_override = os.environ.get("QWEN_UA_BLOCK_M")
    if not block_m_override:
        return 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    block_m = int(block_m_override)
    if block_m < num_queries_per_kv:
        raise ValueError(...)
    return block_m
```

Qwen3.5-27B 的 `num_queries_per_kv=6 ≤ 16`，所以默认走 16。env `QWEN_UA_BLOCK_M` 可覆盖。

## BM64 为什么有效（三机制）

实测（同容器紧邻 A/B，12K prefill 单请求，cleanupfix 三组）：

| BLOCK_M | BLOCK_Q | total_q_blocks (q=4096) | 请求 elapsed | ua2d total | ua2d avg | ua2d pct |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 2049 | 23.26s | 6.95s | 144.8ms | 33.9% |
| 32 | 5 | 820 | 18.98s | 2.69s | 56.0ms | 16.6% |
| 64 | 10 | 410 | 18.34s | 1.38s | 28.7ms | 9.2% |

BM16→BM64 attention 时间降 5 倍，与 total_q_blocks 的 5 倍削减（2049→410）完全吻合。
三个机制同向叠加：

### 机制 1：grid CTA 总数大幅减少（调度层，主因）

```
L1054: BLOCK_Q = BLOCK_M // num_queries_per_kv          (整除派生)
L1066: total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs   (调度上界)
L1115: grid = (total_num_q_blocks, num_kv_heads)        (2d launch grid)
```

`num_kv_heads=4` 固定，CTA 总数 = `total_num_q_blocks × 4`：

```text
BM16: BLOCK_Q=2,  total_q_blocks=4096//2+1=2049, CTA=8196
BM32: BLOCK_Q=5,  total_q_blocks=4096//5+1=820,  CTA=3280
BM64: BLOCK_Q=10, total_q_blocks=4096//10+1=410, CTA=1640
```

BM16→BM64 CTA 总数 8196→1640（5 倍）。更少 CTA → 更少 wave → 更低调度开销 +
wave 尾巴浪费更小。**这是 BM64 有效的主因**：BM16 时大量 CTA 每个干的活太少，
调度/同步开销吃掉时间。

### 机制 2：每 CTA 的 tl.dot 更大（MFMA 利用率）

- BM16: `tl.dot(Q[16,256], K[256,32])`，M 维只填一个 MFMA M-tile（gfx936 BF16 M=16）
- BM64: `tl.dot(Q[64,256], K[256,32])`，M=64 一次喂 4 个 MFMA M-tile，流水线更满

更大 BLOCK_M 让 K/V load 一次给更多 Q 行复用，摊薄 global→LDS 访存。

### 机制 3：GQA padding 浪费降低

`offs_m = arange(0, BLOCK_M)`，query_pos 映射 `offs_m // num_queries_per_kv`
（除数是 num_queries_per_kv=6，不是 BLOCK_M）。BLOCK_M 必须是 nqpk 的覆盖，
无效行由 query_mask 处理：

```text
BM16: 16//6=2 位置 + 16-12=4 行 padding  (25% 浪费)
BM32: 32//6=5 位置 + 32-30=2 行 padding  (6%)
BM64: 64//6=10 位置 + 64-60=4 行 padding (6%)
```

BM16 有 25% Q 行是 padding，BM64 降到 6%。

## 调度链路里的一个权衡：BLOCK_Q slack

L1066 的 `total_num_q_blocks` 是**上界**（注释 L1057-1065：CPU 逐 seq 算 ceil 太慢，
用 `floor(q.shape[0]/BLOCK_Q) + num_seqs` 近似）。多出来的 CTA 在 L222 早退：

```python
if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
    return   # 空 CTA 早退
```

**BLOCK_Q 越大，上界越松，空 CTA 占比越高**：

```text
单 seq prefill (num_seqs=1): 多 1 个空 CTA
  BM16: 1/2049 = 0.05%   BM64: 1/410 = 0.24%    ← 可忽略
decode num_seqs=16, BLOCK_Q=2: launch 24 CTA, 8 空转 (33%)
  BM64 (BLOCK_Q=10) decode 多 seq 时 slack 占比更高
```

所以 **BM64 在 prefill 单 seq 大幅有效（CTA 多、wave 多、slack 小）**；
**decode 多 seq 小 batch 时 slack 浪费占比升高**，可能抵消部分收益。
实测 BM32 vs BM64 的 TPOT 持平（decode 无收益也无退化），说明 decode 场景
slack 代价没占上风——但这是 10-prompt partial，未全量验证。

## launch 次数不变（澄清一个常见误解）

BLOCK_M **不影响 launch 次数**。一次 forward 每层 attention 是 1 次 launch（2d）
或 2 次（3d + reduce_segments），与 BLOCK_M 无关。BLOCK_M 改的是**单次 launch 的
grid 大小**，不是 launch 频率。所以"BM64 减少 launch overhead"说法不准确——
变的是单次 launch 内的 CTA 调度量，不是 launch 数。

## gate 设计：改默认值分两步

### 现状问题

直接把 L29 的 `16` 改成 `64` 有三个问题：
1. 该 kernel 可能被非 Qwen3.5 模型用（nqpk 不同时默认值不同），直接改影响所有
   `nqpk<=16` 的模型。
2. 合规要求"能回退 baseline"，改死默认值就没了回退路径。
3. BM64 的 accuracy 只是 partial 10-prompt sanity（retrieval 100%、hotpotqa 67.71
   但样本小），**没过全量 accuracy gate 前不该改默认**。

### 步骤 1（现在，accuracy 未全量验证）

保持默认 16，BM64 通过 `QWEN_UA_BLOCK_M=64` env 显式启用。比赛提交时服务命令带
这个 env。**不改代码。**

### 步骤 2（全量 accuracy gate 通过后）

加 `QWEN_UA_BLOCK_M_DEFAULT` gate，用 env 传值而非开关+硬编码：

```python
def _get_block_m(num_queries_per_kv: int) -> int:
    # 1. 显式覆盖（最高优先级，A/B / 强制回退用）
    block_m_override = os.environ.get("QWEN_UA_BLOCK_M")
    if block_m_override:
        block_m = int(block_m_override)
        if block_m < num_queries_per_kv:
            raise ValueError(
                "QWEN_UA_BLOCK_M must be >= "
                f"num_queries_per_kv={num_queries_per_kv}, got {block_m}"
            )
        return block_m

    # 2. 新默认值 gate：默认 64，可回退 16
    #    只对 nqpk<=16 生效（Qwen3.5 nqpk=6 命中）
    #    nqpk>16 保持原 next_power_of_2 逻辑，不误伤其他模型
    if num_queries_per_kv <= 16:
        default_bm = int(os.environ.get("QWEN_UA_BLOCK_M_DEFAULT", "64"))
        return max(default_bm, num_queries_per_kv)
    return triton.next_power_of_2(num_queries_per_kv)
```

gate 关键点：

- `QWEN_UA_BLOCK_M` 仍是最高优先级，A/B / 回退走它，不破坏现有用法。
- `QWEN_UA_BLOCK_M_DEFAULT` 控制新默认值（默认 "64"），回退设 `=16` 即恢复 baseline，
  不用改代码。这是合规要求的"能回退"。
- 用 env 传值（"64"）而非开关（"1"/"0"）+ 硬编码，调默认值不用改代码。
- 只对 `nqpk<=16` 改默认，`nqpk>16` 保持原逻辑，不误伤其他模型。
- **不写 `if Qwen3.5` 模型名特判**——用 `num_queries_per_kv` 结构性参数 gate，
  是 backend 适配不是模型特化（合规边界，参考 pra26/aiter.md wrapper gate 讨论）。
- 保留下限校验 `max(default_bm, num_queries_per_kv)`。

### gate 进阶点（第二步之后考虑）

BM64 在 prefill 收益大、decode 有 slack 代价。可按 phase 分别选 BLOCK_M：
在调用处（L1053 附近）根据 `max_seqlen_q > 1`（prefill）vs `==1`（decode）传不同值。
但 `_get_block_m` 是模块级函数不区分 phase，要改调用处，增加复杂度。
**第一步先统一 64，全量验证后再评估分 phase。**

## 改默认前必须过的 gate

```text
1. 全量 accuracy: run_accuracy.sh all（不只 partial 10-prompt）
   重点长上下文：retrieval_multi_point / aggregation / gov_report / hotpotqa
2. 全量 throughput: run_throughput.sh 4-8K / 8-16K / 16-32K（不只 10-prompt）
   三桶 output tok/s + TTFT P99 + TPOT P99，对照 baseline 不熔断
3. CUDA graph 兼容: BLOCK_M 是 constexpr，改默认变 kernel 变体，确认 graph capture 正常
4. 同容器紧邻 A/B: baseline(默认16) vs 新默认(64)，非跨实例
```

未过 1-4 前，BLOCK_M 默认值保持 16，BM64 只走 env 显式启用。

## 相关文件

```text
vllm/v1/attention/ops/triton_unified_attention.py
  L25   _get_block_m              ← gate 改这里
  L153  kernel_unified_attention_2d   ← BM64 优化对象
  L1054 BLOCK_Q = BLOCK_M // num_queries_per_kv
  L1066 total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
  L1115 grid = (total_num_q_blocks, num_kv_heads)
  L222  空 CTA 早退
```

实测数据来源：`misc/profile/profile_runs/rerun_bm{16,32,64}_cleanupfix_20260623_16*.log`、
`misc/profile/blockm64_conclusion.md`、`misc/profile/blockm32_vs_bm64_throughput.md`。
