---
name: pra26-score
description: Calculate 2026 先导杯 Qwen3.5/vLLM preliminary scoring from the three throughput bucket improvements, SLA pass/fail flags, and optional four-task accuracy drops. Use when estimating score impact from 4-8K, 8-16K, and 16-32K throughput changes or comparing optimization priorities.
---

# PRA26 Score

Use this skill when calculating preliminary score for the Qwen3.5/vLLM competition from throughput improvements in the three official buckets.

## Formula

Buckets and max points:

```text
4-8K:    20
8-16K:   50
16-32K:  30
```

For each bucket with relative throughput improvement `r`:

```text
bucket_score = bucket_max * (0.60 + 0.40 * (1 - exp(-1.3 * r)))
```

If a bucket fails TTFT/TPOT SLA, that bucket score is `0`.

The final score is:

```text
final_score = throughput_score * average(k_question_answer, k_summary, k_retrieval, k_aggregation)
```

Accuracy coefficient by relative accuracy drop `delta`:

```text
delta <= 1%        k = 1.00
1% < delta <= 2%   k = 0.97
2% < delta <= 3%   k = 0.94
3% < delta <= 5%   k = 0.90
5% < delta <= 10%  k = 0.85
delta > 10%        k = 0.00
```

## Script

Prefer the bundled deterministic script:

```bash
python3 .codex/skills/pra26-score/scripts/pra26_score.py \
  --lift-4-8 0.10 --lift-8-16 0.20 --lift-16-32 0.05
```

Or pass percentages:

```bash
python3 .codex/skills/pra26-score/scripts/pra26_score.py \
  --lift-4-8 10% --lift-8-16 20% --lift-16-32 5%
```

Or compute improvements from baseline and candidate throughput:

```bash
python3 .codex/skills/pra26-score/scripts/pra26_score.py \
  --baseline-4-8 12.19 --candidate-4-8 13.0 \
  --baseline-8-16 10.0 --candidate-8-16 12.0 \
  --baseline-16-32 5.0 --candidate-16-32 5.5
```

Mark SLA failures explicitly:

```bash
python3 .codex/skills/pra26-score/scripts/pra26_score.py \
  --lift-4-8 10% --lift-8-16 20% --lift-16-32 5% \
  --fail-sla-16-32
```

Include accuracy drops when known:

```bash
python3 .codex/skills/pra26-score/scripts/pra26_score.py \
  --lift-4-8 10% --lift-8-16 20% --lift-16-32 5% \
  --qa-drop 0.5% --summary-drop 1.2% --retrieval-drop 0% --aggregation-drop 0%
```

## Interpretation

- `8-16K` has the largest score weight and highest marginal score per 1% throughput improvement.
- SLA failures zero only the affected throughput bucket, but TPOT P99 is global in the technical plan, so a global TPOT regression can threaten all buckets depending on evaluation handling.
- Accuracy coefficient multiplies the entire throughput score, so risky low-precision optimizations must be checked against all four task classes.
