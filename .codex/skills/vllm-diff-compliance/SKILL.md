---
name: vllm-diff-compliance
description: Review git diffs, patches, PRs, and local changes in this vLLM competition fork for compliance with the 2026 先导杯 Qwen3.5-27B DCU technical plan. Use when Codex is asked whether a change is compliant, legal, allowed, safe for submission, violates rules, changes model semantics, changes benchmark/statistics/API behavior, uses forbidden caching/speculation/quantization/pruning, or needs a competition compliance review before commit/merge.
---

# vLLM Diff Compliance

Use this skill as a code-review mode for competition compliance. Treat it as a rule-based review: lead with violations and risks, then list unclear items and required evidence. Do not rubber-stamp a diff because it improves throughput.

## Inputs

Review the diff the user gives, or collect one locally:

```bash
git diff --stat
git diff --cached
git diff
```

If the user names a base branch or commit, compare against it with `git diff <base>...HEAD`. Include staged and unstaged changes when the request says "current diff" or "local changes".

## Fast Scan

Run the bundled scanner on a patch file or stdin for obvious red flags:

```bash
git diff | python3 ~/.codex/skills/vllm-diff-compliance/scripts/scan_diff.py
```

Use scanner output as leads only. Always inspect the actual code paths and classify by behavior.

## Rules

Read [technical-plan-rules.md](references/technical-plan-rules.md) before giving a compliance judgment. Use it as the authoritative checklist extracted from `misc/2026年全国大学生计算机系统能力大赛-智能计算创新设计赛-基于国产加速卡的千问大模型推理服务优化-技术方案.pdf`.

Apply this decision scale:

- **BLOCKER**: The diff appears to violate a prohibition or changes locked task/benchmark/model semantics.
- **HIGH RISK**: The diff touches a prohibited boundary or could violate it without more evidence.
- **NEEDS EVIDENCE**: The diff may be compliant, but needs accuracy, SLA, build, docs, or path evidence.
- **OK**: No material compliance issue found from the reviewed diff.

## Review Procedure

1. Identify touched areas: model loading, tokenizer/chat template, sampling, scheduler, benchmark scripts, OpenAI API, dataset handling, cache, quantization, pruning, kernels, memory/KV cache, env vars, docs.
2. Check hard prohibitions first: model/weight/tokenizer changes, prompt/output behavior, skipped work, dataset/result caches, speculative decoding, persistent quantized/pruned artifacts, benchmark/statistics/API bypass.
3. Check locked parameters: `max_tokens`, `temperature=0`, `max-model-len`, `max-num-seqs`, `max-num-batched-tokens`, batch scheduler semantics, model/tokenizer/chat template, served model name, OpenAI API route/request/response format, bench percentile/result parsing.
4. Check allowed optimization fit: KV cache allocation, memory budget/block management, DCU-aware resource scheduling, decode attention/linear kernels, operator fusion, launch/copy overhead, non-persistent operator-level low precision such as activation dynamic quantization or KV cache quantization.
5. Check required submission evidence: custom environment variables documented, third-party code disclosed, source builds in the official container, accuracy and SLA validation paths stated when relevant.
6. Return findings in code-review style with file/line references where possible.

## Output Format

Use this concise structure:

```text
Findings:
- [BLOCKER/HIGH RISK/NEEDS EVIDENCE] path:line - issue and why it conflicts with the technical plan.

Unclear / Evidence Needed:
- ...

Compliant Surface:
- ...

Verdict:
- compliant / not compliant / cannot determine yet
```

If there are no findings, say so directly and mention residual risks such as unrun accuracy/SLA/build validation.

## Guardrails

- Do not approve changes that rely on hidden test-set behavior, sample filtering, shorter outputs, or route/statistic changes.
- Do not treat comments, docs, or env flags as sufficient if runtime code can still violate the rules.
- Do not require DCU performance validation for a pure documentation or scanner change, but call out when performance claims lack DCU evidence.
- If a diff is large, review the highest-risk paths first and say what was not reviewed.
