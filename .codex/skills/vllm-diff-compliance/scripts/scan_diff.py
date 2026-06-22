#!/usr/bin/env python3
"""Lightweight red-flag scanner for vLLM competition compliance diffs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pattern:
    severity: str
    name: str
    regex: re.Pattern[str]
    reason: str


PATTERNS = [
    Pattern("BLOCKER?", "sampling/output length", re.compile(r"\b(max_tokens|temperature|top_p|top_k|min_tokens|stop_token|stop)\b", re.I), "locked generation behavior or task definition may change"),
    Pattern("BLOCKER?", "input truncation/filtering", re.compile(r"\b(truncat|skip|filter|drop|ignore|sample_limit|limit_samples|shorten)\w*\b", re.I), "may avoid real inference cost or filter difficult samples"),
    Pattern("BLOCKER?", "model/tokenizer/template", re.compile(r"\b(tokenizer|chat_template|served[-_]?model[-_]?name|model_name|model_path)\b", re.I), "model identity, tokenizer, chat template, or service identity is locked"),
    Pattern("BLOCKER?", "speculative decoding", re.compile(r"\b(speculat|draft|mtp|multi[-_]?head|predictor|early[-_]?exit)\w*\b", re.I), "speculative decoding and auxiliary predictors are prohibited"),
    Pattern("BLOCKER?", "pruning/skipping", re.compile(r"\b(prun|skip_layer|layer_skip|head_prun|token_prun|channel_skip)\w*\b", re.I), "pruning and dynamic skipping are prohibited"),
    Pattern("BLOCKER?", "persistent quant/cache artifacts", re.compile(r"\b(quantized_weight|weight_cache|safetensors|checkpoint|ckpt|save_pretrained|torch\.save|pickle|pkl|cache_dir|answer_cache|result_cache)\b", re.I), "persistent model/result caches can violate quantization or test-cache boundaries"),
    Pattern("HIGH RISK", "benchmark/statistics", re.compile(r"\b(percentile|p99|throughput|ttft|tpot|result_parser|bench|opencompass|longbench|ruler)\b", re.I), "benchmark/statistics/result paths are locked or score-sensitive"),
    Pattern("HIGH RISK", "OpenAI API surface", re.compile(r"\b(/v1/(chat/)?completions|openai|stream|request|response)\b", re.I), "service route and request/response format are locked"),
    Pattern("NEEDS EVIDENCE", "environment variable", re.compile(r"\b(os\.environ|getenv|export [A-Z_][A-Z0-9_]*|VLLM_[A-Z0-9_]+|HIP[A-Z0-9_]*|ROCM[A-Z0-9_]*)\b"), "custom env vars require documentation and defaults must be compliant"),
    Pattern("NEEDS EVIDENCE", "custom kernel/low precision", re.compile(r"\b(hip|rocm|dcu|kernel|bf16|fp16|int8|fp8|quant|kv_cache|pagedattention|attention)\b", re.I), "may be allowed, but needs semantics and validation evidence"),
]

PATH_PATTERNS = [
    Pattern("BLOCKER?", "model assets", re.compile(r"(^|/)(models?|tokenizer|checkpoints?|weights?)(/|$)|\.(safetensors|bin|pt|pth|gguf)$", re.I), "model/tokenizer/weight artifacts are locked or not submit-safe"),
    Pattern("HIGH RISK", "benchmark/eval files", re.compile(r"(bench|benchmark|opencompass|longbench|ruler|result|percentile|accuracy)", re.I), "benchmark/evaluation/statistics paths are score-sensitive"),
    Pattern("HIGH RISK", "API surface files", re.compile(r"(openai|serving|api_server|protocol|entrypoints)", re.I), "service API behavior may be locked"),
    Pattern("HIGH RISK", "scheduler files", re.compile(r"(scheduler|sampling|sequence|engine)", re.I), "locked scheduler/generation semantics may be affected"),
]


def iter_lines(text: str):
    current_file = None
    old_line = None
    new_line = None
    hunk_re = re.compile(r"^@@ -(?P<old>\d+)(?:,\d+)? \+(?P<new>\d+)(?:,\d+)? @@")
    for raw in text.splitlines():
        if raw.startswith("+++ b/"):
            current_file = raw[6:]
            continue
        if raw.startswith("+++ "):
            current_file = raw[4:]
            continue
        match = hunk_re.match(raw)
        if match:
            old_line = int(match.group("old"))
            new_line = int(match.group("new"))
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            yield current_file, new_line, raw[1:]
            if new_line is not None:
                new_line += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            if old_line is not None:
                old_line += 1
        elif not raw.startswith("\\"):
            if old_line is not None:
                old_line += 1
            if new_line is not None:
                new_line += 1


def scan(text: str) -> list[str]:
    findings: list[str] = []
    seen: set[tuple[str, str, str, int | None, str]] = set()

    files = []
    for line in text.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:])

    for file_path in files:
        for pattern in PATH_PATTERNS:
            if pattern.regex.search(file_path):
                key = (pattern.severity, pattern.name, file_path, None, "")
                if key not in seen:
                    seen.add(key)
                    findings.append(f"{pattern.severity}: {file_path} - {pattern.name}: {pattern.reason}")

    for file_path, line_no, added in iter_lines(text):
        if not added.strip():
            continue
        for pattern in PATTERNS:
            if pattern.regex.search(added):
                snippet = added.strip()
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                key = (pattern.severity, pattern.name, file_path or "<unknown>", line_no, snippet)
                if key in seen:
                    continue
                seen.add(key)
                loc = f"{file_path or '<unknown>'}:{line_no or '?'}"
                findings.append(f"{pattern.severity}: {loc} - {pattern.name}: {pattern.reason}\n    + {snippet}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan a git diff for vLLM competition compliance red flags.")
    parser.add_argument("patch", nargs="?", help="Patch file. Reads stdin when omitted.")
    args = parser.parse_args()

    if args.patch:
        text = Path(args.patch).read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()

    findings = scan(text)
    if findings:
        print("Potential compliance review leads:")
        for finding in findings:
            print(f"- {finding}")
        print("\nScanner output is not a verdict. Inspect behavior against references/technical-plan-rules.md.")
        return 1

    print("No obvious red flags found by keyword scan. Manual compliance review is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
