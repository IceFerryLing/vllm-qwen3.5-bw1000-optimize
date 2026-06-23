#!/usr/bin/env python3
"""Compute 2026 PRA Qwen3.5/vLLM preliminary score."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Bucket:
    name: str
    max_points: float
    lift_arg: str
    baseline_arg: str
    candidate_arg: str
    fail_arg: str


BUCKETS = [
    Bucket("4-8K", 20.0, "lift_4_8", "baseline_4_8", "candidate_4_8", "fail_sla_4_8"),
    Bucket("8-16K", 50.0, "lift_8_16", "baseline_8_16", "candidate_8_16", "fail_sla_8_16"),
    Bucket("16-32K", 30.0, "lift_16_32", "baseline_16_32", "candidate_16_32", "fail_sla_16_32"),
]


ACC_DROPS = ["qa_drop", "summary_drop", "retrieval_drop", "aggregation_drop"]


def parse_rate(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def score_bucket(max_points: float, lift: float, fail_sla: bool) -> float:
    if fail_sla:
        return 0.0
    return max_points * (0.60 + 0.40 * (1.0 - math.exp(-1.3 * lift)))


def accuracy_k(drop: float) -> float:
    pct = drop * 100.0
    if pct <= 1.0:
        return 1.00
    if pct <= 2.0:
        return 0.97
    if pct <= 3.0:
        return 0.94
    if pct <= 5.0:
        return 0.90
    if pct <= 10.0:
        return 0.85
    return 0.00


def add_rate_arg(parser: argparse.ArgumentParser, name: str, **kwargs) -> None:
    parser.add_argument(name, type=parse_rate, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_rate_arg(parser, "--lift-4-8", dest="lift_4_8", help="Relative improvement for 4-8K, e.g. 0.1 or 10%%")
    add_rate_arg(parser, "--lift-8-16", dest="lift_8_16", help="Relative improvement for 8-16K")
    add_rate_arg(parser, "--lift-16-32", dest="lift_16_32", help="Relative improvement for 16-32K")

    parser.add_argument("--baseline-4-8", type=float)
    parser.add_argument("--candidate-4-8", type=float)
    parser.add_argument("--baseline-8-16", type=float)
    parser.add_argument("--candidate-8-16", type=float)
    parser.add_argument("--baseline-16-32", type=float)
    parser.add_argument("--candidate-16-32", type=float)

    parser.add_argument("--fail-sla-4-8", action="store_true")
    parser.add_argument("--fail-sla-8-16", action="store_true")
    parser.add_argument("--fail-sla-16-32", action="store_true")

    add_rate_arg(parser, "--qa-drop", help="QA relative accuracy drop, e.g. 0.005 or 0.5%%")
    add_rate_arg(parser, "--summary-drop", help="Summary relative accuracy drop")
    add_rate_arg(parser, "--retrieval-drop", help="Retrieval relative accuracy drop")
    add_rate_arg(parser, "--aggregation-drop", help="Aggregation relative accuracy drop")
    return parser


def resolve_lift(args: argparse.Namespace, bucket: Bucket) -> float:
    direct = getattr(args, bucket.lift_arg)
    baseline = getattr(args, bucket.baseline_arg)
    candidate = getattr(args, bucket.candidate_arg)
    if direct is not None:
        return direct
    if baseline is not None or candidate is not None:
        if baseline is None or candidate is None:
            raise SystemExit(f"{bucket.name}: baseline and candidate throughput must be provided together")
        if baseline <= 0:
            raise SystemExit(f"{bucket.name}: baseline throughput must be > 0")
        return (candidate - baseline) / baseline
    raise SystemExit(f"{bucket.name}: provide --{bucket.lift_arg.replace('_', '-')} or baseline/candidate throughput")


def main() -> None:
    args = build_parser().parse_args()
    rows = []
    throughput_score = 0.0
    for bucket in BUCKETS:
        lift = resolve_lift(args, bucket)
        fail = bool(getattr(args, bucket.fail_arg))
        score = score_bucket(bucket.max_points, lift, fail)
        throughput_score += score
        rows.append((bucket.name, bucket.max_points, lift, fail, score))

    acc_values = [getattr(args, name) for name in ACC_DROPS]
    if all(value is None for value in acc_values):
        acc_rows = []
        acc_factor = 1.0
    elif any(value is None for value in acc_values):
        raise SystemExit("Provide all four accuracy drops or none")
    else:
        labels = ["QA", "Summary", "Retrieval", "Aggregation"]
        acc_rows = [(label, value, accuracy_k(value)) for label, value in zip(labels, acc_values)]
        acc_factor = sum(row[2] for row in acc_rows) / 4.0

    final_score = throughput_score * acc_factor

    print("Throughput buckets:")
    for name, max_points, lift, fail, score in rows:
        sla = "FAIL" if fail else "PASS"
        print(f"  {name:7s} lift={lift * 100:8.3f}% sla={sla:4s} score={score:8.3f}/{max_points:.0f}")
    print(f"Throughput score: {throughput_score:.3f}/100")

    if acc_rows:
        print("Accuracy coefficients:")
        for label, drop, k in acc_rows:
            print(f"  {label:11s} drop={drop * 100:8.3f}% k={k:.2f}")
    print(f"Accuracy factor:  {acc_factor:.4f}")
    print(f"Final score:      {final_score:.3f}/100")


if __name__ == "__main__":
    main()
