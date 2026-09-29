"""Paired comparison of two runs on the same examples, with McNemar's exact test.

Two models scored on the same questions are not independent samples: most questions are easy, or hard, for
both. Only the questions they disagree on say which model is better, and McNemar's test asks whether those
split more unevenly than coin flips would. A gap in EX with a large p-value is no evidence of a difference."""

import json
from math import comb
from pathlib import Path


def mcnemar_exact_p(only_a: int, only_b: int) -> float:
    """Two-sided p-value: the chance of a split at least this uneven if each disagreement were a fair coin."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load_correct(run_dir: Path) -> dict[str, bool]:
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as f:
        return {r["id"]: bool(r["correct"]) for r in map(json.loads, f)}


def compare_runs(run_a: Path, run_b: Path) -> dict:
    """Counts of examples both, one or neither run got right, and McNemar's p for the two EX scores."""
    a, b = load_correct(run_a), load_correct(run_b)
    if a.keys() != b.keys():
        raise ValueError("the runs cover different examples; compare runs of the same split and --limit")
    only_a = sum(a[i] and not b[i] for i in a)
    only_b = sum(b[i] and not a[i] for i in a)
    both = sum(a[i] and b[i] for i in a)
    return {
        "n": len(a),
        "both": both,
        "only_a": only_a,
        "only_b": only_b,
        "neither": len(a) - both - only_a - only_b,
        "p_value": mcnemar_exact_p(only_a, only_b),
    }
