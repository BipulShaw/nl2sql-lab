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


def load_correct(run_dir: Path) -> list[tuple[str, bool]]:
    """(id, correct) per prediction, in the run's order."""
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as f:
        return [(r["id"], bool(r["correct"])) for r in map(json.loads, f)]


def compare_runs(run_a: Path, run_b: Path) -> dict:
    """Counts of examples both, one or neither run got right, and McNemar's p for the two EX scores.

    Records pair up by position, after checking both runs list the same ids in the same order. Pairing by id
    would merge BIRD mini-dev's repeated questions: ids 137 and 138 each appear twice in the official file."""
    a, b = load_correct(run_a), load_correct(run_b)
    if [i for i, _ in a] != [i for i, _ in b]:
        raise ValueError("the runs cover different examples; compare runs of the same split and --limit")
    pairs = [(x, y) for (_, x), (_, y) in zip(a, b, strict=True)]
    only_a = sum(x and not y for x, y in pairs)
    only_b = sum(y and not x for x, y in pairs)
    both = sum(x and y for x, y in pairs)
    return {
        "n": len(pairs),
        "both": both,
        "only_a": only_a,
        "only_b": only_b,
        "neither": len(a) - both - only_a - only_b,
        "p_value": mcnemar_exact_p(only_a, only_b),
    }
