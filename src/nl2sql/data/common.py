"""Dataset-agnostic example model, data location and stratified subsampling."""

import os
import random
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

DATA_DIR = Path(os.environ.get("NL2SQL_DATA_DIR", "data"))


class Example(BaseModel):
    id: str  # stable across runs, e.g. "spider-dev-42", "bird-dev-1471"
    dataset: str  # spider | bird | bird-mini
    split: str  # train | dev
    db_id: str
    question: str
    evidence: str = ""  # BIRD's external-knowledge hint; empty for Spider
    gold_sql: str
    difficulty: str | None = None  # BIRD: simple | moderate | challenging
    db_path: Path


def stratified_sample(examples: list[Example], n: int, seed: int = 42) -> list[Example]:
    """Pick n examples, keeping each stratum (difficulty if the dataset has it, else db_id) at its share.

    Quotas use largest-remainder rounding, then a seeded sample inside each stratum. The subset keeps dataset
    order, so the same (split, n, seed) always selects the same examples.
    """
    if n >= len(examples):
        return list(examples)
    strata: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        strata[ex.difficulty or ex.db_id].append(i)
    quotas = {key: n * len(idx) / len(examples) for key, idx in strata.items()}
    alloc = {key: int(q) for key, q in quotas.items()}
    by_remainder = sorted(quotas, key=lambda key: (alloc[key] - quotas[key], key))
    for key in by_remainder[: n - sum(alloc.values())]:
        alloc[key] += 1
    rng = random.Random(seed)
    picked = [i for key in sorted(strata) for i in rng.sample(strata[key], alloc[key])]
    return [examples[i] for i in sorted(picked)]
