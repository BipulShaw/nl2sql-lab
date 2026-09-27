"""BIRD: dev (June 2024 release), mini-dev (500-question SQLite subset) and train, from data/bird/."""

import json
from pathlib import Path

from nl2sql.data.common import DATA_DIR, Example

# split -> (dataset label, split label, questions file, databases dir); paths are relative to data/bird/
SPLITS = {
    "dev": ("bird", "dev", "dev/dev.json", "dev/dev_databases"),
    "minidev": ("bird-mini", "dev", "minidev/mini_dev_sqlite.json", "minidev/dev_databases"),
    "train": ("bird", "train", "train/train.json", "train/train_databases"),
}


def load_bird(split: str, root: Path | None = None) -> list[Example]:
    if split not in SPLITS:
        raise ValueError(f"BIRD has no split {split!r}; choose from {sorted(SPLITS)}")
    dataset, split_label, questions, databases = SPLITS[split]
    root = root or DATA_DIR / "bird"
    rows = json.loads((root / questions).read_text())
    return [
        Example(
            id=f"{dataset}-{split_label}-{row.get('question_id', i)}",
            dataset=dataset,
            split=split_label,
            db_id=row["db_id"],
            question=row["question"],
            evidence=row.get("evidence") or "",
            gold_sql=row["SQL"],
            difficulty=row.get("difficulty"),
            db_path=root / databases / row["db_id"] / f"{row['db_id']}.sqlite",
        )
        for i, row in enumerate(rows)
    ]
