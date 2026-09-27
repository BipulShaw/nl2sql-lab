"""Spider 1.0: questions from the official JSON files, SQLite databases from data/spider/database/."""

import json
from pathlib import Path

from nl2sql.data.common import DATA_DIR, Example

SPLIT_FILES = {"train": ["train_spider.json", "train_others.json"], "dev": ["dev.json"]}


def load_spider(split: str, root: Path | None = None) -> list[Example]:
    if split not in SPLIT_FILES:
        raise ValueError(f"Spider has no split {split!r}; choose from {sorted(SPLIT_FILES)}")
    root = root or DATA_DIR / "spider"
    rows = [row for name in SPLIT_FILES[split] for row in json.loads((root / name).read_text())]
    return [
        Example(
            id=f"spider-{split}-{i}",
            dataset="spider",
            split=split,
            db_id=row["db_id"],
            question=row["question"],
            gold_sql=row["query"],
            db_path=root / "database" / row["db_id"] / f"{row['db_id']}.sqlite",
        )
        for i, row in enumerate(rows)
    ]
