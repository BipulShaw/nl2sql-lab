"""Dataset loaders. `load_examples("spider" | "bird" | "bird-mini", split)` is the single entry point."""

from nl2sql.data.bird import load_bird
from nl2sql.data.common import DATA_DIR, Example, stratified_sample
from nl2sql.data.spider import load_spider

DATASETS = ("spider", "bird", "bird-mini")


def load_examples(dataset: str, split: str) -> list[Example]:
    if dataset == "spider":
        return load_spider(split)
    if dataset == "bird":
        return load_bird(split)
    if dataset == "bird-mini":
        if split != "dev":
            raise ValueError("bird-mini only has a dev split")
        return load_bird("minidev")
    raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")


__all__ = ["DATASETS", "DATA_DIR", "Example", "load_examples", "stratified_sample"]
