import json
from collections import Counter
from pathlib import Path

from nl2sql.data.bird import load_bird
from nl2sql.data.common import Example, stratified_sample
from nl2sql.data.spider import load_spider


def make_examples(n_per_stratum: dict[str, int]) -> list[Example]:
    return [
        Example(
            id=f"x-{d}-{i}",
            dataset="bird",
            split="dev",
            db_id="db",
            question="q",
            gold_sql="SELECT 1",
            difficulty=d,
            db_path=Path("db.sqlite"),
        )
        for d, n in n_per_stratum.items()
        for i in range(n)
    ]


def test_stratified_sample_keeps_shares_and_order() -> None:
    examples = make_examples({"simple": 60, "moderate": 30, "challenging": 10})

    sample = stratified_sample(examples, 20, seed=42)

    assert Counter(e.difficulty for e in sample) == {"simple": 12, "moderate": 6, "challenging": 2}
    positions = [examples.index(e) for e in sample]
    assert positions == sorted(positions)


def test_stratified_sample_is_deterministic_and_exact_size() -> None:
    examples = make_examples({"simple": 7, "moderate": 5, "challenging": 3})

    first, second = stratified_sample(examples, 11, seed=42), stratified_sample(examples, 11, seed=42)

    assert first == second and len(first) == 11
    assert stratified_sample(examples, 11, seed=7) != first
    assert stratified_sample(examples, 100) == examples


def test_spider_loader(tmp_path: Path) -> None:
    rows = [
        {"db_id": "concert_singer", "question": "How many singers?", "query": "SELECT count(*) FROM singer"}
    ]
    (tmp_path / "dev.json").write_text(json.dumps(rows))

    [example] = load_spider("dev", root=tmp_path)

    assert example.id == "spider-dev-0" and example.gold_sql == "SELECT count(*) FROM singer"
    assert example.db_path == tmp_path / "database" / "concert_singer" / "concert_singer.sqlite"
    assert example.evidence == "" and example.difficulty is None


def test_bird_loader(tmp_path: Path) -> None:
    rows = [
        {
            "question_id": 7,
            "db_id": "financial",
            "question": "How many accounts?",
            "evidence": "account refers to account_id",
            "SQL": "SELECT count(account_id) FROM account",
            "difficulty": "simple",
        }
    ]
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "dev.json").write_text(json.dumps(rows))

    [example] = load_bird("dev", root=tmp_path)

    assert example.id == "bird-dev-7" and example.difficulty == "simple"
    assert example.evidence == "account refers to account_id"
    assert example.db_path == tmp_path / "dev" / "dev_databases" / "financial" / "financial.sqlite"
