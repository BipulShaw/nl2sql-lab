import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import TypeAdapter

from nl2sql.config import RunConfig
from nl2sql.eval.harness import metrics, record_from_events
from nl2sql.linking.linker import SchemaLinker
from nl2sql.llm.base import Generation
from nl2sql.pipeline.events import Event, RunFailed, RunFinished, SchemaLinked
from nl2sql.pipeline.runner import REPAIR_PROMPT, Pipeline
from nl2sql.schema.introspect import introspect_sqlite

GOOD = "SELECT count(*) FROM item"


def fence(sql: str) -> str:
    return f"```sql\n{sql}\n```"


def count(text: str) -> int:
    return len(text.split())


def count_chat(messages: list[dict[str, str]]) -> int:
    return sum(count(m["content"]) for m in messages)


class ScriptedLLM:
    """Gives its replies in order and keeps the messages of every call."""

    def __init__(self, *replies: str, cached: bool = False) -> None:
        self.replies, self.cached = list(replies), cached
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        self.calls.append(messages)
        text = self.replies.pop(0)
        return Generation(text, count_chat(messages), count(text), 5000.0, "stop", cached=self.cached)


class Flat:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.full((len(texts), 4), 0.5)


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("db") / "shop.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE item (id INTEGER PRIMARY KEY, name TEXT, price REAL);"
            "INSERT INTO item VALUES (1, 'pen', 1.5), (2, 'ink', 4.0);"
        )
    conn.close()
    return path


def config(**changes: dict) -> RunConfig:
    return RunConfig(
        **(yaml.safe_load(Path("configs/base_9b_local.yaml").read_text()) | {"name": "t"} | changes)
    )


def run(db: Path, llm: ScriptedLLM, linker: SchemaLinker | None = None, **changes: dict) -> list:
    pipeline = Pipeline(config(**changes), llm, count, count_chat, linker)
    pipeline.schemas[db] = introspect_sqlite(db, "shop")  # keeps the test out of data/cache
    return list(pipeline.run(db, "shop", "How many items are there?"))


def types(events: list) -> list[str]:
    return [e.type for e in events]


def test_a_query_that_runs_finishes_the_run(db: Path) -> None:
    llm = ScriptedLLM(fence(GOOD))

    events = run(db, llm, guard={"enabled": True}, repair={"max_repairs": 2})

    assert types(events) == [
        "run_started",
        "prompt_built",
        "sql_generated",
        "guard_result",
        "executed",
        "run_finished",
    ]
    assert events[-1] == RunFinished(sql=GOOD, rows=1, attempts=1, latency_ms=events[-1].latency_ms)
    assert len(llm.calls) == 1


def test_a_guard_block_is_repaired_with_the_reason_in_a_new_turn(db: Path) -> None:
    llm = ScriptedLLM(fence("SELECT nme FROM item"), fence("SELECT name FROM item"))

    events = run(db, llm, guard={"enabled": True}, repair={"max_repairs": 2})
    record = record_from_events(events)

    assert "repair_started" in types(events) and isinstance(events[-1], RunFinished)
    first, second = llm.calls
    assert second[:-2] == first
    assert second[-2] == {"role": "assistant", "content": fence("SELECT nme FROM item")}
    assert second[-1] == {"role": "user", "content": REPAIR_PROMPT.format(reason="no such column: nme")}
    assert record["attempts"][0]["guard"].startswith("unknown_column")
    assert record["pred"] == "SELECT name FROM item" and record["repairs"] == 1 and record["pipeline_ok"]


def test_an_execution_error_is_repaired_without_the_guard(db: Path) -> None:
    llm = ScriptedLLM(fence("SELECT nosuch(price) FROM item"), fence(GOOD))

    events = run(db, llm, repair={"max_repairs": 1})

    assert "guard_result" not in types(events)
    assert "no such function: nosuch" in llm.calls[1][-1]["content"]
    assert isinstance(events[-1], RunFinished) and events[-1].attempts == 2


def test_the_run_gives_up_after_max_repairs_and_keeps_the_last_sql(db: Path) -> None:
    llm = ScriptedLLM(*(fence(f"SELECT bad{i}() FROM item") for i in range(3)))

    events = run(db, llm, repair={"max_repairs": 2})
    record = record_from_events(events)

    assert len(llm.calls) == 3
    assert isinstance(events[-1], RunFailed) and events[-1].attempts == 3
    assert record["pred"] == "SELECT bad2() FROM item"
    assert record["failure"] == "exec_error" and record["status"] == "ok" and not record["pipeline_ok"]


def test_without_repair_one_failure_ends_the_run(db: Path) -> None:
    llm = ScriptedLLM(fence("SELECT nme FROM item"))

    events = run(db, llm)

    assert types(events)[-2:] == ["executed", "run_failed"] and len(llm.calls) == 1


def test_a_reply_without_sql_is_repaired_and_recorded_as_no_sql(db: Path) -> None:
    events = run(db, ScriptedLLM("I cannot tell.", "Still no."), repair={"max_repairs": 1})
    record = record_from_events(events)

    assert events[-1].reason == "the reply contained no SQL query"
    assert record["status"] == "no_sql" and record["failure"] == "no_sql" and record["pred"] is None


def test_a_prompt_over_budget_is_never_sent(db: Path) -> None:
    llm = ScriptedLLM()
    pipeline = Pipeline(config(), llm, count, lambda messages: 4000)
    pipeline.schemas[db] = introspect_sqlite(db, "shop")

    events = list(pipeline.run(db, "shop", "How many items are there?"))
    record = record_from_events(events)

    assert events[-1].reason == "prompt_too_long" and llm.calls == []
    assert record["status"] == "prompt_too_long" and record["attempts"] == []


def test_cached_replies_count_at_their_original_latency(db: Path) -> None:
    events = run(db, ScriptedLLM(fence(GOOD), cached=True))

    assert events[-1].latency_ms >= 5000


def test_linking_emits_the_selected_tables(db: Path) -> None:
    events = run(
        db,
        ScriptedLLM(fence(GOOD)),
        linker=SchemaLinker(Flat(), lexical_bonus=0.0),
        linking={"enabled": True},
    )

    assert events[1] == SchemaLinked(tables=["item"], scores={"item": 1.0})

    with pytest.raises(ValueError, match="no linker"):
        Pipeline(config(linking={"enabled": True}), ScriptedLLM(), count, count_chat)


def test_events_survive_a_json_round_trip(db: Path) -> None:
    events = run(db, ScriptedLLM(fence("SELECT nme FROM item"), fence(GOOD)), repair={"max_repairs": 1})
    adapter = TypeAdapter(list[Event])

    assert adapter.validate_json(adapter.dump_json(events)) == events


def test_pipeline_metrics(db: Path) -> None:
    scored = {"correct": True, "correct_any_column_order": True, "gold_valid": True, "pred_parses": True}
    scored |= {"pred_exec": {"ok": True}, "difficulty": None}
    changes = {"guard": {"enabled": True}, "repair": {"max_repairs": 2}}
    first_try = record_from_events(run(db, ScriptedLLM(fence(GOOD)), **changes)) | scored
    repaired = record_from_events(run(db, ScriptedLLM(fence("SELECT x FROM item"), fence(GOOD)), **changes))
    records = [first_try, repaired | scored]

    m = metrics(records, config(**changes))

    assert m["guard_block_rate"] == 0.5 and m["repair_rate"] == 0.5 and m["repair_success_rate"] == 1.0
    assert m["link_recall"] is None
    assert m["tokens_in"] == sum(a["prompt_tokens"] for r in records for a in r["attempts"])
    assert metrics(records, None)["repair_rate"] is None
