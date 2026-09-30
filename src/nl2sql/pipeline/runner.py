"""One question through the pipeline (PLAN §6.7, §6.8): link → serialize → prompt → generate → guard →
execute, and on a guard block, an execution error or a timeout, a repair turn, up to max_repairs times.

`Pipeline.run` yields a typed event at every step. It never sees gold SQL: it executes the model's query only
to learn whether it fails, with a small row cap, and the harness scores the final query separately."""

import time
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Protocol

from nl2sql.config import LinkingConfig, RunConfig
from nl2sql.execute import execute_sqlite
from nl2sql.guard.sql_guard import check_sql
from nl2sql.linking.linker import SchemaLinker, Scores
from nl2sql.llm.base import Generation
from nl2sql.pipeline.events import (
    Event,
    Executed,
    GuardChecked,
    PromptBuilt,
    RepairStarted,
    RunFailed,
    RunFinished,
    RunStarted,
    SchemaLinked,
    SqlGenerated,
)
from nl2sql.prompting.builder import build_messages, extract_sql
from nl2sql.schema.introspect import load_schema
from nl2sql.schema.model import Schema
from nl2sql.schema.serialize import SerializedSchema, serialize_schema

REPAIR_PROMPT = "The query failed: {reason}. Fix it and return only the corrected SQL."
EXEC_MAX_ROWS = 10_000  # enough to tell success from failure (PLAN §6.6); scoring uses its own cap


class LLM(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> Generation: ...


def make_linker(config: LinkingConfig) -> SchemaLinker:
    from nl2sql.linking.embed import BgeEmbedder  # imports torch, so only runs that link pay for it

    return SchemaLinker(
        BgeEmbedder(config.embed_model, config.embed_revision),
        top_k=config.top_k,
        all_tables_up_to=config.all_tables_up_to,
        fk_hops=config.fk_hops,
        lexical_bonus=config.lexical_bonus,
        query_instruction=config.query_instruction,
    )


class Pipeline:
    def __init__(
        self,
        config: RunConfig,
        llm: LLM,
        count_tokens: Callable[[str], int],
        count_chat_tokens: Callable[[list[dict[str, str]]], int],
        linker: SchemaLinker | None = None,
    ) -> None:
        if config.linking.enabled and linker is None:
            raise ValueError("the config enables linking but no linker was given")
        self.config, self.llm, self.linker = config, llm, linker if config.linking.enabled else None
        self.count_tokens, self.count_chat_tokens = count_tokens, count_chat_tokens
        self.schemas: dict[Path, Schema] = {}
        self.full_schemas: dict[Path, SerializedSchema] = {}  # without linking: one serialization per db

    def schema(self, db_path: Path, db_id: str) -> Schema:
        if db_path not in self.schemas:
            self.schemas[db_path] = load_schema(db_path, db_id)
        return self.schemas[db_path]

    def serialize(
        self, db_path: Path, schema: Schema, linked: tuple[list[str], Scores] | None
    ) -> SerializedSchema:
        if linked is None and db_path in self.full_schemas:
            return self.full_schemas[db_path]
        serialized = serialize_schema(
            schema,
            self.count_tokens,
            self.config.prompt.schema_token_budget,
            tables=None if linked is None else linked[0],
            include_samples=self.config.prompt.include_samples,
            column_scores=None if linked is None else linked[1].columns,
        )
        if linked is None:
            self.full_schemas[db_path] = serialized
        return serialized

    def run(self, db_path: Path, db_id: str, question: str, evidence: str = "") -> Iterator[Event]:
        started = time.perf_counter()
        cache_credit_ms = 0.0  # cached replies count at their original latency, so reruns report the same

        def latency_ms() -> float:
            return round((time.perf_counter() - started) * 1000 + cache_credit_ms, 1)

        yield RunStarted(db_id=db_id, question=question)
        schema = self.schema(db_path, db_id)
        linked = None
        if self.linker is not None:
            linked = tables, scores = self.linker.link(schema, question, evidence)
            yield SchemaLinked(tables=tables, scores={t: round(s, 4) for t, s in scores.tables.items()})
        serialized = self.serialize(db_path, schema, linked)
        messages = build_messages(db_id, serialized.text, question, evidence)
        prompt_tokens = self.count_chat_tokens(messages)
        yield PromptBuilt(
            tokens=prompt_tokens, schema_tokens=serialized.tokens, schema_level=serialized.level
        )
        if prompt_tokens + self.config.decoding.max_new_tokens > self.config.prompt.max_total_tokens:
            yield RunFailed(reason="prompt_too_long", sql=None, attempts=0, latency_ms=latency_ms())
            return

        last_sql: str | None = None
        failure = reply = ""
        for attempt in range(self.config.repair.max_repairs + 1):
            if attempt:
                # Repair turns may run past max_total_tokens (a first-turn budget, shared with training); the
                # client still refuses any prompt the context window cannot hold.
                yield RepairStarted(attempt=attempt, reason=failure)
                messages = [
                    *messages,
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": REPAIR_PROMPT.format(reason=failure)},
                ]
            call_started = time.perf_counter()
            generation = self.llm.generate(messages)
            if generation.cached:
                cache_credit_ms += generation.latency_ms - (time.perf_counter() - call_started) * 1000
            reply = generation.text
            extraction = extract_sql(reply)
            yield SqlGenerated(
                attempt=attempt,
                raw=reply,
                sql=extraction.sql,
                source=extraction.source,
                prompt_tokens=generation.prompt_tokens,
                output_tokens=generation.output_tokens,
                finish_reason=generation.finish_reason,
                latency_ms=round(generation.latency_ms, 1),
                cached=generation.cached,
                think_leak=extraction.think_leak,
            )
            last_sql = extraction.sql or last_sql
            outcome = yield from self.check(extraction.sql, schema, db_path, attempt)
            if isinstance(outcome, int):
                sql = extraction.sql or ""  # never empty: check() fails a reply without SQL
                yield RunFinished(sql=sql, rows=outcome, attempts=attempt + 1, latency_ms=latency_ms())
                return
            failure = outcome
        yield RunFailed(reason=failure, sql=last_sql, attempts=attempt + 1, latency_ms=latency_ms())

    def check(
        self, sql: str | None, schema: Schema, db_path: Path, attempt: int
    ) -> Generator[Event, None, int | str]:
        """The row count if the query ran, else why it failed, worded for the repair prompt."""
        if sql is None:
            return "the reply contained no SQL query"
        if self.config.guard.enabled:
            verdict = check_sql(sql, schema, self.config.guard.dialect)
            yield GuardChecked(attempt=attempt, ok=verdict.ok, reason=verdict.reason)
            if not verdict.ok:
                return verdict.message or verdict.reason or "blocked"
        timeout_s = self.config.exec.timeout_s
        result = execute_sqlite(db_path, sql, timeout_s, EXEC_MAX_ROWS)
        yield Executed(
            attempt=attempt,
            rows=len(result.rows),
            elapsed_ms=round(result.elapsed_ms, 1),
            error=result.error,
            timed_out=result.timed_out,
        )
        if result.timed_out:
            return f"it did not finish within {timeout_s:g} seconds"
        if result.error is not None:
            return result.error.split(": ", 1)[-1]  # SQLite's message, without the Python exception class
        return len(result.rows)
