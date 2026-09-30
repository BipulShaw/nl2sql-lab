"""Typed events of one pipeline run (PLAN §6.8). The eval harness builds its records from them, and the API
will stream them, so the benchmark and the demo exercise one code path."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class RunStarted(BaseModel):
    type: Literal["run_started"] = "run_started"
    db_id: str
    question: str


class SchemaLinked(BaseModel):
    type: Literal["schema_linked"] = "schema_linked"
    tables: list[str]  # picked by the linker, in schema order
    scores: dict[str, float]  # every table's score


class PromptBuilt(BaseModel):
    type: Literal["prompt_built"] = "prompt_built"
    tokens: int  # the whole chat prompt, counted with the model's tokenizer
    schema_tokens: int
    schema_level: str  # a serialize.Level name, pruned_columns or over_budget: what had to be shed


class SqlGenerated(BaseModel):
    type: Literal["sql_generated"] = "sql_generated"
    attempt: int  # 0 = first answer, 1.. = repairs
    raw: str
    sql: str | None  # None when the reply held no SQL
    source: str  # where the SQL came from: code block, bare statement, or none
    prompt_tokens: int
    output_tokens: int
    finish_reason: str
    latency_ms: float
    cached: bool
    think_leak: bool


class GuardChecked(BaseModel):
    type: Literal["guard_result"] = "guard_result"
    attempt: int
    ok: bool
    reason: str | None  # machine-readable, e.g. unknown_column:T1.nme


class Executed(BaseModel):
    type: Literal["executed"] = "executed"
    attempt: int
    rows: int
    elapsed_ms: float
    error: str | None
    timed_out: bool


class RepairStarted(BaseModel):
    type: Literal["repair_started"] = "repair_started"
    attempt: int  # the attempt this repair produces
    reason: str  # what the model is told failed


class RunFinished(BaseModel):
    type: Literal["run_finished"] = "run_finished"
    sql: str
    rows: int
    attempts: int
    latency_ms: float


class RunFailed(BaseModel):
    type: Literal["run_failed"] = "run_failed"
    reason: str  # prompt_too_long, or the last attempt's failure
    sql: str | None  # the last SQL tried, still scored by the harness
    attempts: int
    latency_ms: float


Event = Annotated[
    RunStarted
    | SchemaLinked
    | PromptBuilt
    | SqlGenerated
    | GuardChecked
    | Executed
    | RepairStarted
    | RunFinished
    | RunFailed,
    Field(discriminator="type"),
]
