"""Run configuration from configs/*.yaml. Settings a phase has not implemented yet are rejected, not ignored,
so a config can never claim a feature the run did not use."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nl2sql.execute.sqlite_exec import MAX_ROWS
from nl2sql.linking.embed import EMBED_MODEL, EMBED_REVISION


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TokenizerConfig(Strict):
    id: str
    revision: str


class PromptConfig(Strict):
    schema_token_budget: int = 1536
    max_total_tokens: int = 2048  # prompt + max_new_tokens; the example is skipped (and wrong) if over
    include_samples: bool = True


class DecodingConfig(Strict):
    max_new_tokens: int = 256
    greedy: Literal[True] = True


class LinkingConfig(Strict):
    enabled: bool = False
    top_k: int = Field(4, ge=1)
    all_tables_up_to: int = 6  # a database with at most this many tables keeps them all
    fk_hops: int = Field(1, ge=0)  # add the foreign-key neighbors of the top k, this many hops out
    lexical_bonus: float = Field(0.1, ge=0)  # added to a table the question names; chosen on Spider train
    query_instruction: bool = True  # bge's retrieval prefix on the question
    embed_model: str = EMBED_MODEL
    embed_revision: str = EMBED_REVISION


class RepairConfig(Strict):
    max_repairs: int = Field(0, ge=0, le=5)
    on_empty_result: Literal[False] = False  # BIRD gold can be empty; off unless measured (PLAN §6.7)


class GuardConfig(Strict):
    enabled: bool = False
    dialect: Literal["sqlite"] = "sqlite"
    inject_limit: Literal[False] = False  # never in eval: a LIMIT changes the answer


class ExecConfig(Strict):
    timeout_s: float = 30.0
    max_rows: int = MAX_ROWS  # cap on the gold result; predictions stop one row past the gold's count


class RunConfig(Strict):
    name: str
    model_id: str
    backend: Literal["ollama"]
    base_url: str = "http://localhost:11434"
    quantization: str  # what the server must report for model_id; checked at run start
    context_window: int  # sent as num_ctx on every call
    thinking: Literal[False]
    tokenizer: TokenizerConfig  # counts prompt tokens exactly as the server does (ADR-012)
    linking: LinkingConfig = LinkingConfig()
    prompt: PromptConfig = PromptConfig()
    decoding: DecodingConfig = DecodingConfig()
    repair: RepairConfig = RepairConfig()
    guard: GuardConfig = GuardConfig()
    exec: ExecConfig = ExecConfig()
    seed: int = 42

    @model_validator(mode="after")
    def budgets_fit(self) -> "RunConfig":
        if self.prompt.max_total_tokens > self.context_window:
            raise ValueError("prompt.max_total_tokens must not exceed context_window")
        if self.prompt.schema_token_budget + self.decoding.max_new_tokens > self.prompt.max_total_tokens:
            raise ValueError("schema_token_budget + max_new_tokens must fit in prompt.max_total_tokens")
        return self


def load_config(path: Path) -> RunConfig:
    return RunConfig(name=path.stem, **(yaml.safe_load(path.read_text(encoding="utf-8")) or {}))
