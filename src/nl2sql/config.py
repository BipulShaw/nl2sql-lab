"""Run configuration from configs/*.yaml. Settings a phase has not implemented yet are rejected, not ignored,
so a config can never claim a feature the run did not use."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from nl2sql.execute.sqlite_exec import MAX_ROWS


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
    enabled: Literal[False] = False  # schema linking arrives in Phase 2


class RepairConfig(Strict):
    max_repairs: Literal[0] = 0  # the repair loop arrives in Phase 2


class GuardConfig(Strict):
    enabled: Literal[False] = False  # the sqlglot guard arrives in Phase 2
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
