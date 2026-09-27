"""`nl2sql eval`: generate SQL for a dataset split, execute and score it, and record the run (PLAN §6.9).

Generation is sequential (one local GPU server); execution and scoring run in a process pool as each
prediction arrives. Output: results/runs/<run_id>/predictions.jsonl (gitignored) and manifest.json."""

import json
import logging
import multiprocessing
import statistics
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from rich.progress import Progress

from nl2sql.config import ExecConfig, RunConfig
from nl2sql.data import DATA_DIR, Example, load_examples, stratified_sample
from nl2sql.eval.ex_metric import score_prediction
from nl2sql.llm.cache import ResponseCache
from nl2sql.llm.ollama import OllamaLLM
from nl2sql.prompting.builder import build_messages, extract_sql
from nl2sql.schema.introspect import load_schema
from nl2sql.schema.serialize import SerializedSchema, serialize_schema

RUNS_DIR = Path("results/runs")
RESPONSE_CACHE = DATA_DIR / "cache" / "llm_responses.sqlite"


class Predictor:
    """Full schema → prompt → model → extracted SQL, for one example at a time."""

    def __init__(self, config: RunConfig, use_cache: bool) -> None:
        from nl2sql.llm.tokens import ChatTokenizer  # imports transformers, which gold runs do not need

        self.config = config
        self.tokenizer = ChatTokenizer(config.tokenizer.id, config.tokenizer.revision)
        self.llm = OllamaLLM(
            config.model_id,
            self.tokenizer.count_chat,
            base_url=config.base_url,
            num_ctx=config.context_window,
            max_new_tokens=config.decoding.max_new_tokens,
            seed=config.seed,
            cache=ResponseCache(RESPONSE_CACHE) if use_cache else None,
        )
        if self.llm.info["quantization"] != config.quantization:
            raise RuntimeError(
                f"Ollama serves {config.model_id} as {self.llm.info['quantization']}, "
                f"but the config says {config.quantization}"
            )
        self.schemas: dict[Path, SerializedSchema] = {}  # no linking yet, so one serialization per database

    def schema_for(self, example: Example) -> SerializedSchema:
        if example.db_path not in self.schemas:
            self.schemas[example.db_path] = serialize_schema(
                load_schema(example.db_path, example.db_id),
                self.tokenizer.count,
                self.config.prompt.schema_token_budget,
                include_samples=self.config.prompt.include_samples,
            )
        return self.schemas[example.db_path]

    def predict(self, example: Example) -> dict:
        schema = self.schema_for(example)
        messages = build_messages(example.db_id, schema.text, example.question, example.evidence)
        prompt_tokens = self.tokenizer.count_chat(messages)
        record = {
            "status": "ok",
            "pred": None,
            "attempts": [],
            "schema_level": schema.level,
            "schema_tokens": schema.tokens,
            "prompt_tokens": prompt_tokens,
            "output_tokens": 0,
            "finish_reason": None,
            "latency_ms": None,
            "cached": False,
            "think_leak": False,
        }
        if prompt_tokens + self.config.decoding.max_new_tokens > self.config.prompt.max_total_tokens:
            record["status"] = "prompt_too_long"  # not sent; counts as wrong
            return record
        generation = self.llm.generate(messages)
        extraction = extract_sql(generation.text)
        record.update(
            status="ok" if extraction.sql else "no_sql",
            pred=extraction.sql,
            attempts=[{"raw": generation.text, "sql": extraction.sql, "source": extraction.source}],
            output_tokens=generation.output_tokens,
            finish_reason=generation.finish_reason,
            latency_ms=round(generation.latency_ms, 1),
            cached=generation.cached,
            think_leak=extraction.think_leak,
        )
        return record


def example_fields(example: Example) -> dict:
    return {
        "id": example.id,
        "db_id": example.db_id,
        "question": example.question,
        "evidence": example.evidence,
        "difficulty": example.difficulty,
        "gold": example.gold_sql,
    }


def run_eval(
    dataset: str,
    split: str,
    config: RunConfig | None,
    limit: int | None = None,
    seed: int = 42,
    gold: bool = False,
    workers: int = 8,
    use_cache: bool = True,
) -> dict:
    """Score a model config, or with gold=True the gold SQL itself (the harness sanity check).
    Returns the manifest, which is also written next to predictions.jsonl."""
    if config is None and not gold:
        raise ValueError("a config is required unless gold=True")
    logging.getLogger("sqlglot").setLevel(logging.ERROR)  # it warns on every query it cannot fully parse
    examples = load_examples(dataset, split)
    n_split = len(examples)
    if limit is not None:
        examples = stratified_sample(examples, limit, seed)
    exec_config = config.exec if config else ExecConfig()
    name = "gold" if gold or config is None else config.name
    started = datetime.now(UTC)
    git = git_state()  # at the start: the code this run loads, whatever is edited while it runs
    server: dict = {"backend": "gold_passthrough"}
    records: list[dict] = []
    # spawn, not fork: the parent may hold tokenizer threads and an open HTTP client
    with (
        ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool,
        Progress() as progress,
    ):
        predictor = Predictor(config, use_cache) if config is not None and not gold else None
        if predictor:
            server = predictor.llm.info
        task = progress.add_task("generating" if predictor else "queueing", total=len(examples))
        futures = []
        for example in examples:
            record = example_fields(example)
            record.update(
                predictor.predict(example) if predictor else {"status": "ok", "pred": example.gold_sql}
            )
            records.append(record)
            futures.append(
                pool.submit(
                    score_prediction,
                    str(example.db_path),
                    example.gold_sql,
                    record["pred"],
                    exec_config.timeout_s,
                    exec_config.max_rows,
                )
            )
            progress.advance(task)
        task = progress.add_task("scoring", total=len(futures))
        for record, future in zip(records, futures, strict=True):
            record.update(future.result())
            progress.advance(task)
    ended = datetime.now(UTC)

    run_id = f"{started:%Y%m%d-%H%M%S}_{dataset}-{split}_{name}_n{len(records)}"
    manifest = {
        "run_id": run_id,
        "mode": "model" if predictor else "gold_passthrough",
        "dataset": dataset,
        "split": split,
        "n": len(records),
        "n_split": n_split,
        "sample": {"limit": limit, "seed": seed, "method": "stratified"} if limit is not None else None,
        "config_name": config.name if predictor and config else None,
        "backend": server.get("backend"),
        "model": server.get("model"),
        "model_digest": server.get("digest"),
        "quantization": server.get("quantization"),
        "adapter": None,
        "thinking": False,
        "server": server,
        "config": config.model_dump() if predictor and config else None,
        "exec": exec_config.model_dump(),
        "git": git,
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": ended.isoformat(timespec="seconds"),
        "duration_s": round((ended - started).total_seconds(), 1),
        "metrics": metrics(records),
        "counts": counts(records),
    }
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def ratio(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def metrics(records: list[dict]) -> dict:
    n = len(records)
    valid_gold = [r for r in records if r["gold_valid"]]
    by_difficulty = {}
    for difficulty in sorted({r["difficulty"] for r in records if r["difficulty"]}):
        group = [r for r in records if r["difficulty"] == difficulty]
        by_difficulty[difficulty] = {
            "n": len(group),
            "ex": ratio(sum(r["correct"] for r in group), len(group)),
        }
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    return {
        "ex": ratio(sum(r["correct"] for r in records), n),  # failing or over-cap gold counts as wrong
        "ex_valid_gold": ratio(sum(r["correct"] for r in valid_gold), len(valid_gold)),
        # secondary: the columns may come in any order, as Spider's official evaluator allows (ADR-013)
        "ex_any_column_order": ratio(sum(r["correct_any_column_order"] for r in records), n),
        "ex_by_difficulty": by_difficulty or None,
        "parse_valid_rate": ratio(sum(r["pred_parses"] for r in records), n),
        "exec_error_rate": ratio(sum(bool(r["pred_exec"]) and not r["pred_exec"]["ok"] for r in records), n),
        "guard_block_rate": None,  # guard, repair and linking arrive in Phase 2
        "repair_rate": None,
        "repair_success_rate": None,
        "link_recall": None,
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        "tokens_in": sum(r.get("prompt_tokens", 0) for r in records),
        "tokens_out": sum(r.get("output_tokens", 0) for r in records),
        "api_cost_usd": 0.0,
    }


def counts(records: list[dict]) -> dict:
    def pred_flag(r: dict, key: str) -> bool:
        return bool(r["pred_exec"] and r["pred_exec"][key])

    return {
        "correct": sum(r["correct"] for r in records),
        "correct_any_column_order": sum(r["correct_any_column_order"] for r in records),
        "gold_errors": sum(not r["gold_exec"]["ok"] for r in records),
        "gold_timeouts": sum(r["gold_exec"]["timed_out"] for r in records),
        "no_sql": sum(r["status"] == "no_sql" for r in records),
        "prompt_too_long": sum(r["status"] == "prompt_too_long" for r in records),
        "pred_errors": sum(bool(r["pred_exec"]) and not r["pred_exec"]["ok"] for r in records),
        "pred_timeouts": sum(pred_flag(r, "timed_out") for r in records),
        "gold_over_row_cap": sum(r["gold_exec"]["truncated"] for r in records),
        "pred_more_rows_than_gold": sum(r["gold_valid"] and pred_flag(r, "truncated") for r in records),
        "ordered_comparisons": sum(r["ordered"] for r in records),
        "think_leaks": sum(r.get("think_leak", False) for r in records),
        "finish_length": sum(r.get("finish_reason") == "length" for r in records),
        "cache_hits": sum(r.get("cached", False) for r in records),
        "schema_levels": dict(Counter(r["schema_level"] for r in records if r.get("schema_level"))),
        "sql_source": dict(Counter(a["source"] for r in records for a in r.get("attempts", [])[:1])),
    }


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD") or None,
        # The tree hash names the code itself; it survives history rewrites that only change commit metadata.
        "tree": git("rev-parse", "HEAD^{tree}") or None,
        # Run outputs are not code: earlier runs' manifests, uncommitted, do not make this run dirty.
        "dirty": bool(git("status", "--porcelain", "--", ".", ":(exclude)results")),
    }
