"""`nl2sql data verify`: example counts per split, a database file per db_id, and every gold query executed
once. Gold that fails, times out or returns more rows than the eval's row cap is listed in
data/gold_failures.json (PLAN §5)."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

from nl2sql.data import DATA_DIR, load_examples
from nl2sql.execute.sqlite_exec import MAX_ROWS, execute_sqlite

# Published split sizes; a mismatch means a different release or a broken download.
EXPECTED = {
    ("spider", "train"): 8659,  # train_spider 7000 + train_others 1659
    ("spider", "dev"): 1034,
    ("bird", "train"): 9428,
    ("bird", "dev"): 1534,
    ("bird-mini", "dev"): 500,
}
SPLIT_FILES = {
    ("spider", "train"): "spider/train_spider.json",
    ("spider", "dev"): "spider/dev.json",
    ("bird", "train"): "bird/train/train.json",
    ("bird", "dev"): "bird/dev/dev.json",
    ("bird-mini", "dev"): "bird/minidev/mini_dev_sqlite.json",
}


def check_gold(db_path: str, sql: str, timeout_s: float, max_rows: int) -> dict:
    result = execute_sqlite(db_path, sql, timeout_s, max_rows)
    return {
        "ok": result.ok,
        "error": result.error,
        "timed_out": result.timed_out,
        "over_row_cap": result.truncated,
        "n_rows": len(result.rows),
    }


def verify(workers: int = 8, timeout_s: float = 30.0, max_rows: int = MAX_ROWS) -> dict[str, dict]:
    report: dict[str, dict] = {}
    failures: dict[str, list[dict]] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        for (dataset, split), path in SPLIT_FILES.items():
            if not (DATA_DIR / path).exists():
                continue  # BIRD train is an optional 9 GB download
            examples = load_examples(dataset, split)
            results = list(
                pool.map(
                    check_gold,
                    [str(e.db_path) for e in examples],
                    [e.gold_sql for e in examples],
                    [timeout_s] * len(examples),
                    [max_rows] * len(examples),
                    chunksize=16,
                )
            )
            name = f"{dataset}-{split}"
            failures[name] = [
                {"id": e.id, "db_id": e.db_id} | {k: r[k] for k in ("error", "timed_out", "over_row_cap")}
                for e, r in zip(examples, results, strict=True)
                if not r["ok"] or r["over_row_cap"]
            ]
            report[name] = {
                "n": len(examples),
                "expected": EXPECTED[(dataset, split)],
                "databases": len({e.db_id for e in examples}),
                "missing_databases": sorted({e.db_id for e in examples if not e.db_path.exists()}),
                "gold_errors": sum(not r["ok"] for r in results),  # timeouts included
                "gold_timeouts": sum(r["timed_out"] for r in results),
                "gold_over_row_cap": sum(r["over_row_cap"] for r in results),
                "max_gold_rows": max((r["n_rows"] for r in results if r["ok"]), default=0),
            }
    out = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "timeout_s": timeout_s,
        "max_rows": max_rows,
        "splits": failures,
    }
    (DATA_DIR / "gold_failures.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
