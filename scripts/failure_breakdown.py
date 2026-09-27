"""Sort a run's wrong predictions by how their result differs from the gold's, to see where EX is lost.

Each wrong example runs again (gold and prediction, with the run's timeout and row cap, and this time the
prediction's rows are all fetched). It goes in the first category that fits:

  gold_invalid      the gold query fails, times out or is over the row cap, so nothing can match it
  no_sql            the reply had no SQL
  prompt_too_long   the prompt was over the config's token limit and was not sent
  pred_error        the prediction fails or times out
  matches_on_rerun  it matches now: the query's result is not deterministic (expected: never)
  more_columns      more columns than the gold (fewer_columns: fewer)
  column_order      the right rows once the prediction's columns are reordered; Spider's official evaluator
                    accepts these
  row_order         the right rows in the wrong order, where the gold's ORDER BY makes order count
  set_equal         the same distinct rows, duplicated differently
  more_rows         same columns, more rows than the gold (fewer_rows: fewer)
  different_values  same shape, different values

BIRD's official evaluator compares sets of rows, so it accepts row_order and set_equal too (ADR-013).

Usage: uv run python scripts/failure_breakdown.py results/runs/<run_id> [--show CATEGORY ...]
Needs the run's predictions.jsonl, which is not committed; re-running the run's eval command recreates it.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from nl2sql.data import load_examples
from nl2sql.eval.ex_metric import results_match, results_match_any_column_order
from nl2sql.execute.sqlite_exec import ExecResult, execute_sqlite

CATEGORIES = (
    "gold_invalid",
    "no_sql",
    "prompt_too_long",
    "pred_error",
    "matches_on_rerun",
    "more_columns",
    "fewer_columns",
    "column_order",
    "row_order",
    "set_equal",
    "more_rows",
    "fewer_rows",
    "different_values",
)


def categorize(record: dict, gold: ExecResult, pred: ExecResult | None) -> str:
    if not record["gold_valid"] or not gold.ok or gold.truncated:
        return "gold_invalid"
    if pred is None:
        return record["status"]  # no_sql or prompt_too_long: nothing was executed
    if not pred.ok:
        return "pred_error"
    g, p, ordered = gold.rows, pred.rows, record["ordered"]
    if results_match(g, p, ordered):
        return "matches_on_rerun"
    if len(pred.columns) != len(gold.columns):
        return "more_columns" if len(pred.columns) > len(gold.columns) else "fewer_columns"
    if results_match_any_column_order(g, p, ordered):
        return "column_order"
    if ordered and results_match(g, p, ordered=False):
        return "row_order"
    if set(g) == set(p):
        return "set_equal"
    if len(p) != len(g):
        return "more_rows" if len(p) > len(g) else "fewer_rows"
    return "different_values"


def describe(result: ExecResult | None, rows: int = 3, width: int = 160) -> str:
    if result is None:
        return "not run"
    if not result.ok:
        return result.error or "error"
    preview = ", ".join(repr(row) for row in result.rows[:rows])
    more = f", ... ({len(result.rows)} rows)" if len(result.rows) > rows else ""
    return f"{result.columns}: {preview[:width]}{more}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--show", nargs="+", default=[], choices=CATEGORIES, metavar="CATEGORY")
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    timeout_s, max_rows = manifest["exec"]["timeout_s"], manifest["exec"]["max_rows"]
    db_paths = {e.id: e.db_path for e in load_examples(manifest["dataset"], manifest["split"])}
    with (args.run_dir / "predictions.jsonl").open(encoding="utf-8") as f:
        wrong = [r for r in map(json.loads, f) if not r["correct"]]

    categories: Counter[str] = Counter()
    for record in wrong:
        db = db_paths[record["id"]]
        gold = execute_sqlite(db, record["gold"], timeout_s, max_rows)
        pred = execute_sqlite(db, record["pred"], timeout_s, max_rows) if record["pred"] else None
        category = categorize(record, gold, pred)
        categories[category] += 1
        if category in args.show:
            print(f"--- {record['id']} [{category}] {record['db_id']}")
            print(f"Q: {record['question']}")
            if record["evidence"]:
                print(f"E: {record['evidence']}")
            print(f"G: {' '.join(record['gold'].split())}")
            print(f"P: {' '.join((record['pred'] or '').split())}")
            print(f"   gold {describe(gold)}")
            print(f"   pred {describe(pred)}")

    n = manifest["n"]
    print(f"\n{manifest['run_id']}: {n - len(wrong)}/{n} correct, {len(wrong)} wrong\n")
    print("| Category | Wrong | Share of wrong | EX points |")
    print("|---|---|---|---|")
    for category in CATEGORIES:
        if categories[category]:
            k = categories[category]
            print(f"| {category} | {k} | {k / len(wrong):.0%} | {100 * k / n:.1f} |")


if __name__ == "__main__":
    main()
