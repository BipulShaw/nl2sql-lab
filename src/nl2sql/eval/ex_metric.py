"""Execution accuracy (EX): do the gold and the predicted query return the same rows? (PLAN §6.9)

Rows are compared as multisets (duplicates count), and as ordered lists when the gold query has a top-level
ORDER BY; columns must come in the same order. A prediction that errors or times out is wrong. Neither
official script does exactly this: BIRD's compares sets of rows, and Spider's test-suite script drops
DISTINCT, accepts any column order and compares row order whenever the gold contains "order by" (ADR-013)."""

import re
from collections import Counter

import sqlglot
from sqlglot import exp

from nl2sql.execute.sqlite_exec import ExecResult, execute_sqlite

ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def top_level_text(sql: str) -> str:
    """The query minus quoted strings/identifiers and anything inside parentheses."""
    out, depth, closing = [], 0, None
    for ch in sql:
        if closing:
            if ch == closing:
                closing = None
        elif ch in "'\"`[":
            closing = "]" if ch == "[" else ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def parse_sqlite(sql: str) -> exp.Expression | None:
    """One statement parsed by sqlglot; None if it fails to parse or only parses as a catch-all Command."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # ParseError, TokenError, and the odd internal error on unusual input
        return None
    return None if tree is None or isinstance(tree, exp.Command) else tree


def has_top_level_order_by(sql: str) -> bool:
    """True when the outermost query sorts its result, which makes row order part of the answer."""
    tree = parse_sqlite(sql)
    if tree is None:  # a few gold queries use syntax sqlglot rejects; scan outside parentheses instead
        return bool(ORDER_BY.search(top_level_text(sql)))
    return tree.args.get("order") is not None


def results_match(gold: list[tuple], pred: list[tuple], ordered: bool) -> bool:
    return gold == pred if ordered else Counter(gold) == Counter(pred)


def exec_summary(result: ExecResult) -> dict:
    return {
        "ok": result.ok,
        "error": result.error,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
        "n_rows": len(result.rows),
        "elapsed_ms": round(result.elapsed_ms, 1),
    }


def score_prediction(
    db_path: str, gold_sql: str, pred_sql: str | None, timeout_s: float, max_rows: int
) -> dict:
    """Execute gold and prediction on the same database and compare. Runs in a worker process.

    `max_rows` caps the gold result; gold over the cap cannot be scored, so it counts as wrong like gold that
    fails. The prediction is fetched only one row past the gold's row count: more rows than the gold is wrong
    whatever they are, so every comparison is between complete results and a runaway query never loads."""
    gold = execute_sqlite(db_path, gold_sql, timeout_s, max_rows)
    gold_valid = gold.ok and not gold.truncated
    pred_cap = len(gold.rows) if gold_valid else max_rows
    pred = execute_sqlite(db_path, pred_sql, timeout_s, pred_cap) if pred_sql else None
    ordered = has_top_level_order_by(gold_sql)
    return {
        "correct": bool(
            gold_valid
            and pred is not None
            and pred.ok
            and not pred.truncated  # more rows than the gold
            and results_match(gold.rows, pred.rows, ordered)
        ),
        "gold_valid": gold_valid,
        "ordered": ordered,
        "pred_parses": pred_sql is not None and parse_sqlite(pred_sql) is not None,
        "gold_exec": exec_summary(gold),
        "pred_exec": exec_summary(pred) if pred is not None else None,
    }
