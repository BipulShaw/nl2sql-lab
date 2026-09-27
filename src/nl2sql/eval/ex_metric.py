"""Execution accuracy (EX): do the gold and the predicted query return the same rows? (PLAN §6.9)

Rows are compared as multisets (duplicates count), and as ordered lists when the gold query has a top-level
ORDER BY; columns must come in the same order. A prediction that errors or times out is wrong. Neither
official script does exactly this: BIRD's compares sets of rows, and Spider's test-suite script drops
DISTINCT, accepts any column order and compares row order whenever the gold contains "order by" (ADR-013).
Each prediction is also scored with any column order allowed, to show what the strict rule costs."""

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


def results_match_any_column_order(gold: list[tuple], pred: list[tuple], ordered: bool) -> bool:
    """`results_match` with pred's columns allowed in another order: one reordering, applied to every row.

    Each gold column can only take a pred column holding the same values, which usually leaves one choice.
    Where several fit, identical columns are tried once and a partial reordering is dropped as soon as the
    columns placed so far disagree with the gold's, so wide results with many alike columns stay fast."""
    if len(gold) != len(pred) or (gold and len(gold[0]) != len(pred[0])):
        return False
    if not gold:
        return True
    width = len(gold[0])
    gold_counts = [Counter(row[i] for row in gold) for i in range(width)]
    pred_columns = [tuple(row[j] for row in pred) for j in range(width)]
    pred_counts = [Counter(column) for column in pred_columns]
    candidates = [[j for j in range(width) if pred_counts[j] == gold_counts[i]] for i in range(width)]

    def extend(order: tuple[int, ...]) -> bool:
        k = len(order)
        if k == width:
            return results_match(gold, [tuple(row[j] for j in order) for row in pred], ordered)
        tried = set()
        for j in candidates[k]:
            if j in order or pred_columns[j] in tried:
                continue  # a column identical to one already tried here leads to the same rows
            tried.add(pred_columns[j])
            placed = (*order, j)
            if len(candidates[k]) > 1 and not results_match(
                [row[: k + 1] for row in gold], [tuple(row[c] for c in placed) for row in pred], ordered
            ):
                continue
            if extend(placed):
                return True
        return False

    return extend(())


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
    # truncated: more rows than the gold
    comparable = gold_valid and pred is not None and pred.ok and not pred.truncated
    correct = comparable and results_match(gold.rows, pred.rows, ordered)
    return {
        "correct": bool(correct),
        "correct_any_column_order": bool(
            correct or (comparable and results_match_any_column_order(gold.rows, pred.rows, ordered))
        ),
        "gold_valid": gold_valid,
        "ordered": ordered,
        "pred_parses": pred_sql is not None and parse_sqlite(pred_sql) is not None,
        "gold_exec": exec_summary(gold),
        "pred_exec": exec_summary(pred) if pred is not None else None,
    }
