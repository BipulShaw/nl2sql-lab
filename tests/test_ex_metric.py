import sqlite3
from pathlib import Path

import pytest

from nl2sql.eval.ex_metric import has_top_level_order_by, results_match, score_prediction, top_level_text


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT name FROM singer ORDER BY age DESC", True),
        ("select name from singer order by age limit 1", True),
        ("SELECT name FROM (SELECT name, age FROM singer ORDER BY age) AS t", False),
        ("SELECT name FROM singer WHERE age = (SELECT max(age) FROM singer ORDER BY 1)", False),
        ("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1", True),
        ("WITH x AS (SELECT a FROM t ORDER BY a) SELECT a FROM x", False),
        ("WITH x AS (SELECT a FROM t) SELECT a FROM x ORDER BY a", True),
        ("SELECT 'order by' FROM t", False),
        ("SELECT `order by` FROM t", False),
    ],
)
def test_top_level_order_by(sql: str, expected: bool) -> None:
    assert has_top_level_order_by(sql) is expected


def test_fallback_scan_ignores_parentheses_and_quotes() -> None:
    text = top_level_text("SELECT a FROM (SELECT a FROM t ORDER BY a) WHERE b = 'x) order by (' ORDER BY a")

    assert text.lower().count("order by") == 1


def test_unordered_comparison_is_a_multiset() -> None:
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not results_match([(1,), (1,)], [(1,)], ordered=False)  # duplicates count


def test_ordered_comparison_needs_the_same_order() -> None:
    assert results_match([(1,), (2,)], [(1,), (2,)], ordered=True)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)


def test_int_and_float_with_the_same_value_match() -> None:
    assert results_match([(3,)], [(3.0,)], ordered=False)


def test_column_order_matters() -> None:
    assert not results_match([(1, "a")], [("a", 1)], ordered=False)


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = tmp_path / "t.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE t (a INTEGER, b TEXT); INSERT INTO t VALUES (1, 'x'), (2, 'y'), (3, 'y');"
        )
    conn.close()
    return str(path)


def test_score_gold_passthrough_is_correct(db: str) -> None:
    sql = "SELECT b, count(*) FROM t GROUP BY b"

    result = score_prediction(db, sql, sql, timeout_s=5, max_rows=100)

    assert result["correct"] and result["pred_parses"] and not result["ordered"]


def test_score_equivalent_query_is_correct(db: str) -> None:
    result = score_prediction(db, "SELECT a FROM t WHERE b = 'y'", "SELECT a FROM t WHERE a >= 2", 5, 100)

    assert result["correct"]


def test_score_wrong_order_is_wrong_when_gold_is_ordered(db: str) -> None:
    result = score_prediction(db, "SELECT a FROM t ORDER BY a DESC", "SELECT a FROM t ORDER BY a", 5, 100)

    assert result["ordered"] and not result["correct"]


def test_score_failing_or_missing_prediction_is_wrong(db: str) -> None:
    failing = score_prediction(db, "SELECT a FROM t", "SELECT nope FROM t", 5, 100)
    missing = score_prediction(db, "SELECT a FROM t", None, 5, 100)

    assert not failing["correct"] and "no such column" in failing["pred_exec"]["error"]
    assert not missing["correct"] and missing["pred_exec"] is None and not missing["pred_parses"]


def test_score_broken_gold_is_wrong_even_for_identical_sql(db: str) -> None:
    result = score_prediction(db, "SELECT nope FROM t", "SELECT nope FROM t", 5, 100)

    assert not result["correct"] and not result["gold_valid"] and not result["gold_exec"]["ok"]


def test_score_extra_rows_are_wrong_and_not_fetched(db: str) -> None:
    result = score_prediction(db, "SELECT a FROM t WHERE a = 1", "SELECT a FROM t ORDER BY a", 5, 100)

    # the first row matches the gold's only row; the second is where the fetch stops
    assert not result["correct"]
    assert result["pred_exec"]["truncated"] and result["pred_exec"]["n_rows"] == 1


def test_score_gold_over_the_row_cap_cannot_be_scored(db: str) -> None:
    result = score_prediction(db, "SELECT a FROM t", "SELECT a FROM t", 5, max_rows=2)

    assert not result["correct"] and not result["gold_valid"] and result["gold_exec"]["truncated"]
