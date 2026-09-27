import sqlite3
from pathlib import Path

import pytest

from nl2sql.execute import execute_sqlite


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "t.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript("CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1), (2), (3);")
    conn.close()
    return path


def test_returns_rows_and_columns(db: Path) -> None:
    result = execute_sqlite(db, "SELECT a, a * 2 AS twice FROM t ORDER BY a")

    assert result.ok
    assert result.columns == ["a", "twice"]
    assert result.rows == [(1, 2), (2, 4), (3, 6)]


def test_database_is_read_only(db: Path) -> None:
    result = execute_sqlite(db, "INSERT INTO t VALUES (4)")

    assert not result.ok and "readonly" in result.error
    assert execute_sqlite(db, "SELECT count(*) FROM t").rows == [(3,)]


def test_timeout_interrupts_a_runaway_query(db: Path) -> None:
    endless = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n"

    result = execute_sqlite(db, endless, timeout_s=0.3)

    assert result.timed_out and not result.ok
    assert result.elapsed_ms < 5000


def test_row_cap_truncates_and_says_so(db: Path) -> None:
    result = execute_sqlite(db, "SELECT a FROM t", max_rows=2)

    assert result.truncated and len(result.rows) == 2


def test_errors_are_results_not_exceptions(db: Path) -> None:
    assert "syntax error" in execute_sqlite(db, "SELEC a FROM t").error
    assert "one statement" in execute_sqlite(db, "SELECT 1; SELECT 2").error


def test_invalid_utf8_text_is_replaced(db: Path) -> None:
    result = execute_sqlite(db, "SELECT CAST(x'ff41' AS TEXT)")

    assert result.rows == [("�A",)]
