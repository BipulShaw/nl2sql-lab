import sqlite3
from pathlib import Path

import pytest
import sqlglot

from nl2sql.execute import execute_sqlite
from nl2sql.guard.sql_guard import check_sql, resolve_references
from nl2sql.schema.introspect import introspect_sqlite
from nl2sql.schema.model import Schema

DDL = """
CREATE TABLE stadium (stadium_id INTEGER PRIMARY KEY, name TEXT, capacity INT);
CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, country TEXT, age INT, "Home Town" TEXT,
                     "order" INT);
CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, stadium_id INT REFERENCES stadium, year INT);
INSERT INTO stadium VALUES (1, 'Hampden', 52000), (2, 'Ibrox', 50000);
INSERT INTO singer VALUES (1, 'Joe', 'France', 52, 'Paris', 1), (2, 'Ann', 'Spain', 31, NULL, 2);
INSERT INTO concert VALUES (1, 1, 2014), (2, 1, 2015);
"""
JOIN = "FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id"

# (sql, expected block reason or None). Every allowed query runs in SQLite, and every unknown-name block
# fails in SQLite with the guard's own message: test_guard_agrees_with_sqlite checks both.
CASES = [
    # allowed
    ("SELECT name FROM singer", None),
    ("select NAME from SINGER", None),  # identifiers are case-insensitive
    ("SELECT name FROM singer;", None),  # a trailing semicolon is still one statement
    (f"SELECT T2.name, T1.year {JOIN}", None),
    ("SELECT country, count(*) AS n FROM singer GROUP BY country ORDER BY n DESC", None),  # result alias
    ("SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer)", None),
    # `year` exists only in the subquery's table: resolved there, not against the outer query's stadium
    ("SELECT name FROM stadium WHERE stadium_id IN (SELECT stadium_id FROM concert WHERE year = 2014)", None),
    # correlated: a subquery reads the columns of the query around it, qualified or not
    (
        "SELECT name FROM singer AS s WHERE age > (SELECT avg(age) FROM singer WHERE country = s.country)",
        None,
    ),
    (
        "SELECT name FROM stadium WHERE stadium_id IN (SELECT stadium_id FROM concert WHERE capacity > 0)",
        None,
    ),
    (
        "SELECT name FROM stadium WHERE EXISTS (SELECT 1 FROM concert WHERE stadium_id = stadium.stadium_id)",
        None,
    ),
    ("WITH big AS (SELECT stadium_id, capacity AS cap FROM stadium) SELECT big.cap FROM big", None),
    ("SELECT t.n FROM (SELECT count(*) AS n FROM singer) AS t", None),
    ("SELECT t.age FROM (SELECT * FROM singer) AS t", None),  # a star provides every column
    ("SELECT name FROM singer UNION SELECT name FROM stadium ORDER BY name", None),
    ("SELECT stadium_id FROM stadium EXCEPT SELECT stadium_id FROM concert", None),
    ('SELECT "Home Town", `order`, [order] FROM singer', None),  # three ways to quote a name
    ("SELECT rowid, name FROM singer", None),  # SQLite's implicit column
    ('SELECT name FROM singer WHERE country = "France"', None),  # double quotes, no such column: a string
    ("SELECT (SELECT max(capacity) FROM stadium) - (SELECT min(capacity) FROM stadium)", None),
    ("SELECT T1.* FROM singer AS T1", None),
    ("SELECT CAST(SUM(CASE WHEN age > 40 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM singer", None),
    # names that do not exist
    ("SELECT nme FROM singer", "unknown_column:nme"),
    ("SELECT T1.nme FROM singer AS T1", "unknown_column:T1.nme"),
    ("SELECT name FROM singers", "unknown_table:singers"),
    (
        "SELECT name FROM stadium WHERE stadium_id IN (SELECT stadium_id FROM concert WHERE yr = 2014)",
        "unknown_column:yr",
    ),
    (f"SELECT T2.name {JOIN} WHERE T1.name = 'x'", "unknown_column:T1.name"),  # exists, but not on that table
    ("SELECT x.name FROM singer", "unknown_column:x.name"),
    ("SELECT t.age FROM (SELECT name FROM singer) AS t", "unknown_column:t.age"),
    ("SELECT name FROM singer GROUP BY country ORDER BY cnt", "unknown_column:cnt"),
    # not one read-only query
    ("DELETE FROM singer", "write:delete"),
    ("UPDATE singer SET age = 1", "write:update"),
    ("INSERT INTO singer (name) SELECT name FROM stadium", "write:insert"),
    ("DROP TABLE singer", "write:drop"),
    ("CREATE TABLE t AS SELECT * FROM singer", "write:create"),
    ("PRAGMA table_info(singer)", "write:pragma"),
    ("ATTACH DATABASE 'x.db' AS y", "write:attach"),
    ("SELECT * INTO backup FROM singer", "write:into"),
    ("WITH d AS (DELETE FROM singer RETURNING *) SELECT * FROM d", "write:delete"),
    ("REPLACE INTO singer (name) VALUES ('a')", "not_a_query:command"),
    ("EXPLAIN SELECT 1", "not_a_query:command"),
    ("VALUES (1)", "not_a_query:values"),
    ("SELECT name FROM singer; DROP TABLE singer", "statement_count"),
    ("", "statement_count"),
    ("SELEC name FROM singer", "parse_error"),
    ("SELECT name FROM singer WHERE", "parse_error"),
]


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("guard") / "concert.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(DDL)
    conn.close()
    return path


@pytest.fixture(scope="module")
def schema(db: Path) -> Schema:
    return introspect_sqlite(db, "concert")


@pytest.mark.parametrize(("sql", "reason"), CASES)
def test_guard(sql: str, reason: str | None, schema: Schema) -> None:
    result = check_sql(sql, schema)

    assert (result.ok, result.reason) == (reason is None, reason)
    assert result.ok or result.message


@pytest.mark.parametrize(("sql", "reason"), [c for c in CASES if c[1] is None or c[1].startswith("unknown")])
def test_guard_agrees_with_sqlite(sql: str, reason: str | None, db: Path, schema: Schema) -> None:
    executed = execute_sqlite(db, sql)

    if reason is None:
        assert executed.ok, executed.error
    else:
        assert not executed.ok and check_sql(sql, schema).message in executed.error


def test_some_errors_are_left_to_execution(db: Path, schema: Schema) -> None:
    sql = "SELECT name FROM singer AS a JOIN singer AS b ON a.singer_id = b.singer_id"

    assert check_sql(sql, schema).ok  # the guard checks that names exist, not that they are unambiguous
    assert "ambiguous column name: name" in execute_sqlite(db, sql).error


def test_references_list_real_tables_and_columns(schema: Schema) -> None:
    tree = sqlglot.parse_one(f"SELECT T2.name {JOIN} WHERE year > 2000", read="sqlite")

    refs = resolve_references(tree, schema)

    assert refs.tables == {"concert", "stadium"}
    assert refs.columns == {
        ("stadium", "name"),
        ("concert", "stadium_id"),
        ("stadium", "stadium_id"),
        ("concert", "year"),
    }
    assert refs.unknown == []
