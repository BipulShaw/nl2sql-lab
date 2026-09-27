import pytest

from nl2sql.prompting.builder import SYSTEM, build_messages, extract_sql


def test_messages_follow_the_fixed_template() -> None:
    messages = build_messages("concert_singer", "CREATE TABLE singer (...);", "  How many singers? ")

    assert messages[0] == {"role": "system", "content": SYSTEM}
    assert messages[1]["content"] == (
        "### Database: concert_singer\n### Schema\nCREATE TABLE singer (...);\n"
        "### Evidence\n(none)\n### Question\nHow many singers?"
    )


def test_evidence_is_included_when_present() -> None:
    content = build_messages("db", "schema", "q", evidence="rate = a / b")[1]["content"]

    assert "### Evidence\nrate = a / b\n" in content


@pytest.mark.parametrize(
    ("reply", "sql", "source"),
    [
        ("```sql\nSELECT count(*) FROM singer;\n```", "SELECT count(*) FROM singer", "fence"),
        ("Here you go:\n```SQLite\nSELECT 1\n```\nDone.", "SELECT 1", "fence"),
        ("```\nSELECT 2\n```", "SELECT 2", "fence"),
        ("```sql\nSELECT 3 FROM t WHERE a = 1", "SELECT 3 FROM t WHERE a = 1", "fence"),  # cut off by length
        ("```sql SELECT 4```", "SELECT 4", "fence"),
        ("The query with a join:\nSELECT a FROM t;\nMore text", "SELECT a FROM t", "bare"),
        ("WITH x AS (SELECT 1) SELECT * FROM x", "WITH x AS (SELECT 1) SELECT * FROM x", "bare"),
        ("I cannot answer that.", None, "none"),
        ("```sql\n```", None, "none"),
    ],
)
def test_extract_sql(reply: str, sql: str | None, source: str) -> None:
    result = extract_sql(reply)

    assert (result.sql, result.source) == (sql, source)


def test_think_blocks_are_removed_and_flagged() -> None:
    result = extract_sql("<think>maybe SELECT wrong</think>\n```sql\nSELECT right\n```")

    assert result.sql == "SELECT right" and result.think_leak
