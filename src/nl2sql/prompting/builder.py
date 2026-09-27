"""The prompt, identical for training and inference (PLAN §6.3), and SQL extraction from the model's reply."""

import re
from dataclasses import dataclass

SYSTEM = (
    "You are an expert SQL assistant. Given a database schema and a question, write a single SQLite SELECT "
    "query that answers the question. Use only tables and columns from the schema. Output only the SQL in a "
    "```sql code block."
)
USER_TEMPLATE = (
    "### Database: {db_id}\n### Schema\n{schema}\n### Evidence\n{evidence}\n### Question\n{question}"
)

THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
SQL_FENCE = re.compile(r"```(?:sqlite|sql)?[ \t]*\n?(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)
# Fallback when there is no fence: the first line that starts a SELECT/WITH statement, up to `;` or a fence.
BARE_STATEMENT = re.compile(r"^[ \t]*((?:SELECT|WITH)\b(?:(?!```)[^;])*)", re.IGNORECASE | re.MULTILINE)


def build_messages(db_id: str, schema_text: str, question: str, evidence: str = "") -> list[dict[str, str]]:
    user = USER_TEMPLATE.format(
        db_id=db_id, schema=schema_text, evidence=evidence.strip() or "(none)", question=question.strip()
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class Extraction:
    sql: str | None
    source: str  # "fence", "bare" or "none"
    think_leak: bool  # a <think> block appeared although thinking was off


def extract_sql(text: str) -> Extraction:
    """First ```sql block; else the first bare SELECT/WITH statement (PLAN §6.4). Any <think> block is removed
    first, and trailing semicolons are dropped."""
    think_leak = "<think>" in text
    text = THINK_BLOCK.sub("", text)
    if match := SQL_FENCE.search(text):
        sql, source = match.group(1), "fence"
    elif match := BARE_STATEMENT.search(text):
        sql, source = match.group(1), "bare"
    else:
        return Extraction(None, "none", think_leak)
    sql = sql.strip().rstrip(";").strip()
    return Extraction(sql, source, think_leak) if sql else Extraction(None, "none", think_leak)
