"""Read a SQLite database's schema, keys and a few sample values into a `Schema`, with an on-disk cache."""

import csv
import hashlib
import io
import re
import sqlite3
from pathlib import Path

from nl2sql.data.common import DATA_DIR
from nl2sql.execute.sqlite_exec import connect_readonly
from nl2sql.schema.model import Column, ForeignKey, Schema, Table

CACHE_VERSION = 1  # bump when introspection output changes, to invalidate cached schemas
SCAN_ROWS = 1000  # sample values come from the first non-null rows only, so huge tables stay cheap
MAX_DESCRIPTION_CHARS = 200  # BIRD value descriptions run to 800+ chars; p95 is ~170
BOILERPLATE = re.compile(r"common\s*sense\s*(evidence|reasoning)\s*:\s*", re.IGNORECASE)
TABLES_QUERY = (
    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
)


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sample_values(
    conn: sqlite3.Connection, table: str, column: str, n: int, max_chars: int
) -> list[str | int | float]:
    """Up to n distinct non-null values: numbers as-is, strings whitespace-collapsed and cut to max_chars."""
    col, tab = quote_ident(column), quote_ident(table)
    query = (
        f"SELECT DISTINCT {col} FROM (SELECT {col} FROM {tab} WHERE {col} IS NOT NULL LIMIT {SCAN_ROWS}) "
        f"LIMIT {n * 3}"  # over-fetch: blobs and blank strings are skipped below
    )
    values: list[str | int | float] = []
    for (value,) in conn.execute(query):
        if isinstance(value, str):
            value = " ".join(value.split())[:max_chars]  # one line, so it can sit in a -- comment
            if not value or value in values:
                continue
        elif not isinstance(value, int | float):
            continue  # blobs
        values.append(value)
        if len(values) == n:
            break
    return values


def introspect_sqlite(db_path: Path, db_id: str, n_samples: int = 3, max_chars: int = 30) -> Schema:
    conn = connect_readonly(db_path)
    try:
        names = [row[0] for row in conn.execute(TABLES_QUERY)]
        tables, raw_fks = [], []
        for name in names:
            info = conn.execute(f"PRAGMA table_info({quote_ident(name)})").fetchall()
            columns = [
                Column(
                    name=col_name,
                    type=(col_type or "").upper(),
                    is_pk=pk > 0,
                    pk_position=pk,
                    sample_values=sample_values(conn, name, col_name, n_samples, max_chars),
                )
                for _cid, col_name, col_type, _notnull, _default, pk in info
            ]
            tables.append(Table(name=name, columns=columns))
            # foreign_key_list rows: id, seq, table, from, to, on_update, on_delete, match
            fk_rows = conn.execute(f"PRAGMA foreign_key_list({quote_ident(name)})").fetchall()
            raw_fks += [(name, row[3], row[2], row[4]) for row in fk_rows]
    finally:
        conn.close()
    description_dir = db_path.parent / "database_description"  # BIRD ships one CSV per table here
    if description_dir.is_dir():
        add_descriptions(tables, read_descriptions(description_dir))
    return Schema(db_id=db_id, tables=tables, foreign_keys=resolve_foreign_keys(tables, raw_fks))


def resolve_foreign_keys(
    tables: list[Table], raw: list[tuple[str, str, str, str | None]]
) -> list[ForeignKey]:
    """Match declared FKs to real tables and columns (case-insensitively, as SQLite does), fill an omitted
    target column with the target's primary key, and drop duplicates and FKs to things that do not exist."""
    by_name = {t.name.lower(): t for t in tables}

    def find_column(table: Table, name: str | None) -> str | None:
        if name is None:  # `REFERENCES t` with no column means t's primary key
            return table.primary_key[0].name if len(table.primary_key) == 1 else None
        return next((c.name for c in table.columns if c.name.lower() == name.lower()), None)

    out: list[ForeignKey] = []
    for from_table, from_column, to_table, to_column in raw:
        source, target = by_name[from_table.lower()], by_name.get(to_table.lower())
        if target is None:
            continue
        from_name, to_name = find_column(source, from_column), find_column(target, to_column)
        if from_name is None or to_name is None:
            continue
        fk = ForeignKey(
            from_table=source.name, from_column=from_name, to_table=target.name, to_column=to_name
        )
        if fk not in out:
            out.append(fk)
    return out


def describe(column: str, row: dict[str, str]) -> str | None:
    """One line from BIRD's expanded name, description and value notes, minus parts repeating the name."""
    parts: list[str] = []
    for key in ("column_name", "column_description", "value_description"):
        text = BOILERPLATE.sub("", " ".join((row.get(key) or "").split())).strip()
        if text and text.lower() != column.lower() and text.lower() not in (p.lower() for p in parts):
            parts.append(text)
    text = "; ".join(parts)
    if len(text) > MAX_DESCRIPTION_CHARS:
        text = text[: MAX_DESCRIPTION_CHARS - 1].rstrip() + "…"
    return text or None


def read_descriptions(description_dir: Path) -> dict[tuple[str, str], str]:
    """BIRD `database_description/<table>.csv` files, keyed by lower-cased (table, column)."""
    out: dict[tuple[str, str], str] = {}
    for csv_file in sorted(description_dir.glob("*.csv")):
        raw = csv_file.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")  # a few BIRD files are Windows-1252
        reader = csv.DictReader(io.StringIO(text))
        reader.fieldnames = [(name or "").strip() for name in reader.fieldnames or []]
        for row in reader:
            column = " ".join((row.get("original_column_name") or "").split())
            description = describe(column, row) if column else None
            if description:
                out[(csv_file.stem.strip().lower(), column.lower())] = description
    return out


def add_descriptions(tables: list[Table], descriptions: dict[tuple[str, str], str]) -> None:
    for table in tables:
        for column in table.columns:
            column.description = descriptions.get((table.name.lower(), " ".join(column.name.split()).lower()))


def load_schema(db_path: Path, db_id: str, cache_dir: Path | None = None) -> Schema:
    """Introspect once per database file; later calls read data/cache/schemas/<db_id>-<key>.json."""
    stat = db_path.stat()
    key = hashlib.sha256(f"{CACHE_VERSION}|{db_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    cache_file = (cache_dir or DATA_DIR / "cache" / "schemas") / f"{db_id}-{key.hexdigest()[:12]}.json"
    if cache_file.exists():
        return Schema.model_validate_json(cache_file.read_text(encoding="utf-8"))
    schema = introspect_sqlite(db_path, db_id)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(schema.model_dump_json(), encoding="utf-8")
    return schema
