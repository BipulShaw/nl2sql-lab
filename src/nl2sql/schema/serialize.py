"""Render a `Schema` as DDL-style prompt text, shrinking it to a token budget in a fixed order (PLAN §6.1)."""

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass

from nl2sql.schema.model import Column, ForeignKey, Schema, Table

SIMPLE_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Words a model would have to quote if used bare as a table or column name; `order` and `group` do occur.
# One string reads better than a 90-line list literal, hence the noqa.
RESERVED = frozenset(
    "add all alter and as asc between by case check collate column commit constraint create "  # noqa: SIM905
    "cross current_date current_time current_timestamp default delete desc distinct drop else end "
    "escape except exists foreign from full group having in index inner insert intersect into is "
    "join left like limit match natural not null of offset on or order outer primary references "
    "right rollback select set table then to transaction union unique update using values when "
    "where with".split()
)

# Each level sheds more than the one before; the first that fits the budget wins. The last two only touch
# tables the linker did not select, so without a linker (every table kept) the list stops at no_descriptions.
LEVELS = ("full", "no_samples", "no_descriptions", "keys_only_unselected", "selected_only")


@dataclass(frozen=True)
class SerializedSchema:
    text: str
    tokens: int
    level: str  # the LEVELS entry that fit, or "over_budget" if even the smallest one did not


def quote(name: str) -> str:
    if SIMPLE_IDENT.fullmatch(name) and name.lower() not in RESERVED:
        return name
    return "`" + name.replace("`", "``") + "`"


def format_value(value: str | int | float) -> str:
    """As a SQL literal, so the model sees exactly how to write the value (quotes, case, date format)."""
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def column_comment(column: Column, samples: bool, descriptions: bool) -> str:
    parts = []
    if descriptions and column.description:
        parts.append(column.description)
    if samples and column.sample_values:
        parts.append("e.g. " + ", ".join(format_value(v) for v in column.sample_values))
    return " | ".join(parts)


def render_table(
    table: Table, foreign_keys: list[ForeignKey], samples: bool, descriptions: bool, keys_only: bool
) -> str:
    key_columns = {c.name for c in table.primary_key}
    key_columns |= {fk.from_column for fk in foreign_keys if fk.from_table == table.name}
    key_columns |= {fk.to_column for fk in foreign_keys if fk.to_table == table.name}
    composite_pk = len(table.primary_key) > 1

    body: list[tuple[str, str]] = []  # (definition, comment)
    for column in table.columns:
        if keys_only and column.name not in key_columns:
            continue
        definition = f"{quote(column.name)} {column.type}".rstrip()
        if column.is_pk and not composite_pk:
            definition += " PRIMARY KEY"
        body.append((definition, column_comment(column, samples, descriptions)))
    if composite_pk:
        body.append((f"PRIMARY KEY ({', '.join(quote(c.name) for c in table.primary_key)})", ""))

    lines = [f"  {definition}{',' if i < len(body) - 1 else ''}" for i, (definition, _) in enumerate(body)]
    width = max((len(line) for line, (_, comment) in zip(lines, body, strict=True) if comment), default=0)
    lines = [
        f"{line:<{width}}  -- {comment}" if comment else line
        for line, (_, comment) in zip(lines, body, strict=True)
    ]
    fk_lines = [
        f"-- FK: {quote(fk.from_table)}.{quote(fk.from_column)} -> {quote(fk.to_table)}.{quote(fk.to_column)}"
        for fk in foreign_keys
        if fk.from_table == table.name
    ]
    return "\n".join([f"CREATE TABLE {quote(table.name)} (", *lines, ");", *fk_lines])


def render(schema: Schema, selected: set[str], level: str) -> str:
    rank = LEVELS.index(level)
    shown = [t for t in schema.tables if rank < LEVELS.index("selected_only") or t.name.lower() in selected]
    names = {t.name for t in shown}
    foreign_keys = [fk for fk in schema.foreign_keys if fk.from_table in names and fk.to_table in names]
    return "\n\n".join(
        render_table(
            table,
            foreign_keys,
            samples=rank < LEVELS.index("no_samples"),
            descriptions=rank < LEVELS.index("no_descriptions"),
            keys_only=rank >= LEVELS.index("keys_only_unselected") and table.name.lower() not in selected,
        )
        for table in shown
    )


def serialize_schema(
    schema: Schema,
    count_tokens: Callable[[str], int],
    token_budget: int = 1536,
    tables: Collection[str] | None = None,
    include_samples: bool = True,
) -> SerializedSchema:
    """`tables` are the linker's picks and are never dropped. With `tables=None` (no linker) every table is
    kept, so only sample values and descriptions can be shed and the result may stay over budget."""
    if tables is None:
        selected, levels = {t.name.lower() for t in schema.tables}, LEVELS[:3]
    else:
        selected, levels = {t.lower() for t in tables}, LEVELS
    if not include_samples:
        levels = levels[1:]
    for level in levels:
        text = render(schema, selected, level)
        tokens = count_tokens(text)
        if tokens <= token_budget:
            return SerializedSchema(text, tokens, level)
    return SerializedSchema(text, tokens, "over_budget")
