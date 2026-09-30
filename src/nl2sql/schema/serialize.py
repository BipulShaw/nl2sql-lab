"""Render a `Schema` as DDL-style prompt text, shrinking it to a token budget in a fixed order (PLAN §6.1,
ADR-015, ADR-017)."""

import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace

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


@dataclass(frozen=True)
class Level:
    name: str
    samples: bool
    descriptions: bool
    unselected: (
        str  # how tables the linker did not select appear: "full", "keys" (key columns only), "hidden"
    )


# Each level sheds more than the one before; the first that fits the budget wins. Without a linker every table
# is kept, so only detail can go.
FULL_SCHEMA_LEVELS = (
    Level("full", True, True, "full"),
    Level("no_samples", False, True, "full"),
    Level("no_descriptions", False, False, "full"),
)
# With a linker, unselected tables shrink and go first, then the selected ones lose detail (ADR-017).
LINKED_LEVELS = (
    Level("full", True, True, "full"),
    Level("keys_only_unselected", True, True, "keys"),
    Level("selected_only", True, True, "hidden"),
    Level("selected_no_samples", False, True, "hidden"),
    Level("selected_no_descriptions", False, False, "hidden"),
)
PRUNED = "pruned_columns"  # past the last linked level: the selected tables' lowest-scored columns go


@dataclass(frozen=True)
class SerializedSchema:
    text: str
    tokens: int
    level: str  # the Level that fit, PRUNED, or "over_budget" if nothing did


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


def key_columns(table: Table, foreign_keys: list[ForeignKey]) -> set[str]:
    keys = {c.name for c in table.primary_key}
    keys |= {fk.from_column for fk in foreign_keys if fk.from_table == table.name}
    return keys | {fk.to_column for fk in foreign_keys if fk.to_table == table.name}


def render_table(
    table: Table,
    foreign_keys: list[ForeignKey],
    samples: bool,
    descriptions: bool,
    columns: set[str] | None = None,
) -> str:
    """`columns`: the only columns to show, by name; None shows them all."""
    composite_pk = len(table.primary_key) > 1

    body: list[tuple[str, str]] = []  # (definition, comment)
    for column in table.columns:
        if columns is not None and column.name not in columns:
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


def render(schema: Schema, selected: set[str], level: Level, kept: set[tuple[str, str]] | None = None) -> str:
    """`selected`: lower-cased table names. `kept`: if given, the only lower-cased (table, column) pairs that
    selected tables show besides their keys."""
    shown = [t for t in schema.tables if level.unselected != "hidden" or t.name.lower() in selected]
    names = {t.name for t in shown}
    foreign_keys = [fk for fk in schema.foreign_keys if fk.from_table in names and fk.to_table in names]
    parts = []
    for table in shown:
        columns = None
        if table.name.lower() not in selected:
            columns = key_columns(table, foreign_keys) if level.unselected == "keys" else None
        elif kept is not None:
            columns = key_columns(table, foreign_keys)
            columns |= {c.name for c in table.columns if (table.name.lower(), c.name.lower()) in kept}
        parts.append(render_table(table, foreign_keys, level.samples, level.descriptions, columns))
    return "\n\n".join(parts)


def prune_columns(
    schema: Schema,
    selected: set[str],
    level: Level,
    count_tokens: Callable[[str], int],
    token_budget: int,
    column_scores: Mapping[tuple[str, str], float],
) -> SerializedSchema:
    """The selected tables with as many of their best-scored columns as fit (PLAN §6.2); keys always stay."""
    scores = {(t.lower(), c.lower()): s for (t, c), s in column_scores.items()}
    ranked = [
        (t.name.lower(), c.name.lower())
        for t in schema.tables
        if t.name.lower() in selected
        for c in t.columns
    ]
    ranked.sort(key=lambda column: -scores.get(column, -1.0))  # stable: ties keep schema order

    def keep(n: int) -> SerializedSchema:
        text = render(schema, selected, level, kept=set(ranked[:n]))
        return SerializedSchema(text, count_tokens(text), PRUNED)

    best = keep(0)
    if best.tokens > token_budget:
        return replace(best, level="over_budget")
    fits, too_many = 0, len(ranked)  # keeping every column is the level that already failed
    while too_many - fits > 1:  # binary search: tokens only grow with each column kept
        middle = (fits + too_many) // 2
        candidate = keep(middle)
        if candidate.tokens <= token_budget:
            fits, best = middle, candidate
        else:
            too_many = middle
    return best


def serialize_schema(
    schema: Schema,
    count_tokens: Callable[[str], int],
    token_budget: int = 1536,
    tables: Collection[str] | None = None,
    include_samples: bool = True,
    column_scores: Mapping[tuple[str, str], float] | None = None,
) -> SerializedSchema:
    """`tables` are the linker's picks and are never dropped; `column_scores` (linker similarity per
    (table, column)) lets their columns be pruned as a last resort. With `tables=None` (no linker) every table
    is kept, so only sample values and descriptions can be shed and the result may stay over budget."""
    if tables is None:
        selected, levels = {t.name.lower() for t in schema.tables}, FULL_SCHEMA_LEVELS
    else:
        selected, levels = {t.lower() for t in tables}, LINKED_LEVELS
    if not include_samples:
        without: dict[tuple[bool, str], Level] = {}
        for level in levels:  # levels that now show the same thing merge; the later name says what went
            without[(level.descriptions, level.unselected)] = replace(level, samples=False)
        levels = tuple(without.values())
    for level in levels:
        text = render(schema, selected, level)
        tokens = count_tokens(text)
        if tokens <= token_budget:
            return SerializedSchema(text, tokens, level.name)
    if tables is not None and column_scores is not None:
        return prune_columns(schema, selected, level, count_tokens, token_budget, column_scores)
    return SerializedSchema(text, tokens, "over_budget")
