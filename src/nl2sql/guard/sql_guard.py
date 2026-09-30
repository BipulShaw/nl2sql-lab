"""SQL guard (PLAN §6.5): allow one read-only query whose tables and columns all exist, or say why not.

Rules, in order: the SQL parses as exactly one statement; that statement is a query (SELECT, or a set
operation such as UNION, with or without WITH); nothing in it writes (INSERT, PRAGMA, SELECT ... INTO, ...);
every table and column it names exists. A block carries a machine-readable reason, such as
`unknown_column:singer.nme`, and a message in SQLite's own words for the repair prompt.

Names are checked against the whole database, not only the tables the prompt showed: a query that would run
is never blocked because the linker left a table out (ADR-016). Name resolution follows SQLite: identifiers
are case-insensitive, a query may refer to its own result aliases anywhere, a subquery may refer to the
tables of the queries around it, and a double-quoted name that matches no column is read as a string."""

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from nl2sql.schema.model import Schema

WRITES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Into,
)
ROWID = frozenset({"rowid", "oid", "_rowid_"})  # SQLite's implicit column on every ordinary table


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str | None = None  # machine-readable, e.g. "unknown_table:singers"; None when ok
    message: str | None = None  # for the model, in SQLite's wording, e.g. "no such table: singers"


@dataclass
class References:
    tables: set[str] = field(default_factory=set)  # real tables, lower-cased
    columns: set[tuple[str, str]] = field(default_factory=set)  # lower-cased (table, column) pairs
    unknown: list[GuardResult] = field(default_factory=list)  # one block per name that does not exist


def block(reason: str, message: str) -> GuardResult:
    return GuardResult(False, reason, message)


def check_sql(sql: str, schema: Schema, dialect: str = "sqlite") -> GuardResult:
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except Exception as exc:  # ParseError, TokenError, and the odd internal error on unusual input
        return block("parse_error", f"syntax error: {str(exc).splitlines()[0]}")
    if len(statements) != 1:
        return block("statement_count", f"expected one SQL statement, got {len(statements)}")
    tree = statements[0]
    if isinstance(tree, WRITES):
        return block(f"write:{type(tree).__name__.lower()}", "only a read-only SELECT query is allowed")
    if not isinstance(tree, exp.Select | exp.SetOperation):  # incl. Command: EXPLAIN, REPLACE, VACUUM, ...
        return block(f"not_a_query:{type(tree).__name__.lower()}", "only a single SELECT query is allowed")
    if (node := tree.find(*WRITES)) is not None:
        return block(f"write:{type(node).__name__.lower()}", "only a read-only SELECT query is allowed")
    refs = resolve_references(tree, schema)
    return refs.unknown[0] if refs.unknown else GuardResult(True)


def resolve_references(tree: exp.Expression, schema: Schema) -> References:
    """The real tables and columns a parsed query uses, and a block for each name that does not exist."""
    columns_of = {t.name.lower(): {c.name.lower() for c in t.columns} for t in schema.tables}
    refs = References()
    home: dict[int, tuple[int, exp.Column, Scope]] = {}  # id(column) -> (depth, column, innermost scope)
    for scope in traverse_scope(tree):
        for source in scope.sources.values():
            if not isinstance(source, exp.Table):
                continue  # a CTE or a subquery in FROM, checked in its own scope
            if source.name.lower() in columns_of:
                refs.tables.add(source.name.lower())
            else:
                refs.unknown.append(block(f"unknown_table:{source.name}", f"no such table: {source.name}"))
        # sqlglot also lists a subquery's unqualified columns under the query around it (they might be
        # correlated); each column is resolved once, from the innermost query that contains it
        depth = scope_depth(scope)
        for column in scope.columns:
            if id(column) not in home or depth > home[id(column)][0]:
                home[id(column)] = (depth, column, scope)
    for _, column, scope in home.values():
        resolve_column(column, scope, columns_of, refs)
    return refs


def scope_depth(scope: Scope) -> int:
    depth = 0
    while scope.parent is not None:
        scope, depth = scope.parent, depth + 1
    return depth


def resolve_column(
    column: exp.Column, scope: Scope, columns_of: dict[str, set[str]], refs: References
) -> None:
    name, qualifier = column.name.lower(), column.table.lower()
    if qualifier:
        source = find_source(scope, qualifier)
        if isinstance(source, exp.Table):
            table = source.name.lower()
            if table not in columns_of:
                return  # already reported as an unknown table
            if name in columns_of[table] or name in ROWID:
                refs.columns.add((table, name))
                return
        elif isinstance(source, Scope) and provides(source, name):
            return
        shown = f"{column.table}.{column.name}"
        refs.unknown.append(block(f"unknown_column:{shown}", f"no such column: {shown}"))
        return

    if name in ROWID:
        return
    current: Scope | None = scope
    while current is not None:  # this query first, then the queries around it (correlated subqueries)
        tables = [
            s.name.lower()
            for s in current.sources.values()
            if isinstance(s, exp.Table) and name in columns_of.get(s.name.lower(), ())
        ]
        refs.columns.update((table, name) for table in tables)
        derived = any(isinstance(s, Scope) and provides(s, name) for s in current.sources.values())
        if tables or derived or name in result_aliases(current):
            return
        current = current.parent
    identifier = column.this
    if isinstance(identifier, exp.Identifier) and identifier.quoted:
        return  # SQLite reads a double-quoted name that matches no column as a string literal
    refs.unknown.append(block(f"unknown_column:{column.name}", f"no such column: {column.name}"))


def find_source(scope: Scope, alias: str) -> exp.Table | Scope | None:
    current: Scope | None = scope
    while current is not None:
        for key, source in current.sources.items():
            if key.lower() == alias:
                return source
        current = current.parent
    return None


def result_aliases(scope: Scope) -> set[str]:
    """Names a query may use for its own result columns: `AS` aliases, and for a UNION (whose ORDER BY
    names result columns) every output name. A bare column is not its own alias, or it would always
    resolve."""
    query = scope.expression
    if isinstance(query, exp.SetOperation):
        return {s.lower() for s in query.named_selects}
    if isinstance(query, exp.Select):
        return {e.alias.lower() for e in query.selects if isinstance(e, exp.Alias)}
    return set()


def provides(scope: Scope, name: str) -> bool:
    """Whether a CTE or subquery in FROM has an output column `name`; a `*` in it could provide anything."""
    query = scope.expression
    if not isinstance(query, exp.Query):
        return True
    return name in {s.lower() for s in query.named_selects} or any(e.is_star for e in query.selects)
