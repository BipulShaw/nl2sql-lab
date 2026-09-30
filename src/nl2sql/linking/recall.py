"""Link recall (PLAN §6.2): the share of questions whose selected tables include every table the gold query
reads. A missed table is one the model cannot see in full, so this bounds what linking can cost."""

import logging
from collections.abc import Collection

import sqlglot

from nl2sql.config import LinkingConfig
from nl2sql.data import load_examples
from nl2sql.guard.sql_guard import resolve_references
from nl2sql.linking.linker import SchemaLinker
from nl2sql.schema.introspect import load_schema
from nl2sql.schema.model import Schema


def gold_tables(gold_sql: str, schema: Schema) -> set[str] | None:
    """Lower-cased names of the real tables the gold query reads; None if sqlglot cannot parse it."""
    try:
        return resolve_references(sqlglot.parse_one(gold_sql, read="sqlite"), schema).tables
    except sqlglot.errors.ParseError:
        return None


def link_hit(gold: set[str] | None, linked: Collection[str]) -> bool | None:
    return None if gold is None else gold <= {t.lower() for t in linked}


def link_recall(dataset: str, split: str, config: LinkingConfig, linker: SchemaLinker) -> dict:
    """Recall over the whole split, and over questions on databases big enough for linking to drop tables."""
    logging.getLogger("sqlglot").setLevel(logging.ERROR)
    schemas: dict = {}
    n = hits = big = big_hits = unparsed = kept = size = 0
    misses = []
    for example in load_examples(dataset, split):
        if example.db_path not in schemas:
            schemas[example.db_path] = load_schema(example.db_path, example.db_id)
        schema = schemas[example.db_path]
        tables, _ = linker.link(schema, example.question, example.evidence)
        hit = link_hit(gold_tables(example.gold_sql, schema), tables)
        if hit is None:
            unparsed += 1
            continue
        n += 1
        hits += hit
        if len(schema.tables) > config.all_tables_up_to:
            big, big_hits, kept, size = big + 1, big_hits + hit, kept + len(tables), size + len(schema.tables)
            if not hit:
                misses.append({"id": example.id, "db_id": example.db_id, "linked": tables})
    return {
        "dataset": dataset,
        "split": split,
        "n": n,
        "recall": round(hits / n, 4) if n else None,
        "n_big_db": big,
        "recall_big_db": round(big_hits / big, 4) if big else None,
        "mean_tables_kept_big_db": round(kept / big, 2) if big else None,
        "mean_tables_big_db": round(size / big, 2) if big else None,
        "gold_unparsed": unparsed,
        "misses": misses,
    }
