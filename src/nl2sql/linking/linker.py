"""Schema linking (PLAN §6.2): score every table and column of a database against a question, then pick
tables.

Each table gets one doc (its name and column names) and each column one doc (table and column name, BIRD
description, sample values). A table scores max(sim(table doc), best sim over its column docs), plus a small
bonus when its name appears word for word in the question. Selection keeps every table of a small database,
else the top k plus their foreign-key neighbors."""

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from nl2sql.linking.embed import QUERY_INSTRUCTION, Embedder
from nl2sql.schema.model import Column, Schema, Table

CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
WORD = re.compile(r"[a-z0-9]+")


def words(name: str) -> str:
    """A name as words: `setCode` -> `set code`, `singer_in_concert` -> `singer in concert`."""
    return " ".join(WORD.findall(CAMEL.sub(" ", name).lower()))


def stems(text: str) -> set[str]:
    """Lower-cased words with a plural `s` dropped, so `singers` in a question matches the table `singer`."""
    return {
        w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        for w in WORD.findall(text.lower())
    }


def table_doc(table: Table) -> str:
    return f"{words(table.name)}: " + ", ".join(words(c.name) for c in table.columns)


def column_doc(table: Table, column: Column) -> str:
    doc = f"{words(table.name)} {words(column.name)}"
    if column.description:
        doc += f": {column.description}"
    if column.sample_values:
        doc += " e.g. " + ", ".join(str(v) for v in column.sample_values)
    return doc


@dataclass(frozen=True)
class Scores:
    tables: dict[str, float]  # table name -> score, every table of the database
    columns: dict[tuple[str, str], float]  # (table, column) -> similarity, every column


@dataclass(frozen=True)
class DocVectors:
    tables: np.ndarray  # one row per table, in schema order
    columns: np.ndarray  # one row per column, tables in schema order, columns in table order


class SchemaLinker:
    def __init__(
        self,
        embedder: Embedder,
        top_k: int = 4,
        all_tables_up_to: int = 6,
        fk_hops: int = 1,
        lexical_bonus: float = 0.1,
        query_instruction: bool = True,
    ) -> None:
        self.embedder = embedder
        self.top_k, self.all_tables_up_to, self.fk_hops = top_k, all_tables_up_to, fk_hops
        self.lexical_bonus, self.query_instruction = lexical_bonus, query_instruction
        self.vectors: dict[str, DocVectors] = {}  # by a hash of the docs, so a changed database re-embeds

    def doc_vectors(self, schema: Schema) -> DocVectors:
        table_docs = [table_doc(t) for t in schema.tables]
        column_docs = [column_doc(t, c) for t in schema.tables for c in t.columns]
        key = hashlib.sha256("\0".join(table_docs + ["\0"] + column_docs).encode()).hexdigest()
        if key not in self.vectors:
            self.vectors[key] = DocVectors(
                self.embedder.encode(table_docs), self.embedder.encode(column_docs)
            )
        return self.vectors[key]

    def score(self, schema: Schema, question: str, evidence: str = "") -> Scores:
        query = f"{question} {evidence}".strip()
        prefix = QUERY_INSTRUCTION if self.query_instruction else ""
        q = self.embedder.encode([prefix + query])[0]
        vectors = self.doc_vectors(schema)
        table_sims, column_sims = vectors.tables @ q, vectors.columns @ q
        asked = stems(query)

        tables: dict[str, float] = {}
        columns: dict[tuple[str, str], float] = {}
        i = 0
        for t, table in enumerate(schema.tables):
            sims = column_sims[i : i + len(table.columns)]
            i += len(table.columns)
            columns |= {(table.name, c.name): float(s) for c, s in zip(table.columns, sims, strict=True)}
            best = max(float(table_sims[t]), float(sims.max(initial=-1.0)))
            named = stems(words(table.name)) <= asked
            tables[table.name] = best + (self.lexical_bonus if named else 0.0)
        return Scores(tables, columns)

    def select(self, schema: Schema, scores: Scores) -> list[str]:
        """Picked table names, in schema order."""
        names = [t.name for t in schema.tables]
        if len(names) <= self.all_tables_up_to:
            return names
        picked = set(sorted(names, key=lambda n: -scores.tables[n])[: self.top_k])
        for _ in range(self.fk_hops):
            picked |= {fk.to_table for fk in schema.foreign_keys if fk.from_table in picked} | {
                fk.from_table for fk in schema.foreign_keys if fk.to_table in picked
            }
        return [n for n in names if n in picked]

    def link(self, schema: Schema, question: str, evidence: str = "") -> tuple[list[str], Scores]:
        scores = self.score(schema, question, evidence)
        return self.select(schema, scores), scores
