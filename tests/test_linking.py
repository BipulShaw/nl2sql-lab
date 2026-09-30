import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from nl2sql.linking.linker import SchemaLinker, column_doc, stems, table_doc, words
from nl2sql.schema.model import Column, ForeignKey, Schema, Table


class BagOfWords:
    """Stand-in embedder without the ml extra: each word hashes to one of 4096 dimensions."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), 4096))
        for i, text in enumerate(texts):
            for word in stems(text):
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 4096] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def table(name: str, *columns: str) -> Table:
    return Table(
        name=name, columns=[Column(name=c, is_pk=c == "id", pk_position=int(c == "id")) for c in columns]
    )


SMALL = Schema(db_id="small", tables=[table("singer", "id", "name"), table("concert", "id", "year")])
BIG = Schema(
    db_id="big",
    tables=[
        table("singer", "id", "name", "country"),
        table("concert", "id", "stadium_id", "year"),
        table("singer_in_concert", "concert_id", "singer_id"),
        table("stadium", "id", "capacity", "location"),
        table("album", "id", "title", "released"),
        table("genre", "id", "label"),
        table("award", "id", "category", "prize"),
        table("festival", "id", "city", "month"),
    ],
    foreign_keys=[
        ForeignKey(from_table="concert", from_column="stadium_id", to_table="stadium", to_column="id"),
        ForeignKey(
            from_table="singer_in_concert", from_column="concert_id", to_table="concert", to_column="id"
        ),
        ForeignKey(
            from_table="singer_in_concert", from_column="singer_id", to_table="singer", to_column="id"
        ),
    ],
)


def test_names_read_as_words() -> None:
    assert words("setCode") == "set code"
    assert words("singer_in_concert") == "singer in concert"
    assert words("FRPM Count (K-12)") == "frpm count k 12"
    assert stems("How many singers? Class") == {"how", "many", "singer", "class"}


def test_docs() -> None:
    column = Column(name="homeTown", description="where the singer grew up", sample_values=["Paris", 3])

    assert table_doc(table("singer", "id", "homeTown")) == "singer: id, home town"
    assert column_doc(table("singer"), column) == "singer home town: where the singer grew up e.g. Paris, 3"


def test_small_database_keeps_every_table() -> None:
    tables, _ = SchemaLinker(BagOfWords(), top_k=1).link(SMALL, "How many singers are there?")

    assert tables == ["singer", "concert"]


def test_big_database_keeps_top_k_and_their_foreign_key_neighbors() -> None:
    linker = SchemaLinker(BagOfWords(), top_k=1, fk_hops=1)

    tables, scores = linker.link(BIG, "What is the capacity of each stadium?")

    assert max(scores.tables, key=scores.tables.get) == "stadium"
    assert tables == ["concert", "stadium"]  # schema order; concert references stadium


def test_without_foreign_key_hops_only_top_k() -> None:
    tables, _ = SchemaLinker(BagOfWords(), top_k=2, fk_hops=0).link(BIG, "Which album titles won a prize?")

    assert set(tables) == {"album", "award"}


def test_lexical_bonus_goes_to_tables_the_question_names() -> None:
    plain = SchemaLinker(BagOfWords(), lexical_bonus=0.0).score(BIG, "list every festival")
    bonus = SchemaLinker(BagOfWords(), lexical_bonus=0.1).score(BIG, "list every festival")

    assert bonus.tables["festival"] - plain.tables["festival"] == pytest.approx(0.1)
    assert bonus.tables["album"] == plain.tables["album"]


def test_schema_docs_are_embedded_once_per_database() -> None:
    embedder = BagOfWords()
    linker = SchemaLinker(embedder)

    for question in ("a", "b", "c"):
        linker.link(BIG, question)

    assert embedder.calls == 2 + 3  # table and column docs once, then one call per question
