import sqlite3
from pathlib import Path

import pytest

from nl2sql.schema.introspect import introspect_sqlite, load_schema
from nl2sql.schema.serialize import LINKED_LEVELS, render, serialize_schema


def count_words(text: str) -> int:
    """Stand-in tokenizer: good enough to exercise the budget logic without transformers."""
    return len(text.split())


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "concert" / "concert.sqlite"
    path.parent.mkdir()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE stadium (stadium_id INTEGER PRIMARY KEY, name TEXT, capacity INT);
            CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, "order" INT, notes BLOB,
                                 "Home Town" TEXT);
            CREATE TABLE concert (concert_id INT PRIMARY KEY, stadium_id INT REFERENCES stadium,
                                  FOREIGN KEY (concert_id) REFERENCES missing_table(id));
            CREATE TABLE singer_in_concert (concert_id INT, singer_id INT,
                                            PRIMARY KEY (concert_id, singer_id),
                                            FOREIGN KEY (singer_id) REFERENCES SINGER(Singer_ID));
            INSERT INTO stadium VALUES (1, 'Stark''s Park', 10104), (2, 'Somerset Park', 11998);
            INSERT INTO singer VALUES (1, 'Joe Sharp', 1, x'00', 'a   very long home town name that goes on'),
                                      (2, 'Joe Sharp', 2, NULL, NULL), (3, '   ', 3, NULL, 'Paris');
            """
        )
    conn.close()
    return path


def test_introspection_reads_keys_and_samples(db: Path) -> None:
    schema = introspect_sqlite(db, "concert")

    assert [t.name for t in schema.tables] == ["stadium", "singer", "concert", "singer_in_concert"]
    singer = schema.tables[1]
    by_name = {c.name: c for c in singer.columns}
    assert by_name["singer_id"].is_pk and by_name["name"].type == "TEXT"
    assert by_name["name"].sample_values == ["Joe Sharp"]  # distinct, blank strings skipped
    assert by_name["notes"].sample_values == []  # blobs skipped
    assert by_name["Home Town"].sample_values == ["a very long home town name tha", "Paris"]  # cut to 30
    assert [c.name for c in schema.tables[3].primary_key] == ["concert_id", "singer_id"]


def test_foreign_keys_are_resolved_and_filtered(db: Path) -> None:
    fks = {
        (fk.from_table, fk.from_column, fk.to_table, fk.to_column)
        for fk in introspect_sqlite(db, "c").foreign_keys
    }

    assert fks == {
        ("concert", "stadium_id", "stadium", "stadium_id"),  # omitted target column -> target's PK
        ("singer_in_concert", "singer_id", "singer", "singer_id"),  # case-insensitive match, real casing kept
    }  # the FK to missing_table is dropped


def test_serialized_format(db: Path) -> None:
    text = serialize_schema(introspect_sqlite(db, "concert"), count_words, token_budget=10_000).text

    assert "CREATE TABLE stadium (\n  stadium_id INTEGER PRIMARY KEY," in text
    assert "-- e.g. 'Stark''s Park', 'Somerset Park'" in text  # values as SQL literals
    assert "  `order` INT," in text and "`Home Town` TEXT" in text  # reserved and non-simple names quoted
    assert "  PRIMARY KEY (concert_id, singer_id)\n);" in text
    assert "-- FK: concert.stadium_id -> stadium.stadium_id" in text


def test_budget_sheds_samples_before_anything_else(db: Path) -> None:
    schema = introspect_sqlite(db, "concert")
    full = serialize_schema(schema, count_words, token_budget=10_000)

    smaller = serialize_schema(schema, count_words, token_budget=full.tokens - 1)

    assert full.level == "full" and smaller.level == "no_samples"
    assert "e.g." not in smaller.text and "stadium" in smaller.text


def test_without_a_linker_every_table_is_kept_even_over_budget(db: Path) -> None:
    result = serialize_schema(introspect_sqlite(db, "concert"), count_words, token_budget=5)

    assert result.level == "over_budget"
    assert result.text.count("CREATE TABLE") == 4


SELECTED = ["Singer", "singer_in_concert"]


def test_linker_selected_tables_are_never_dropped(db: Path) -> None:
    schema = introspect_sqlite(db, "concert")
    no_samples = serialize_schema(schema, count_words, 10_000, tables=SELECTED, include_samples=False)

    keys_only = serialize_schema(schema, count_words, no_samples.tokens - 1, SELECTED, include_samples=False)
    smallest = serialize_schema(schema, count_words, token_budget=1, tables=SELECTED)

    assert keys_only.level == "keys_only_unselected"
    assert (
        "capacity" not in keys_only.text and "`order` INT" in keys_only.text
    )  # only unselected tables shrink
    assert smallest.level == "over_budget"
    assert smallest.text.count("CREATE TABLE") == 2 and "TABLE singer (" in smallest.text


def test_unselected_tables_go_before_selected_ones_lose_samples(db: Path) -> None:
    schema = introspect_sqlite(db, "concert")
    keys_only = serialize_schema(schema, count_words, 10_000, tables=SELECTED, include_samples=False)
    full = serialize_schema(schema, count_words, 10_000, tables=SELECTED)

    result = serialize_schema(schema, count_words, full.tokens - 1, tables=SELECTED)

    assert keys_only.tokens < full.tokens
    assert result.level in ("keys_only_unselected", "selected_only")
    assert "'Joe Sharp'" in result.text and "capacity" not in result.text


def test_columns_are_pruned_by_linker_score_when_nothing_else_fits(db: Path) -> None:
    schema = introspect_sqlite(db, "concert")
    smallest_level = count_words(render(schema, {"singer"}, LINKED_LEVELS[-1]))
    scores = {("singer", "Home Town"): 0.9, ("singer", "name"): 0.5, ("singer", "order"): 0.1}

    result = serialize_schema(
        schema, count_words, smallest_level - 2, tables=["singer"], column_scores=scores
    )

    assert result.level == "pruned_columns" and result.tokens <= smallest_level - 2
    assert "singer_id INTEGER PRIMARY KEY" in result.text  # keys always stay
    assert "`Home Town`" in result.text and "notes" not in result.text


def test_bird_descriptions_are_merged(db: Path) -> None:
    desc = db.parent / "database_description"
    desc.mkdir()
    csv_text = (
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "name,,name,text,\n"
        'Home Town ,home town,where the singer grew up,text,"commonsense evidence:\n\ncaf\xe9 capital"\n'
    )
    (desc / "singer.csv").write_bytes(csv_text.encode("cp1252"))  # a few BIRD files are not UTF-8

    schema = introspect_sqlite(db, "concert")

    by_name = {c.name: c for c in schema.tables[1].columns}
    assert by_name["name"].description is None  # every part only repeats the column name
    assert by_name["Home Town"].description == "where the singer grew up; café capital"
    text = serialize_schema(schema, count_words, token_budget=10_000).text
    assert "-- where the singer grew up; café capital | e.g. " in text


def test_schema_cache_round_trips(db: Path, tmp_path: Path) -> None:
    first = load_schema(db, "concert", cache_dir=tmp_path / "cache")
    second = load_schema(db, "concert", cache_dir=tmp_path / "cache")

    assert first == second
    assert len(list((tmp_path / "cache").glob("concert-*.json"))) == 1
