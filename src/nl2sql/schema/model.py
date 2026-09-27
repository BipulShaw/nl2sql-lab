"""Database schema as data: what introspection produces and what the serializer and linker consume."""

from pydantic import BaseModel


class Column(BaseModel):
    name: str
    type: str = ""  # declared type, upper-cased; SQLite allows none
    is_pk: bool = False
    pk_position: int = 0  # 1-based position inside a composite primary key; 0 if not part of the key
    description: str | None = None  # BIRD column descriptions; None for Spider
    sample_values: list[str | int | float] = []


class ForeignKey(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str


class Table(BaseModel):
    name: str
    columns: list[Column]

    @property
    def primary_key(self) -> list[Column]:
        return sorted((c for c in self.columns if c.is_pk), key=lambda c: c.pk_position)


class Schema(BaseModel):
    db_id: str
    tables: list[Table]
    foreign_keys: list[ForeignKey] = []
