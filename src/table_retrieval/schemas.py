from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable


@dataclasses.dataclass
class Column:
    name: str
    type: str


@dataclasses.dataclass
class ForeignKey:
    source_table: str
    source_column: str
    target_table: str
    target_column: str


@dataclasses.dataclass
class TableSchema:
    database: str
    name: str
    columns: list[Column]
    primary_keys: list[str]
    foreign_keys: list[ForeignKey]

    @property
    def identifier(self) -> str:
        return f"{self.database}.{self.name}"

    def as_retrieval_text(self) -> str:
        """Flatten this table into a string for lexical retrieval."""
        def _expand(name: str) -> str:
            spaced = name.replace("_", " ")
            if spaced == name:
                return name
            return f"{name} {spaced}"

        friendly_table_name = _expand(self.name)
        column_text = ", ".join(f"{_expand(column.name)} ({column.type})" for column in self.columns)
        pk_text = (
            f"primary key: {', '.join(self.primary_keys)}"
            if self.primary_keys
            else "primary key: none"
        )
        if self.foreign_keys:
            fk_text = "; ".join(
                f"{fk.source_column} -> {fk.target_table}.{fk.target_column}" for fk in self.foreign_keys
            )
            fk_text = f"foreign keys: {fk_text}"
        else:
            fk_text = "foreign keys: none"
        return (
            f"{self.identifier} {friendly_table_name} database {self.database}. "
            f"columns: {column_text}. {pk_text}. {fk_text}."
        )


@dataclasses.dataclass
class DatabaseSchema:
    db_id: str
    tables: list[TableSchema]

    def table_map(self) -> dict[str, TableSchema]:
        return {table.name: table for table in self.tables}


def _extract_column_lookup(column_names: list[List[str]]) -> dict[int, tuple[int, str]]:
    """Map column index -> (table_idx, column_name)."""
    lookup: dict[int, tuple[int, str]] = {}
    for idx, (table_idx, column_name) in enumerate(column_names):
        lookup[idx] = (table_idx, column_name)
    return lookup


def load_spider_schemas(tables_path: Path) -> list[DatabaseSchema]:
    """Load SPIDER table metadata into structured dataclasses."""
    with tables_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    databases: list[DatabaseSchema] = []
    for entry in payload:
        db_id: str = entry["db_id"]
        table_names: list[str] = entry.get("table_names_original") or entry["table_names"]
        column_names: list[list[str]] = entry.get("column_names_original") or entry["column_names"]
        column_types: list[str] = entry["column_types"]
        primary_keys: list[int] = entry.get("primary_keys", [])
        foreign_keys: list[list[int]] = entry.get("foreign_keys", [])

        tables: list[TableSchema] = [
            TableSchema(database=db_id, name=table_name, columns=[], primary_keys=[], foreign_keys=[])
            for table_name in table_names
        ]
        column_lookup = _extract_column_lookup(column_names)

        for idx, (table_idx, column_name) in enumerate(column_names):
            if table_idx == -1:
                continue
            tables[table_idx].columns.append(Column(name=column_name, type=column_types[idx]))

        for pk_idx in primary_keys:
            table_idx, column_name = column_lookup[pk_idx]
            if table_idx == -1:
                continue
            tables[table_idx].primary_keys.append(column_name)

        for source_idx, target_idx in foreign_keys:
            source_table_idx, source_column = column_lookup[source_idx]
            target_table_idx, target_column = column_lookup[target_idx]
            if source_table_idx == -1 or target_table_idx == -1:
                continue
            tables[source_table_idx].foreign_keys.append(
                ForeignKey(
                    source_table=tables[source_table_idx].name,
                    source_column=source_column,
                    target_table=tables[target_table_idx].name,
                    target_column=target_column,
                )
            )

        databases.append(DatabaseSchema(db_id=db_id, tables=tables))
    return databases


def flatten_tables(databases: Iterable[DatabaseSchema]) -> list[TableSchema]:
    """Flatten all tables across databases for indexing."""
    flat: list[TableSchema] = []
    for db in databases:
        flat.extend(db.tables)
    return flat
