from __future__ import annotations

from .schemas import DatabaseSchema, TableSchema


def _format_columns(table: TableSchema) -> str:
    parts: list[str] = []
    for column in table.columns:
        parts.append(f"{column.name} {column.type}")
    return ", ".join(parts)


def format_table_schema(table: TableSchema) -> str:
    """Readable one-line string for a single table (db id, name, columns)."""
    return f"- {table.database}.{table.name}({_format_columns(table)})"


def format_database_schema(database: DatabaseSchema) -> str:
    """Readable schema for an entire database."""
    lines = [f"Database {database.db_id} schema:"]
    for table in database.tables:
        lines.append(format_table_schema(table))
    return "\n".join(lines)
