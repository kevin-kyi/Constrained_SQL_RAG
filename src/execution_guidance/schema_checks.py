from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, Any, Tuple
import json
import re


def load_spider_schema(tables_json_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load Spider tables.json and build a simple schema index:

    schema_index[db_id] = {
        'tables': set of table names (lowercased),
        'columns': dict[table_name_lower] -> set of column names (lowercased)
    }
    """
    with tables_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    index: Dict[str, Dict[str, Any]] = {}

    for db in data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]  # list[str]
        column_names = db["column_names_original"]  # list[[table_id, col_name], ...]

        table_set: Set[str] = set()
        table_to_cols: Dict[str, Set[str]] = {}

        # normalize table names
        for t in table_names:
            t_lower = t.lower()
            table_set.add(t_lower)
            table_to_cols.setdefault(t_lower, set())

        # assign columns to tables
        for tbl_idx, col_name in column_names:
            if tbl_idx == -1:
                # special "*"
                continue
            if not col_name:
                continue
            t_name = table_names[tbl_idx].lower()
            c_lower = col_name.lower()
            table_to_cols.setdefault(t_name, set()).add(c_lower)

        index[db_id] = {
            "tables": table_set,
            "columns": table_to_cols,
        }

    return index


_TABLE_FROM_JOIN_PATTERN = re.compile(
    r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    r"|\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    flags=re.IGNORECASE,
)


def extract_tables_from_sql(sql: str) -> Set[str]:
    """
    Very lightweight parser: extract table names from FROM / JOIN clauses.

    It ignores aliases and only grabs the raw table token immediately after
    FROM or JOIN. This is not perfect, but works decently on Spider-style SQL.
    """
    tables: Set[str] = set()
    for match in _TABLE_FROM_JOIN_PATTERN.finditer(sql):
        g1, g2 = match.groups()
        name = g1 or g2
        if not name:
            continue
        # strip trailing punctuation, just in case
        name = name.rstrip(",;")
        tables.add(name.lower())
    return tables


def table_schema_check_simple(
    db_id: str,
    sql: str,
    schema_index: Dict[str, Dict[str, Any]],
) -> bool:
    """
    Simple schema check:
      - extract tables from SQL
      - ensure all of them exist in the Spider schema for this db_id

    If db_id is unknown or we can't find schema, we return True (do not penalize).
    """
    db_schema = schema_index.get(db_id)
    if db_schema is None:
        # No schema info → don't block anything
        return True

    all_tables = db_schema["tables"]
    used_tables = extract_tables_from_sql(sql)

    # If we didn't detect any tables (e.g., weird SQL), don't penalize.
    if not used_tables:
        return True

    # If any table is not in the schema, reject this candidate.
    for t in used_tables:
        if t not in all_tables:
            return False

    return True
