"""
schema_to_prompt.py

Utilities to:
- Load Spider schema from tables.json
- Expand a set of candidate tables by following foreign keys (recursive)
- Build a text prompt for SQLCoder from a schema.json entry

Expected schema.json format under spider_dataset/:

[
  {
    "db_id": "flight_1",
    "question": "Which airport has the highest number of departures?",
    "candidate_tables": [
      {
        "table_name": "flights",
        "columns": [
          ["flight_id", "int"],
          ["origin_airport", "text"],
          ["destination_airport", "text"]
        ]
      },
      {
        "table_name": "airports",
        "columns": [
          ["airport_code", "text"],
          ["airport_name", "text"],
          ["city", "text"]
        ]
      }
    ]
  },
  ...
]
"""

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Set


# Paths (relative to repo root)
SPIDER_DATA_ROOT = Path("spider_dataset") / "spider_data"
TABLES_JSON = SPIDER_DATA_ROOT / "tables.json"
SCHEMA_JSON = Path("spider_dataset") / "schema.json"


# ──────────────────────────────────────────────────────────────────────────────
# Load Spider schema from tables.json
# ──────────────────────────────────────────────────────────────────────────────

class SpiderSchema:
    """
    Convenience wrapper for Spider's tables.json
    Provides:
        - table names, columns, FKs per db_id
        - table-level adjacency via FKs
        - helpers to expand tables by following FK edges
    """
    def __init__(self, tables_json_path: Path):
        if not tables_json_path.is_file():
            raise FileNotFoundError(
                f"tables.json not found at {tables_json_path}. "
                "Make sure you placed the Spider dataset in spider_dataset/spider_data/"
            )
        with open(tables_json_path, "r", encoding="utf-8") as f:
            tables_data = json.load(f)

        self.by_db: Dict[str, Dict] = {}

        for db in tables_data:
            db_id = db["db_id"]
            table_names = db["table_names_original"]  # list[str]
            column_names = db["column_names_original"]  # list[[table_idx, col_name], ...]
            foreign_keys = db["foreign_keys"]  # list[[col_idx1, col_idx2], ...]

            # Map column index -> table index
            col_to_table = [tbl_idx for (tbl_idx, _col_name) in column_names]

            # Build table-level adjacency from foreign_keys
            table_adj: Dict[int, Set[int]] = defaultdict(set)
            fk_edges: List[Tuple[int, int]] = []  # (col_idx1, col_idx2)
            for col1, col2 in foreign_keys:
                t1 = col_to_table[col1]
                t2 = col_to_table[col2]
                if t1 == -1 or t2 == -1:
                    continue  # skip special "all tables" column index
                table_adj[t1].add(t2)
                table_adj[t2].add(t1)
                fk_edges.append((col1, col2))

            self.by_db[db_id] = {
                "table_names": table_names,
                "column_names": column_names,
                "col_to_table": col_to_table,
                "foreign_keys": fk_edges,
                "table_adj": table_adj,
            }

    def expand_tables_via_fks(self, db_id: str, initial_table_names: List[str]) -> Tuple[List[int], List[Tuple[str, str]]]:
        """
        Given a db_id and a list of table names (strings),
        return:
          - a sorted list of table indices reachable via FK edges
            starting from the initial set (recursive BFS),
          - a list of FK edges between those tables, in name form:
                [("flights.origin_airport", "airports.airport_code"), ...]
        """
        if db_id not in self.by_db:
            raise KeyError(f"db_id {db_id} not found in Spider tables.json")

        schema = self.by_db[db_id]
        table_names = schema["table_names"]
        column_names = schema["column_names"]
        col_to_table = schema["col_to_table"]
        table_adj = schema["table_adj"]
        foreign_keys = schema["foreign_keys"]

        # Map table_name -> index (using original names)
        name_to_idx = {name.lower(): i for i, name in enumerate(table_names)}

        # Convert initial table names to indices, ignoring unknowns
        start_indices: Set[int] = set()
        for tname in initial_table_names:
            key = tname.lower()
            if key in name_to_idx:
                start_indices.add(name_to_idx[key])
            else:
                # Best-effort: warn via comment; in real code you might log
                print(f"[WARN] Table name '{tname}' not found in db '{db_id}' table_names_original.")

        if not start_indices:
            # If we couldn't map anything, bail out with empty
            return [], []

        # BFS to expand by foreign-key adjacency (table-level graph)
        visited: Set[int] = set()
        q: deque = deque()

        for idx in start_indices:
            visited.add(idx)
            q.append(idx)

        while q:
            cur = q.popleft()
            for nei in table_adj[cur]:
                if nei not in visited:
                    visited.add(nei)
                    q.append(nei)

        # Now visited = all reachable tables via FK graph (including starting ones)
        sorted_table_idxs = sorted(visited)

        # Collect FK edges that lie entirely inside this visited set, in name form
        fk_pairs_named: List[Tuple[str, str]] = []
        for col1, col2 in foreign_keys:
            t1 = col_to_table[col1]
            t2 = col_to_table[col2]
            if t1 in visited and t2 in visited:
                # Build "table.col" names
                tbl1_name = table_names[t1]
                tbl2_name = table_names[t2]

                col1_name = column_names[col1][1]
                col2_name = column_names[col2][1]

                fk_pairs_named.append(
                    (f"{tbl1_name}.{col1_name}", f"{tbl2_name}.{col2_name}")
                )

        return sorted_table_idxs, fk_pairs_named


# ──────────────────────────────────────────────────────────────────────────────
# schema.json interface and prompt construction
# ──────────────────────────────────────────────────────────────────────────────

def load_schema_entries(schema_json_path: Path = SCHEMA_JSON):
    """Load the partner-produced schema.json entries."""
    if not schema_json_path.is_file():
        raise FileNotFoundError(
            f"schema.json not found at {schema_json_path}. "
            "Make sure your retrieval pipeline has written it."
        )
    with open(schema_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt_for_entry(entry: Dict, spider_schema: SpiderSchema) -> str:
    """
    Given a schema.json entry and a SpiderSchema, build a SQLCoder-ready prompt.

    1. Take the db_id and candidate_tables from the entry
    2. Use SpiderSchema.expand_tables_via_fks to:
           - expand the table set by following FKs (recursive)
           - gather FK edges between them
    3. Build a textual prompt with:
           - Question
           - Database schema (tables + columns)
           - Foreign keys
           - "Write a SQL query..." + "SQL:" marker
    """
    db_id = entry["db_id"]
    question = entry["question"]
    candidate_tables = entry["candidate_tables"]

    # Initial table names from the entry
    initial_table_names = [tbl["table_name"] for tbl in candidate_tables]

    # Expand via FKs using Spider's tables.json
    extended_table_idxs, fk_pairs_named = spider_schema.expand_tables_via_fks(
        db_id, initial_table_names
    )

    db_schema = spider_schema.by_db[db_id]
    spider_table_names = db_schema["table_names"]
    # Note: we do NOT replace candidate_tables' columns; we trust partner's column info.
    # We just expand the table set and add FK info.

    # Build a map from table_name.lower() -> provided columns
    partner_cols_by_table = {
        tbl["table_name"].lower(): tbl.get("columns", [])
        for tbl in candidate_tables
    }

    # Construct schema lines for all extended tables
    schema_lines: List[str] = []
    for t_idx in extended_table_idxs:
        t_name = spider_table_names[t_idx]
        # Use partner's column types if present, else we just print the name
        cols = partner_cols_by_table.get(t_name.lower(), [])
        if cols:
            col_str = ", ".join(f"{cname} {ctype}".upper() for cname, ctype in cols)
            schema_lines.append(f"- Table {t_name}({col_str})")
        else:
            schema_lines.append(f"- Table {t_name}")

    # Foreign key lines
    fk_lines = [f"- {src} -> {tgt}" for (src, tgt) in fk_pairs_named]

    # Final prompt
    lines: List[str] = []
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Database ID: " + db_id)
    lines.append("")
    lines.append("Database schema:")
    if schema_lines:
        lines.extend(schema_lines)
    else:
        lines.append("- (no tables found)")
    lines.append("")
    lines.append("Foreign keys:")
    if fk_lines:
        lines.extend(fk_lines)
    else:
        lines.append("- (no foreign key relationships found)")
    lines.append("")
    lines.append("Write a SQL query that correctly answers the question using the schema above.")
    lines.append("")
    lines.append("SQL:")

    return "\n".join(lines)


def main():
    # Small CLI: print prompts for the first few entries in schema.json
    spider_schema = SpiderSchema(TABLES_JSON)
    entries = load_schema_entries()

    print(f"[INFO] Loaded {len(entries)} entries from {SCHEMA_JSON}")
    for i, entry in enumerate(entries[:5], 1):
        prompt = build_prompt_for_entry(entry, spider_schema)
        print(f"\n========== PROMPT {i} (db_id={entry['db_id']}) ==========\n")
        print(prompt)
        print("\n=============================================\n")


if __name__ == "__main__":
    main()
