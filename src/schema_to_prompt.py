from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import sys

# Project root
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCHEMA_PATH = PROJECT_ROOT / "schema.json"
PROMPT_OUT = PROJECT_ROOT / "prompts.jsonl"

# Spider schema loader
from src.table_retrieval.schemas import load_spider_schemas


# -----------------------------------------------------
# Helper: CREATE TABLE block
# -----------------------------------------------------
def make_create_block(table_name: str, columns: List[List[str]]) -> str:
    col_lines = [f"  {name} {ctype.upper()}" for name, ctype in columns]
    return f"CREATE TABLE {table_name} (\n" + ",\n".join(col_lines) + "\n);"


# -----------------------------------------------------
# Helper: FK comment block
# -----------------------------------------------------
def make_fk_comments(fk_pairs: List[List[str]]) -> str:
    if not fk_pairs:
        return "-- No foreign key relationships"
    return "\n".join([f"-- {src} = {tgt}" for src, tgt in fk_pairs])


# -----------------------------------------------------
# Helper: Build prompt
# -----------------------------------------------------

# GENERAL SQLCODER PROMPTING TEMPLATE
# def build_prompt(question: str, create_tables: List[str], fk_text: str):
#     tables_block = "\n\n".join(create_tables)

#     return f"""
#         ## Task
#         Generate a SQL query to answer the following question:
#         `{question}`

#         ### Database Schema
#         This query will run on a database whose schema is represented in this string:
#         {tables_block}

#         {fk_text}

#         ### SQL
#         Given the database schema, here is the SQL query that answers `{question}`:
#         ```sql
#         """.strip()


# GBNF PROMPTING TEMPLATE WITHOUT ANY BACKTICKS: `
def build_prompt(question, create_blocks, fk_text):
    tables_block = "\n\n".join(create_blocks)

    prompt = (
        "## Task\n"
        "Generate a SQLite SQL query to answer the following question:\n"
        f"{question}\n\n"
        "### Database Schema\n"
        "This query will run on a database whose schema is represented as:\n"
        f"{tables_block}\n\n"
        f"{fk_text}\n\n"
        "### SQL\n"
        "Given the database schema, Write only the valid SQLite SQL query that answers the question:\n"
    )

    return prompt

# Extra prompt guidance: Use the **fewest tables necessary** to answer the question, and do **not** introduce extra joins or columns that are not clearly required.


# -----------------------------------------------------
# Main logic: FK expansion + prompt generation
# -----------------------------------------------------
def main():
    # Load schema.json
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_entries = json.load(f)

    # Load full Spider schema
    spider_tables = load_spider_schemas(PROJECT_ROOT / "spider_dataset" / "spider_data" / "tables.json")
    spider_db_map = {db.db_id: db for db in spider_tables}

    print(f"[INFO] Loaded {len(schema_entries)} entries from schema.json")
    print(f"[INFO] Loaded {len(spider_db_map)} databases from Spider")

    with open(PROMPT_OUT, "w", encoding="utf-8") as fout:
        for entry in schema_entries:
            db_id = entry["db_id"]
            question = entry["question"]
            candidate = entry["candidate_tables"]
            fk_pairs = entry["fk_pairs"]

            spider_db = spider_db_map[db_id]
            table_lookup = {t.name: t for t in spider_db.tables}

            # 1. Start with retrieved tables
            tables_needed = {tbl["table_name"]: tbl["columns"] for tbl in candidate}

            # 2. Expand via FK pairs
            for src, tgt in fk_pairs:
                src_table = src.split(".")[0]
                tgt_table = tgt.split(".")[0]

                for tbl in [src_table, tgt_table]:
                    if tbl not in tables_needed:
                        # Get from Spider metadata
                        spider_tbl = table_lookup[tbl]
                        cols = [(c.name, c.type) for c in spider_tbl.columns]
                        tables_needed[tbl] = cols

            # 3. Build CREATE TABLE blocks
            create_blocks = [
                make_create_block(tname, cols)
                for tname, cols in tables_needed.items()
            ]

            # 4. Build FK comments
            fk_block = make_fk_comments(fk_pairs)

            # 5. Build final prompt
            prompt = build_prompt(question, create_blocks, fk_block)

            # 6. Save to prompts.jsonl
            fout.write(json.dumps({
                "question_id": entry["question_id"],
                "db_id": db_id,
                "question": question,
                "prompt": prompt
            }) + "\n")

    print(f"[SUCCESS] prompts.jsonl written → {PROMPT_OUT}")


if __name__ == "__main__":
    main()
