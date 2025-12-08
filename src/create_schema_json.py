from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List

# -----------------------------------------------------
# Add PROJECT_ROOT to sys.path
# -----------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # /Constrained_SQL_RAG
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------
# Project imports
# -----------------------------------------------------
from src.config import DEFAULT_CONFIG
from src.table_retrieval.schemas import (
    DatabaseSchema,
    load_spider_schemas,
    flatten_tables,
)
from src.table_retrieval.retriever import TableRetriever


# -----------------------------------------------------
# Load Spider split
# -----------------------------------------------------
def load_spider_split(path: Path, num_samples: int | None = None, seed: int = 0) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if num_samples is not None and 0 < num_samples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, num_samples)

    # add question_id inside the entry for reproducibility
    for idx, ex in enumerate(data):
        ex["question_id"] = idx

    return data


# -----------------------------------------------------
# Build db_id → DatabaseSchema lookup
# -----------------------------------------------------
def build_db_map(schemas: List[DatabaseSchema]) -> Dict[str, DatabaseSchema]:
    return {db.db_id: db for db in schemas}


# -----------------------------------------------------
# Extract metadata from TableSchema (columns, PK, FK)
# -----------------------------------------------------
def build_table_metadata(table_schema):
    """
    Convert TableSchema → dict format expected by schema.json.
    """

    # Columns are Column(name, type)
    columns = [(c.name, c.type) for c in table_schema.columns]

    # Primary keys are just strings
    primary_keys = list(table_schema.primary_keys)

    # ForeignKey = dataclass(source_table, source_column, target_table, target_column)
    fk_pairs = []
    for fk in table_schema.foreign_keys:
        fk_pairs.append([f"{fk.source_table}.{fk.source_column}", f"{fk.target_table}.{fk.target_column}"])

    return {
        "columns": columns,
        "primary_keys": primary_keys,
        "foreign_keys": fk_pairs,
    }


# -----------------------------------------------------
# Build one schema.json entry
# -----------------------------------------------------
def build_schema_entry(question: str, db_id: str, db_schema: DatabaseSchema, retriever: TableRetriever,
                       top_k: int, question_id: int):

    hits = retriever.search(
        query=question,
        top_k=top_k,
        limit_to_db=db_id
    )

    candidate_tables = []
    fk_pairs_flat = []

    # Use db_schema.table_map() properly (table_map is a method!)
    table_lookup = db_schema.table_map()

    for hit in hits:
        tname = hit.table.name
        if tname not in table_lookup:
            continue

        table_schema = table_lookup[tname]
        meta = build_table_metadata(table_schema)

        candidate_tables.append({
            "table_name": tname,
            "columns": meta["columns"],
            "primary_keys": meta["primary_keys"],
            "foreign_keys": meta["foreign_keys"],
        })

        fk_pairs_flat.extend(meta["foreign_keys"])

    return {
        "db_id": db_id,
        "question_id": question_id,
        "question": question,
        "candidate_tables": candidate_tables,
        "fk_pairs": fk_pairs_flat,
    }


# -----------------------------------------------------
# Main: Create schema.json + questions_used.jsonl
# -----------------------------------------------------
def create_schema_json(split_path: Path, out_schema: Path, out_questions: Path,
                       top_k: int = 5, num_samples: int = 50, seed: int = 0):

    config = DEFAULT_CONFIG
    spider_paths = config.spider_paths

    print(f"[INFO] Loading schemas from: {spider_paths.tables_json}")
    schemas = load_spider_schemas(spider_paths.tables_json)
    db_map = build_db_map(schemas)

    # -----------------------------------------------------
    # Build TableRetriever
    # -----------------------------------------------------
    print("[INFO] Building TableRetriever...")
    all_tables = flatten_tables(schemas)

    retriever = TableRetriever(
        tables=all_tables,
        mode="hybrid",  
        dense_model_name=config.retrieval.dense_model_name,
        top_k_sparse=config.retrieval.top_k_sparse,
        top_k_dense=config.retrieval.top_k_dense,
        hybrid_weight_sparse=config.retrieval.hybrid_weight_sparse,
        hybrid_weight_dense=config.retrieval.hybrid_weight_dense,
        hybrid_strategy=config.retrieval.hybrid_strategy,
        rrf_k=config.retrieval.rrf_k,
    )

    # -----------------------------------------------------
    # Load questions
    # -----------------------------------------------------
    print(f"[INFO] Loading Spider data from: {split_path}")
    examples = load_spider_split(split_path, num_samples=num_samples, seed=seed)
    print(f"[INFO] Selected {len(examples)} questions for schema.json")

    schema_entries = []

    with out_questions.open("w", encoding="utf-8") as fq:
        for ex in examples:
            question = ex["question"]
            db_id = ex["db_id"]
            gold_sql = ex.get("query") or ex.get("sql_query") or ""
            qid = ex["question_id"]

            if db_id not in db_map:
                print(f"[WARN] db_id={db_id} not found in tables.json — skipping.")
                continue

            db_schema = db_map[db_id]

            # Save reproducibility entry
            fq.write(json.dumps({
                "db_id": db_id,
                "question_id": qid,
                "question": question,
                "gold_sql": gold_sql
            }) + "\n")

            # Build schema.json entry
            entry = build_schema_entry(
                question=question,
                db_id=db_id,
                db_schema=db_schema,
                retriever=retriever,
                top_k=top_k,
                question_id=qid,
            )
            schema_entries.append(entry)

    # -----------------------------------------------------
    # Write schema.json
    # -----------------------------------------------------
    with out_schema.open("w", encoding="utf-8") as f:
        json.dump(schema_entries, f, indent=2)

    print(f"[SUCCESS] schema.json saved → {out_schema}")
    print(f"[SUCCESS] questions_used.jsonl saved → {out_questions}")


# ---------------------------------------------------------------------
# CLI — run with: python3 src/create_schema_json.py
# ---------------------------------------------------------------------
if __name__ == "__main__":
    config = DEFAULT_CONFIG
    spider_paths = config.spider_paths

    split_path = spider_paths.data_dir / "dev.json"

    create_schema_json(
        split_path=split_path,
        out_schema=PROJECT_ROOT / "schema.json",
        out_questions=PROJECT_ROOT / "questions_used.jsonl",
        top_k=5,
        num_samples=200,
        seed=0,
    )
