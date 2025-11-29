import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Any

# Resolve paths
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]  # Constrained_SQL_RAG/
SPIDER_DATA_ROOT = REPO_ROOT / "spider_dataset" / "spider_data"
TABLES_JSON = SPIDER_DATA_ROOT / "tables.json"
TRAIN_JSON = SPIDER_DATA_ROOT / "train_spider.json"
SCHEMA_JSON_OUT = REPO_ROOT / "spider_dataset" / "schema.json"


def load_spider_schema() -> Dict[str, Dict[str, Any]]:
    """
    Load tables.json and return a mapping:
        db_id -> {
            "table_names": [...],
            "column_names": [...],
            "column_types": [...],
        }
    """
    if not TABLES_JSON.is_file():
        raise FileNotFoundError(
            f"tables.json not found at {TABLES_JSON}. "
            "Make sure the Spider dataset is in spider_dataset/spider_data/."
        )

    with open(TABLES_JSON, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    schema_by_db: Dict[str, Dict[str, Any]] = {}
    for db in tables_data:
        db_id = db["db_id"]
        schema_by_db[db_id] = {
            "table_names": db["table_names_original"],    # list[str]
            "column_names": db["column_names_original"],  # list[[table_idx, col_name], ...]
            "column_types": db["column_types"],           # list[str], same index as column_names
        }

    return schema_by_db


def load_train_examples() -> List[Dict[str, Any]]:
    """Load all examples from train_spider.json."""
    if not TRAIN_JSON.is_file():
        raise FileNotFoundError(
            f"train_spider.json not found at {TRAIN_JSON}. "
            "Make sure the Spider dataset is in spider_dataset/spider_data/."
        )
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_candidate_tables_for_db(db_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build candidate_tables list for a given db schema using ALL tables in that DB.

    Each candidate table has:
        {
            "table_name": str,
            "columns": [
                ["col_name", "col_type"],
                ...
            ]
        }
    """
    table_names = db_schema["table_names"]
    column_names = db_schema["column_names"]
    column_types = db_schema["column_types"]

    # Map: table_idx -> list of (col_name, col_type)
    cols_by_table_idx: Dict[int, List[List[str]]] = {}
    for (tbl_idx, col_name), col_type in zip(column_names, column_types):
        if tbl_idx == -1:
            # special "all tables" index, skip
            continue
        cols_by_table_idx.setdefault(tbl_idx, []).append([col_name, col_type])

    candidate_tables: List[Dict[str, Any]] = []
    for t_idx, t_name in enumerate(table_names):
        cols = cols_by_table_idx.get(t_idx, [])
        candidate_tables.append(
            {
                "table_name": t_name,
                "columns": cols,
            }
        )

    return candidate_tables


def main(num_examples: int = 5, seed: int = 42) -> None:
    print("[INFO] Building example schema.json ...")
    schema_by_db = load_spider_schema()
    train_examples = load_train_examples()

    if len(train_examples) == 0:
        raise RuntimeError("No examples found in train_spider.json.")

    random.seed(seed)
    samples = random.sample(train_examples, k=min(num_examples, len(train_examples)))
    print(f"[INFO] Sampled {len(samples)} examples from train_spider.json")

    output_entries: List[Dict[str, Any]] = []

    for i, ex in enumerate(samples, 1):
        db_id = ex["db_id"]
        question = ex["question"]

        if db_id not in schema_by_db:
            print(f"[WARN] db_id {db_id} not found in tables.json; skipping example.")
            continue

        db_schema = schema_by_db[db_id]
        candidate_tables = build_candidate_tables_for_db(db_schema)

        entry = {
            "db_id": db_id,
            "question": question,
            "candidate_tables": candidate_tables,
        }
        output_entries.append(entry)
        print(f"[INFO] Added example {i}: db_id={db_id}, question='{question}'")

    # Write to spider_dataset/schema.json
    SCHEMA_JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEMA_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(output_entries, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Wrote {len(output_entries)} entries to {SCHEMA_JSON_OUT}")


if __name__ == "__main__":
    # You can optionally pass a different number of examples via CLI, e.g.:
    #   python src/test_script/build_example_schema_json.py 3
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print("[WARN] Invalid num_examples arg, defaulting to 5.")
            n = 5
    else:
        n = 5

    main(num_examples=n)
