from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Set

# -----------------------------------------------------
# Add PROJECT_ROOT to sys.path so `import src.*` works
# -----------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # Constrained_SQL_RAG/
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


# ----------------------------
# Data loading helpers
# ----------------------------
def load_spider_split(path: Path, num_examples: int | None = None, seed: int = 0) -> List[dict]:
    """Load a Spider split (train/dev JSON) and optionally subsample."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if num_examples is not None and num_examples > 0 and num_examples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, num_examples)

    return data


def build_db_map(schemas: List[DatabaseSchema]) -> Dict[str, DatabaseSchema]:
    """Map db_id -> DatabaseSchema for quick lookup."""
    return {db.db_id: db for db in schemas}


# ----------------------------
# Gold table extraction
# ----------------------------
def extract_gold_tables_from_ast(example: dict, db_schema: DatabaseSchema) -> Set[str]:
    """
    Extract gold table names from Spider's structured SQL AST.

    Spider's `sql` field has the structure:
        sql["from"]["table_units"] -> list of table units

    Common pattern:
        ["table_unit", table_id]

    We map table_id (index into db_schema.tables) -> table name.
    """
    sql = example.get("sql")
    if not isinstance(sql, dict):
        return set()

    from_clause = sql.get("from", {})
    if not isinstance(from_clause, dict):
        return set()

    table_units = from_clause.get("table_units", [])
    if not isinstance(table_units, list):
        return set()

    table_ids: Set[int] = set()

    for unit in table_units:
        # Typical: ["table_unit", table_id]
        if not isinstance(unit, (list, tuple)) or len(unit) < 2:
            continue

        unit_type, table_id = unit[0], unit[1]

        # Standard Spider format
        if unit_type == "table_unit" and isinstance(table_id, int) and table_id >= 0:
            table_ids.add(table_id)
            continue

        # In case of alternative encodings like [table_id, alias]
        if isinstance(unit_type, int) and unit_type >= 0:
            table_ids.add(unit_type)

    gold_tables: Set[str] = set()
    for tid in table_ids:
        if 0 <= tid < len(db_schema.tables):
            gold_tables.add(db_schema.tables[tid].name)

    return gold_tables


def extract_gold_tables_from_text(sql_query: str, db_schema: DatabaseSchema) -> Set[str]:
    """
    Fallback: heuristically extract gold tables by matching table names as substrings
    in the SQL text. This is less reliable than the AST, so it's only used when
    the AST-based method fails.
    """
    if not sql_query:
        return set()

    sql_lower = sql_query.lower()
    gold_tables: Set[str] = set()

    for table in db_schema.tables:
        name = table.name.lower()
        if (
            f" {name} " in sql_lower
            or f"{name}." in sql_lower
            or f"`{name}`" in sql_lower
            or f'"{name}"' in sql_lower
        ):
            gold_tables.add(table.name)

    return gold_tables


# ----------------------------
# Main evaluation logic
# ----------------------------
def evaluate_retriever(
    split_path: Path,
    mode: str = "bm25",
    top_k: int = 5,
    num_examples: int | None = 100,
    seed: int = 0,
) -> None:
    """
    Build the retriever and evaluate table retrieval on a subset of Spider examples.

    Metrics:
      - avg_gold_tables:      average number of gold tables per example
      - avg_retrieved_tables: average number of retrieved tables per example
      - avg_precision:        mean(|gold ∩ retrieved| / |retrieved|)
      - avg_recall:           mean(|gold ∩ retrieved| / |gold|)
      - avg_f1:               mean(2 * P * R / (P + R)) per example
      - full_recall_rate:     fraction of examples where all gold tables were retrieved
    """
    config = DEFAULT_CONFIG
    spider_paths = config.spider_paths

    print(f"[INFO] Loading schemas from: {spider_paths.tables_json}")
    schemas = load_spider_schemas(spider_paths.tables_json)
    db_map = build_db_map(schemas)
    all_tables = flatten_tables(schemas)

    print(f"[INFO] Building TableRetriever in mode='{mode}' over {len(all_tables)} tables...")
    retriever = TableRetriever(
        tables=all_tables,
        mode=mode,
        dense_model_name=config.retrieval.dense_model_name,
        top_k_sparse=config.retrieval.top_k_sparse,
        top_k_dense=config.retrieval.top_k_dense,
        hybrid_weight_sparse=config.retrieval.hybrid_weight_sparse,
        hybrid_weight_dense=config.retrieval.hybrid_weight_dense,
        hybrid_strategy=config.retrieval.hybrid_strategy,
        rrf_k=config.retrieval.rrf_k,
        # device left as default (None)
    )

    print(f"[INFO] Loading Spider split from: {split_path}")
    examples = load_spider_split(split_path, num_examples=num_examples, seed=seed)
    print(f"[INFO] Evaluating on {len(examples)} examples (top_k={top_k})")

    total_gold_tables = 0
    total_retrieved_tables = 0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    full_recall_count = 0
    counted_examples = 0

    for ex in examples:
        question = ex["question"]
        db_id = ex["db_id"]
        gold_sql_str = ex.get("query") or ex.get("sql_query") or ""

        if db_id not in db_map:
            # Shouldn't happen if tables.json and split are consistent
            continue

        db_schema = db_map[db_id]

        # 1) Primary: extract gold tables from Spider's SQL AST
        gold_tables = extract_gold_tables_from_ast(ex, db_schema)
        if not gold_tables:
            # 2) Fallback: heuristic from SQL text
            gold_tables = extract_gold_tables_from_text(gold_sql_str, db_schema)
            if not gold_tables:
                # If we still can't find any gold tables, skip this example
                # to avoid 0/0 recall. This should be rare.
                continue

        hits = retriever.search(
            query=question,
            top_k=top_k,
            limit_to_db=db_id,
        )
        retrieved_tables = {hit.table.name for hit in hits}

        intersection = gold_tables & retrieved_tables

        # Recall: fraction of gold tables retrieved
        recall = len(intersection) / len(gold_tables)

        # Precision: fraction of retrieved tables that are gold
        if len(retrieved_tables) > 0:
            precision = len(intersection) / len(retrieved_tables)
        else:
            precision = 0.0

        # F1 score per example
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        total_gold_tables += len(gold_tables)
        total_retrieved_tables += len(retrieved_tables)
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        counted_examples += 1

        if recall == 1.0:
            full_recall_count += 1

    if counted_examples == 0:
        print("[WARN] No examples with detected gold tables; check gold table extraction.")
        return

    avg_gold_tables = total_gold_tables / counted_examples
    avg_retrieved_tables = total_retrieved_tables / counted_examples
    avg_precision = precision_sum / counted_examples
    avg_recall = recall_sum / counted_examples
    avg_f1 = f1_sum / counted_examples
    full_recall_rate = full_recall_count / counted_examples

    print("\n========== Table Retrieval Metrics ==========")
    print(f"Average # gold tables:         {avg_gold_tables:.3f}")
    print(f"Average # retrieved tables:    {avg_retrieved_tables:.3f}")
    print(f"Average F1:                    {avg_f1:.3f}")
    print(f"Average precision:             {avg_precision:.3f}")
    print(f"Average recall:                {avg_recall:.3f}")
    print(f"Full recall rate (recall=1.0): {full_recall_rate:.3f}")
    print("=============================================")


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate table retrieval on Spider.")
    parser.add_argument(
        "--split-path",
        type=str,
        default=None,
        help=(
            "Path to Spider JSON split (e.g., dev.json or train_spider.json). "
            "Defaults to <spider_data_dir>/dev.json if not provided."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="bm25",
        choices=["bm25", "dense", "hybrid"],
        help="Retrieval mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of tables to retrieve per query (before any FK expansion).",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to evaluate (subset). Use -1 for all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling.",
    )

    args = parser.parse_args()

    config = DEFAULT_CONFIG
    spider_paths = config.spider_paths

    if args.split_path is not None:
        split_path = Path(args.split_path)
    else:
        # Default to dev.json under the Spider data dir
        split_path = spider_paths.data_dir / "dev.json"

    num_examples = None if args.num_examples == -1 else args.num_examples

    evaluate_retriever(
        split_path=split_path,
        mode=args.mode,
        top_k=args.top_k,
        num_examples=num_examples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
