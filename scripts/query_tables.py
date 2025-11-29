#!/usr/bin/env python

"""Retrieve SPIDER tables by natural language query and print their schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.table_retrieval.schemas import flatten_tables, load_spider_schemas
from src.table_retrieval.retriever import TableRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        "-q",
        required=True,
        help="Natural language description of the tables you need (e.g., 'flights and airports').",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="Number of tables to return (defaults to config.retrieval.top_k).",
    )
    parser.add_argument(
        "--mode",
        choices=["bm25", "dense", "hybrid"],
        default="bm25",
        help="Retrieval mode: bm25 (exact-match), dense (BGE), or hybrid (fused).",
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default=None,
        help="Dense encoder model name (defaults to config.retrieval.dense_model_name).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for dense retrieval (e.g., cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--top-k-sparse",
        type=int,
        default=None,
        help="Candidate pool from BM25 when using hybrid (defaults to config).",
    )
    parser.add_argument(
        "--top-k-dense",
        type=int,
        default=None,
        help="Candidate pool from dense when using hybrid (defaults to config).",
    )
    parser.add_argument(
        "--hybrid-weight-sparse",
        type=float,
        default=None,
        help="Sparse weight for weighted hybrid fusion (defaults to config).",
    )
    parser.add_argument(
        "--hybrid-weight-dense",
        type=float,
        default=None,
        help="Dense weight for weighted hybrid fusion (defaults to config).",
    )
    parser.add_argument(
        "--hybrid-strategy",
        choices=["weighted", "rrf"],
        default=None,
        help="Hybrid fusion strategy: weighted (default) or rrf.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=None,
        help="RRF k parameter when hybrid strategy is rrf (defaults to config).",
    )
    parser.add_argument(
        "--db",
        "-d",
        type=str,
        default=None,
        help="Optional SPIDER database id to restrict retrieval.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (defaults to current working directory).",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    config = PipelineConfig.from_root(args.root)
    top_k = args.top_k or config.retrieval.top_k
    dense_model = args.dense_model or config.retrieval.dense_model_name
    top_k_sparse = args.top_k_sparse or config.retrieval.top_k_sparse
    top_k_dense = args.top_k_dense or config.retrieval.top_k_dense
    hybrid_weight_sparse = args.hybrid_weight_sparse
    if hybrid_weight_sparse is None:
        hybrid_weight_sparse = config.retrieval.hybrid_weight_sparse
    hybrid_weight_dense = args.hybrid_weight_dense
    if hybrid_weight_dense is None:
        hybrid_weight_dense = config.retrieval.hybrid_weight_dense
    hybrid_strategy = args.hybrid_strategy or config.retrieval.hybrid_strategy
    rrf_k = args.rrf_k or config.retrieval.rrf_k

    databases = load_spider_schemas(config.spider_paths.tables_json)
    if args.db:
        databases = [db for db in databases if db.db_id == args.db]
        if not databases:
            raise SystemExit(f"No database '{args.db}' found in {config.spider_paths.tables_json}")

    tables = flatten_tables(databases)
    if not tables:
        raise SystemExit("No tables available to index.")

    retriever = TableRetriever(
        tables,
        mode=args.mode,
        dense_model_name=dense_model,
        top_k_sparse=top_k_sparse,
        top_k_dense=top_k_dense,
        hybrid_weight_sparse=hybrid_weight_sparse,
        hybrid_weight_dense=hybrid_weight_dense,
        hybrid_strategy=hybrid_strategy,
        rrf_k=rrf_k,
        device=args.device,
    )
    hits = retriever.search(args.query, top_k=top_k, limit_to_db=args.db)

    all_db_ids = {hit.table.database for hit in hits}
    candidate_tables = []
    for hit in hits:
        candidate_tables.append(
            {
                "db_id": hit.table.database,
                "table_name": hit.table.name,
                "columns": [[col.name, col.type] for col in hit.table.columns],
                "score": hit.score,
            }
        )

    payload: dict[str, object] = {
        "question": args.query,
        "mode": args.mode,
        "candidate_tables": candidate_tables,
    }
    if len(all_db_ids) == 1:
        payload["db_id"] = next(iter(all_db_ids))

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys

    try:
        main(args=parse_args())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
