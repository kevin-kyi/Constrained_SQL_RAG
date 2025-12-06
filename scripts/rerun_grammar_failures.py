#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "spider_dataset" / "spider_data"
TABLES_JSON_PATH = DATA_ROOT / "tables.json"
DATABASE_DIR = DATA_ROOT / "database"

from src.schema_to_prompt import SpiderSchema, build_prompt_for_entry  # type: ignore
from src.sql_gbnf import build_sqlite_prefix_allowed_tokens_fn, load_sqlite_grammar  # type: ignore
from src.table_retrieval.schemas import flatten_tables, load_spider_schemas  # type: ignore
from src.table_retrieval.retriever import TableRetriever  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify grammar-related failures from a previous eval JSONL and rerun them.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/pipeline_eval.jsonl"),
        help="Existing eval JSONL to scan for grammar failures.",
    )
    parser.add_argument(
        "--indices-out",
        type=Path,
        default=Path("results/grammar_failure_indices.txt"),
        help="File to write failing indices (one per line).",
    )
    parser.add_argument(
        "--rerun-out",
        type=Path,
        default=Path("results/pipeline_eval_grammar_rerun.jsonl"),
        help="Where to write rerun JSONL results.",
    )
    parser.add_argument(
        "--mode",
        choices=["bm25", "dense", "hybrid"],
        default="bm25",
        help="Retrieval mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of tables to retrieve per question.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="defog/sqlcoder-7b-2",
        help="HF model id for SQL generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--no-gbnf",
        action="store_true",
        help="Disable grammar filtering when rerunning.",
    )
    return parser.parse_args()


def load_pipeline_results(path: Path) -> list[dict]:
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "summary" in obj:
                continue
            entries.append(obj)
    return entries


def is_grammar_exception(exc: Exception) -> bool:
    msg = str(exc).lower()
    keywords = ["syntax error", "incomplete input", "unrecognized token", "unterminated"]
    return any(k in msg for k in keywords)


def find_grammar_failures(entries: Iterable[dict]) -> list[dict]:
    failures: list[dict] = []
    for obj in entries:
        db_id = obj.get("db_id")
        pred_sql = obj.get("pred_sql")
        if not db_id or not pred_sql:
            continue
        db_path = DATABASE_DIR / db_id / f"{db_id}.sqlite"
        if not db_path.is_file():
            continue
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(pred_sql).fetchall()
        except Exception as exc:
            if is_grammar_exception(exc):
                failures.append(obj)
    return failures


def write_indices(indices: List[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for idx in indices:
            fp.write(f"{idx}\n")


def normalize_sql(sql: str) -> str:
    s = sql.strip()
    if s.endswith(";"):
        s = s[:-1]
    s = " ".join(s.split())
    return s.lower()


def execute_query(db_path: Path, sql: str) -> Optional[List[Tuple]]:
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return None


def evaluate_pair(db_dir: Path, db_id: str, gold_sql: str, pred_sql: str) -> Tuple[bool, Optional[bool]]:
    norm_gold = normalize_sql(gold_sql)
    norm_pred = normalize_sql(pred_sql)
    em = norm_gold == norm_pred

    db_path = db_dir / db_id / f"{db_id}.sqlite"
    if not db_path.is_file():
        return em, None

    gold_rows = execute_query(db_path, gold_sql)
    pred_rows = execute_query(db_path, pred_sql)
    if gold_rows is None or pred_rows is None:
        return em, None
    return em, gold_rows == pred_rows


def build_candidate_tables(hits) -> list[dict]:
    candidates: list[dict] = []
    for hit in hits:
        candidates.append(
            {
                "table_name": hit.table.name,
                "columns": [[col.name, col.type] for col in hit.table.columns],
                "score": hit.score,
            }
        )
    return candidates


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    return tokenizer, model


def rerun(entries: list[dict], args: argparse.Namespace) -> None:
    databases = load_spider_schemas(TABLES_JSON_PATH)
    tables = flatten_tables(databases)
    retriever = TableRetriever(tables, mode=args.mode)
    spider_schema = SpiderSchema(TABLES_JSON_PATH)

    tokenizer, model = load_model_and_tokenizer(args.model_name)
    prefix_fn = None
    if not args.no_gbnf:
        grammar_text = load_sqlite_grammar()
        prefix_fn = build_sqlite_prefix_allowed_tokens_fn(tokenizer, grammar_text)

    args.rerun_out.parent.mkdir(parents=True, exist_ok=True)
    outfile = args.rerun_out.open("a", encoding="utf-8")

    start_time = time.time()
    pbar = tqdm(total=len(entries), desc="Rerunning grammar failures", ncols=100)

    for ex in entries:
        t_start = time.time()
        record: Dict[str, object] = {
            "idx": ex["idx"],
            "db_id": ex["db_id"],
            "question": ex["question"],
            "gold_sql": ex["gold_sql"],
            "retrieval_mode": args.mode,
        }
        try:
            hits = retriever.search(ex["question"], top_k=args.top_k, limit_to_db=ex["db_id"])
            record["retrieval_time"] = 0.0  # retrieval already done previously; negligible for subset
            candidate_tables = build_candidate_tables(hits)
            prompt = build_prompt_for_entry(
                {
                    "db_id": ex["db_id"],
                    "question": ex["question"],
                    "candidate_tables": candidate_tables,
                },
                spider_schema,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "num_beams": 4,
                "early_stopping": True,
            }
            if prefix_fn is not None:
                gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
            t_gen = time.time()
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)
            record["generation_time"] = time.time() - t_gen
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            pred_sql = full_text.split("SQL:", 1)[1].strip() if "SQL:" in full_text else full_text.strip()
            record["pred_sql"] = pred_sql

            em, exec_match = evaluate_pair(DATABASE_DIR, ex["db_id"], ex["gold_sql"], pred_sql)
            record["em"] = em
            record["exec_match"] = exec_match
        except Exception as exc:  # pragma: no cover
            record["error"] = repr(exc)

        record["total_time"] = time.time() - t_start
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        outfile.flush()
        os.fsync(outfile.fileno())
        pbar.update(1)

    pbar.close()
    outfile.write(
        json.dumps(
            {
                "summary": {
                    "count": len(entries),
                    "runtime_sec": time.time() - start_time,
                }
            }
        )
        + "\n"
    )
    outfile.flush()
    os.fsync(outfile.fileno())
    outfile.close()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input results file not found: {args.input}")

    entries = load_pipeline_results(args.input)
    grammar_failures = find_grammar_failures(entries)
    indices = sorted([obj["idx"] for obj in grammar_failures])
    write_indices(indices, args.indices_out)
    print(f"[INFO] Found {len(indices)} grammar failures. Indices written to {args.indices_out}")

    if grammar_failures:
        rerun(grammar_failures, args)
        print(f"[DONE] Reran grammar failures. Output: {args.rerun_out}")


if __name__ == "__main__":
    main()
