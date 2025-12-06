from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]   
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.schema_to_prompt import SpiderSchema, build_prompt_for_entry
from src.sql_gbnf import (
    build_sqlite_prefix_allowed_tokens_fn,
    load_sqlite_grammar,
)
from src.table_retrieval.schemas import (
    flatten_tables,
    load_spider_schemas,
)
from src.table_retrieval.retriever import TableRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate end-to-end pipeline: question -> table retrieval -> SQL generation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pipeline_eval.jsonl"),
        help="Path to write JSONL results (appends if exists).",
    )
    parser.add_argument(
        "--mode",
        choices=["bm25", "dense", "hybrid"],
        default="bm25",
        help="Table retrieval mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of tables to retrieve per question.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for SQL generation.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="defog/sqlcoder-7b-2",
        help="HuggingFace model id for SQL generation.",
    )
    parser.add_argument(
        "--no-gbnf",
        action="store_true",
        help="Disable GBNF grammar enforcement.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of examples to process.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root (defaults to repo root).",
    )
    return parser.parse_args()


def load_dataset(entries_paths: Iterable[Path]) -> list[dict]:
    records: list[dict] = []
    for path in entries_paths:
        if not path.is_file():
            print(f"[WARN] Missing dataset file: {path}")
            continue
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        for ex in data:
            records.append(
                {
                    "db_id": ex["db_id"],
                    "question": ex["question"],
                    "gold_sql": ex["query"],
                }
            )
    return records


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
    except Exception as exc:  # pragma: no cover - runtime safety
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


def main(args: argparse.Namespace) -> None:
    config = PipelineConfig.from_root(args.root)
    spider_schema = SpiderSchema(TABLES_JSON_PATH)
    datasets = [
        DATA_ROOT / "train_spider.json",
        DATA_ROOT / "train_others.json",
        DATA_ROOT / "dev.json",
    ]
    entries = load_dataset(datasets)
    if args.limit:
        entries = entries[: args.limit]
    total_examples = len(entries)
    if total_examples == 0:
        raise SystemExit("No examples found.")

    databases = load_spider_schemas(TABLES_JSON_PATH)
    tables = flatten_tables(databases)
    retriever = TableRetriever(tables, mode=args.mode, dense_model_name=config.retrieval.dense_model_name)

    tokenizer, model = load_model_and_tokenizer(args.model_name)
    prefix_fn = None
    if not args.no_gbnf:
        grammar_text = load_sqlite_grammar()
        prefix_fn = build_sqlite_prefix_allowed_tokens_fn(tokenizer, grammar_text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    outfile = args.output.open("a", encoding="utf-8")

    start_time = time.time()
    pbar = tqdm(total=total_examples, desc="Evaluating", ncols=100)

    for idx, ex in enumerate(entries):
        t_total_start = time.time()
        record: Dict[str, object] = {
            "idx": idx,
            "db_id": ex["db_id"],
            "question": ex["question"],
            "gold_sql": ex["gold_sql"],
            "retrieval_mode": args.mode,
        }
        try:
            t_ret = time.time()
            hits = retriever.search(ex["question"], top_k=args.top_k, limit_to_db=ex["db_id"])
            record["retrieval_time"] = time.time() - t_ret
            record["retrieved_tables"] = [hit.table.name for hit in hits]
            candidate_tables = build_candidate_tables(hits)

            entry = {
                "db_id": ex["db_id"],
                "question": ex["question"],
                "candidate_tables": candidate_tables,
            }
            prompt = build_prompt_for_entry(entry, spider_schema)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            t_gen = time.time()
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                    "num_beams": 4,
                    "early_stopping": True,
                }
                if prefix_fn is not None:
                    gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
                output_ids = model.generate(**inputs, **gen_kwargs)
            record["generation_time"] = time.time() - t_gen
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            pred_sql = full_text.split("SQL:", 1)[1].strip() if "SQL:" in full_text else full_text.strip()
            record["pred_sql"] = pred_sql

            em, exec_match = evaluate_pair(
                DATABASE_DIR,
                ex["db_id"],
                ex["gold_sql"],
                pred_sql,
            )
            record["em"] = em
            record["exec_match"] = exec_match
        except Exception as exc:  # pragma: no cover - safety
            record["error"] = repr(exc)

        record["total_time"] = time.time() - t_total_start
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        outfile.flush()
        os.fsync(outfile.fileno())
        pbar.update(1)

    pbar.close()
    total_runtime = time.time() - start_time
    outfile.write(json.dumps({"summary": {"total_examples": total_examples, "runtime_sec": total_runtime}}) + "\n")
    outfile.flush()
    os.fsync(outfile.fileno())
    outfile.close()
    print(f"[DONE] Wrote results to {args.output} in {total_runtime:.2f}s")


if __name__ == "__main__":
    main(parse_args())
