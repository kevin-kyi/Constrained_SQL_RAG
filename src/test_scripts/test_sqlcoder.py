# **********************************
# NON QUANTIZED TEST SCRIPT
# **********************************

"""
test_sqlcoder.py

Smoke test for:
- schema_to_prompt pipeline
- SQLCoder generation
- Side-by-side comparison to gold SQL
- Simple evaluation: string-based EM + execution-based match

Run from repo root:
    python -m src.test_script.test_sqlcoder
or:
    cd Constrained_SQL_RAG
    python src/test_script/test_sqlcoder.py
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# 
# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]  # Constrained_SQL_RAG/
SRC_ROOT = REPO_ROOT / "src"
sys.path.append(str(SRC_ROOT))

from schema_to_prompt import (  # type: ignore
    SpiderSchema,
    load_schema_entries,
    TABLES_JSON,
)

SPIDER_DATA_ROOT = REPO_ROOT / "spider_dataset" / "spider_data"
TRAIN_JSON = SPIDER_DATA_ROOT / "train_spider.json"
DEV_JSON = SPIDER_DATA_ROOT / "dev.json"
DB_DIR = SPIDER_DATA_ROOT / "database"

MODEL_NAME = "defog/sqlcoder-7b-2"  # change if you use a different SQLCoder variant


# ---------------------------------------------------------------------------
# Gold SQL loading and lookup
# ---------------------------------------------------------------------------

def load_gold_map() -> Dict[Tuple[str, str], str]:
    """
    Load gold SQL from train_spider.json and dev.json.
    Returns a dict mapping (db_id, question) -> gold_query.
    """
    gold_map: Dict[Tuple[str, str], str] = {}

    for json_path in [TRAIN_JSON, DEV_JSON]:
        if not json_path.is_file():
            print(f"[WARN] Gold file not found: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ex in data:
            key = (ex["db_id"], ex["question"])
            # if duplicates exist, keep the first
            if key not in gold_map:
                gold_map[key] = ex["query"]

    print(f"[INFO] Loaded {len(gold_map)} gold (db_id, question) pairs.")
    return gold_map


# ---------------------------------------------------------------------------
# SQLCoder loading + generation
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    print(f"[INFO] Loading SQLCoder model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )
    return tokenizer, model


def generate_sql(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """
    Run a single prompt through SQLCoder and return the full generated text.
    Currently uses beam search, deterministic.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text


def extract_sql_from_output(full_text: str) -> str:
    """
    Given the model's full text, try to extract the SQL following the 'SQL:' marker.
    This is heuristic but good enough for debugging.
    """
    marker = "SQL:"
    if marker in full_text:
        sql_part = full_text.split(marker, 1)[1]
    else:
        sql_part = full_text

    # Strip leading/trailing whitespace and any obvious natural language trailing commentary.
    # For now we just strip whitespace.
    return sql_part.strip()


# ---------------------------------------------------------------------------
# Simple evaluation: EM + execution match
# ---------------------------------------------------------------------------

def normalize_sql(sql: str) -> str:
    """
    Very simple normalization:
    - strip leading/trailing whitespace
    - drop trailing semicolon
    - collapse whitespace to single spaces
    - lowercase
    This is NOT the full Spider evaluation, but useful for quick EM checks.
    """
    s = sql.strip()
    if s.endswith(";"):
        s = s[:-1]
    # Collapse whitespace
    s = " ".join(s.split())
    s = s.lower()
    return s


def execute_query(db_path: Path, sql: str) -> Optional[List[Tuple]]:
    """
    Execute SQL against a sqlite database and return the result rows.

    Returns:
        list of tuples on success
        None if an error occurs
    """
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"[EXEC ERROR] {e}")
        return None


def evaluate_pair(db_id: str, gold_sql: str, pred_sql: str) -> Tuple[bool, Optional[bool]]:
    """
    Compute:
      - em: bool, normalized string equality
      - exec_match: bool or None (None if either query fails to execute)

    Returns: (em, exec_match)
    """
    norm_gold = normalize_sql(gold_sql)
    norm_pred = normalize_sql(pred_sql)
    em = (norm_gold == norm_pred)

    # Execution match
    db_path = DB_DIR / db_id / f"{db_id}.sqlite"
    if not db_path.is_file():
        print(f"[WARN] DB file not found for db_id={db_id}: {db_path}")
        return em, None

    gold_rows = execute_query(db_path, gold_sql)
    pred_rows = execute_query(db_path, pred_sql)

    if gold_rows is None or pred_rows is None:
        exec_match = None
    else:
        exec_match = (gold_rows == pred_rows)

    return em, exec_match


# ---------------------------------------------------------------------------
# Main test driver
# ---------------------------------------------------------------------------

def main():
    # Load Spider schema + schema.json entries from your retrieval pipeline
    spider_schema = SpiderSchema(TABLES_JSON)
    entries = load_schema_entries()
    print(f"[INFO] Loaded {len(entries)} entries from spider_dataset/schema.json")

    # Load gold map
    gold_map = load_gold_map()

    # Load SQLCoder
    tokenizer, model = load_model_and_tokenizer()

    # Import prompt builder
    from schema_to_prompt import build_prompt_for_entry  # type: ignore

    num_examples = min(5, len(entries))
    print(f"[INFO] Running SQLCoder + eval on {num_examples} example(s) ...")

    for i, entry in enumerate(entries[:num_examples], 1):
        db_id = entry["db_id"]
        question = entry["question"]

        prompt = build_prompt_for_entry(entry, spider_schema)
        print(f"\n================= EXAMPLE {i} =================")
        print(f"DB ID: {db_id}")
        print(f"Question: {question}")

        # Look up gold SQL
        gold_sql = gold_map.get((db_id, question))
        if gold_sql is None:
            print("[WARN] No gold SQL found for this (db_id, question) pair in train/dev.")
        else:
            print("\n--- Gold SQL ---")
            print(gold_sql)

        print("\n--- Prompt ---")
        print(prompt)
        print("--------------")

        # Generate SQL
        generated_full = generate_sql(model, tokenizer, prompt)
        pred_sql = extract_sql_from_output(generated_full)

        print("\n--- Generated full text ---")
        print(generated_full)
        print("\n--- Extracted Predicted SQL ---")
        print(pred_sql)

        # Evaluate
        if gold_sql is not None:
            em, exec_match = evaluate_pair(db_id, gold_sql, pred_sql)
            print("\n--- Evaluation ---")
            print(f"Exact Match (normalized): {em}")
            if exec_match is None:
                print("Execution Match: N/A (execution error)")
            else:
                print(f"Execution Match (result equality): {exec_match}")

        print("==============================================")


if __name__ == "__main__":
    main()
