from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from collections import Counter
import sys

import torch

if sys.version_info >= (3, 13):
    def _noop_compile(fn=None, *args, **kwargs):
        # Used as @torch.compile or torch.compile(fn, ...)
        if fn is not None and callable(fn):
            return fn
        def decorator(inner_fn):
            return inner_fn
        return decorator
    torch.compile = _noop_compile

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel  

# --------------------------------------------------------------------
# Path setup
# --------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROMPTS_PATH = PROJECT_ROOT / "prompts.jsonl"
QUESTIONS_PATH = PROJECT_ROOT / "questions_used.jsonl"
SPIDER_DB_ROOT = PROJECT_ROOT / "spider_dataset" / "spider_data" / "database"
RESULTS_PATH = PROJECT_ROOT / "sqlcoder_lora_results.jsonl"


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SQLCoder + LoRA on Spider prompts.jsonl."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="defog/sqlcoder-7b-2",
        help="Base SQLCoder model name.",
    )
    parser.add_argument(
        "--lora_path",
        type=Path,
        required=True,
        help="Path to LoRA checkpoint directory (e.g. outputs/sqlcoder-lora-spider/checkpoint-800).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=160,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Device for inference (e.g. "cuda:0" or "auto").',
    )
    return parser.parse_args()


# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_sql(generated_text: str) -> str:
    """
    Extract SQL from a SQLCoder-style generation.

    Priority:
      1. Text inside ```sql ... ```
      2. Last fenced block after ``` (any)
      3. Last 'select ' chunk in the text
      4. Entire text stripped
    """
    lower = generated_text.lower()

    # 1) ```sql fenced block
    if "```sql" in lower:
        try:
            _, after = generated_text.split("```sql", 1)
            sql_body = after.split("```", 1)[0]
            return sql_body.strip()
        except Exception:
            pass

    # 2) Any fenced block
    if "```" in generated_text:
        try:
            parts = generated_text.split("```")
            return parts[-1].strip()
        except Exception:
            pass

    # 3) Try from last 'select '
    idx = lower.rfind("select ")
    if idx != -1:
        return generated_text[idx:].strip()

    # 4) Fallback: whole thing
    return generated_text.strip()


def execute_sql(db_id: str, sql: str) -> Dict[str, Any]:
    """
    Execute SQL against the correct Spider SQLite database.
    Returns a dict with either rows or an error.
    """
    db_path = SPIDER_DB_ROOT / db_id / f"{db_id}.sqlite"

    if not db_path.exists():
        return {"ok": False, "error": f"DB file not found at {db_path}"}

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return {"ok": True, "rows": rows}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# --------------------------------------------------------------------
# Execution metrics: strict, lenient, Jaccard
# --------------------------------------------------------------------
def _normalize_rows(rows: List[Any]) -> List[Tuple[Any, ...]]:
    return [tuple(r) for r in rows]


def exec_match_strict(gold: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    if not gold.get("ok") or not pred.get("ok"):
        return False
    return gold["rows"] == pred["rows"]


def exec_match_lenient(gold: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    if not gold.get("ok") or not pred.get("ok"):
        return False

    g_rows = gold["rows"]
    p_rows = pred["rows"]

    # 1) strict equality
    if g_rows == p_rows:
        return True

    g_norm = _normalize_rows(g_rows)
    p_norm = _normalize_rows(p_rows)

    g_set = set(g_norm)
    p_set = set(p_norm)

    # 2) sets equal
    if g_set == p_set:
        return True

    # 3) single-row aggregate: ignore column order
    if len(g_rows) == 1 and len(p_rows) == 1:
        from collections import Counter

        g_counter = Counter(g_rows[0])
        p_counter = Counter(p_rows[0])
        if g_counter == p_counter:
            return True

    # 4) gold subset of pred
    if g_set.issubset(p_set):
        return True

    return False


def row_jaccard(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    if not gold.get("ok") or not pred.get("ok"):
        return 0.0

    g_norm = _normalize_rows(gold["rows"])
    p_norm = _normalize_rows(pred["rows"])

    g_set = set(g_norm)
    p_set = set(p_norm)

    if not g_set and not p_set:
        return 0.0

    union = g_set | p_set
    inter = g_set & p_set

    if not union:
        return 0.0

    return len(inter) / len(union)


def diff_sql(gold_sql: str, pred_sql: str) -> str:
    diff_lines = unified_diff(
        gold_sql.splitlines(),
        pred_sql.splitlines(),
        fromfile="gold_sql",
        tofile="pred_sql",
        lineterm="",
    )
    return "\n".join(diff_lines)


# --------------------------------------------------------------------
# SQLCoder + LoRA wrapper
# --------------------------------------------------------------------
@dataclass
class SQLCoderLoraWrapper:
    model_name: str
    lora_path: Path
    max_new_tokens: int = 160
    device: str = "cuda:0"  

    def __post_init__(self) -> None:
        print(f"[INFO] Loading base SQLCoder model: {self.model_name}")
        print(f"[INFO] Applying LoRA adapter from: {self.lora_path}")
        print(f"[INFO] Requested device: {self.device} (used for logging only; model uses device_map='auto')")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # fp16 inference is fine here
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Use accelerate-style loading with device_map="auto"
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",        
            low_cpu_mem_usage=True,
        )

        # Attach LoRA adapter on top of base model
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(
            self.base_model,
            str(self.lora_path),
        )
        self.model.eval()

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            num_beams=4,
            early_stopping=True,
        )

    def generate_sql(self, prompt: str) -> str:
        out = self.pipe(prompt)[0]["generated_text"]
        return extract_sql(out)


# --------------------------------------------------------------------
# Main evaluation loop
# --------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"prompts.jsonl not found at {PROMPTS_PATH}")
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"questions_used.jsonl not found at {QUESTIONS_PATH}")
    if not args.lora_path.exists():
        raise FileNotFoundError(f"LoRA path not found at {args.lora_path}")

    prompts = list(load_jsonl(PROMPTS_PATH))
    questions = {row["question_id"]: row for row in load_jsonl(QUESTIONS_PATH)}

    print(f"[INFO] Loaded {len(prompts)} prompts")
    print(f"[INFO] Loaded {len(questions)} gold questions")

    sqlcoder = SQLCoderLoraWrapper(
        model_name=args.model_name,
        lora_path=args.lora_path,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    total = 0
    strict_correct = 0
    lenient_correct = 0
    sql_success = 0
    jaccard_sum = 0.0

    with RESULTS_PATH.open("w", encoding="utf-8") as fout:
        for entry in prompts:
            total += 1
            qid = entry["question_id"]
            db_id = entry["db_id"]
            question = entry["question"]
            prompt = entry["prompt"]

            gold_record = questions.get(qid)
            if gold_record is None:
                print(f"[WARN] Missing gold record for question_id={qid}, skipping.")
                continue

            gold_sql = gold_record["gold_sql"]

            # 1. Generate SQL
            pred_sql = sqlcoder.generate_sql(prompt)

            # 2. Execute gold + predicted SQL
            gold_exec = execute_sql(db_id, gold_sql)
            pred_exec = execute_sql(db_id, pred_sql)

            if pred_exec.get("ok"):
                sql_success += 1

            # 3. Compute metrics
            strict = exec_match_strict(gold_exec, pred_exec)
            loose = exec_match_lenient(gold_exec, pred_exec)
            jacc = row_jaccard(gold_exec, pred_exec)
            jaccard_sum += jacc

            if strict:
                strict_correct += 1
            if loose:
                lenient_correct += 1

            # 4. Print summary
            print("\n" + "=" * 80)
            print(f"QUESTION ID: {qid}")
            print(f"DB ID      : {db_id}")
            print(f"QUESTION   : {question}")
            print("-" * 80)
            print("GOLD SQL:")
            print(gold_sql)
            print("\nPRED SQL (LoRA):")
            print(pred_sql)
            print("\nSQL DIFF (gold vs pred):")
            print(diff_sql(gold_sql, pred_sql) or "(no diff)")
            print("\nGOLD EXECUTION RESULT:")
            print(gold_exec)
            print("\nPRED EXECUTION RESULT:")
            print(pred_exec)
            print(f"\nSTRICT EXEC MATCH?   {'✅ YES' if strict else '❌ NO'}")
            print(f"LENIENT EXEC MATCH?  {'✅ YES' if loose else '❌ NO'}")
            print(f"ROW JACCARD SIMILARITY: {jacc:.3f}")
            print("=" * 80)

            # 5. Log to results file
            fout.write(json.dumps({
                "question_id": qid,
                "db_id": db_id,
                "question": question,
                "prompt": prompt,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "gold_exec": gold_exec,
                "pred_exec": pred_exec,
                "strict_exec_match": strict,
                "lenient_exec_match": loose,
                "row_jaccard": jacc,
            }) + "\n")

    # ----------------------------------------------------------------
    # Final metrics
    # ----------------------------------------------------------------
    strict_acc = strict_correct / total if total > 0 else 0.0
    lenient_acc = lenient_correct / total if total > 0 else 0.0
    exec_ok_rate = sql_success / total if total > 0 else 0.0
    avg_jaccard = jaccard_sum / total if total > 0 else 0.0

    print("\n" + "#" * 80)
    print("# SQLCoder + LoRA Evaluation Summary")
    print("#" * 80)
    print(f"Total examples                  : {total}")
    print(f"Pred SQL exec succeeded         : {sql_success} ({exec_ok_rate:.3f})")
    print(f"STRICT exec matches gold        : {strict_correct} ({strict_acc:.3f})")
    print(f"LENIENT exec matches gold       : {lenient_correct} ({lenient_acc:.3f})")
    print(f"Average row Jaccard similarity  : {avg_jaccard:.3f}")
    print(f"Results saved to                : {RESULTS_PATH}")
    print("#" * 80)


if __name__ == "__main__":
    main()
