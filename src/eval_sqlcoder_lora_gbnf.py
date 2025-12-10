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

# --- Python 3.13 + bitsandbytes hotfix ---------------------------------
# bitsandbytes uses @torch.compile, which is not supported on Python 3.13.
# We override torch.compile to be a no-op decorator BEFORE peft/bnb imports it.
if sys.version_info >= (3, 13):
    def _noop_compile(fn=None, *args, **kwargs):
        # Used as @torch.compile or torch.compile(fn, ...)
        if fn is not None and callable(fn):
            return fn

        def decorator(inner_fn):
            return inner_fn

        return decorator

    torch.compile = _noop_compile
# -----------------------------------------------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
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
RESULTS_PATH = PROJECT_ROOT / "sqlcoder_gbnf_lora_results.jsonl"

SQLCODER_MODEL_NAME = "defog/sqlcoder-7b-2"

# GBNF helper
from src.sql_gbnf import build_sqlite_prefix_allowed_tokens_fn


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
    Extract SQL from the model completion (we already slice off the prompt).
    Here we just handle fenced ```sql blocks if they appear, otherwise return
    the raw completion.
    """
    text = generated_text.strip()
    lower = text.lower()

    # Prefer ```sql fenced block if present
    if "```sql" in lower:
        try:
            _, after = text.split("```sql", 1)
            sql_body = after.split("```", 1)[0]
            return sql_body.strip()
        except Exception:
            pass

    # Any generic ``` fenced block
    if "```" in text:
        try:
            parts = text.split("```")
            return parts[-1].strip()
        except Exception:
            pass

    # Fallback: whole completion
    return text


def execute_sql(db_id: str, sql: str) -> Dict[str, Any]:
    db_path = SPIDER_DB_ROOT / db_id / f"{db_id}.sqlite"

    if not db_path.exists():
        return {"ok": False, "error": f"DB not found at {db_path}"}

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
# Execution metrics
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

    # strict equals
    if g_rows == p_rows:
        return True

    g_norm = _normalize_rows(g_rows)
    p_norm = _normalize_rows(p_rows)

    g_set = set(g_norm)
    p_set = set(p_norm)

    # set-equal
    if g_set == p_set:
        return True

    # SINGLE ROW aggregate (ignore column order)
    if len(g_rows) == 1 and len(p_rows) == 1:
        if Counter(g_rows[0]) == Counter(p_rows[0]):
            return True

    # subset test
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
        lineterm=""
    )
    return "\n".join(diff_lines)


# --------------------------------------------------------------------
# SQLCoder + LoRA + GBNF wrapper
# --------------------------------------------------------------------
@dataclass
class SQLCoderGBNFLoraWrapper:
    model_name: str
    lora_path: Path

    def __post_init__(self) -> None:
        print(f"[INFO] Loading base SQLCoder model: {self.model_name}")
        print(f"[INFO] Applying LoRA adapter from: {self.lora_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Base model (single GPU via device_map="cuda:0" is fine here)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )

        # Attach LoRA adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            str(self.lora_path),
        )
        self.model.eval()

        # Build GBNF grammar callback from tokenizer (vocabulary unchanged by LoRA)
        self.prefix_allowed_tokens_fn = build_sqlite_prefix_allowed_tokens_fn(
            self.tokenizer
        )

    def generate_sql(self, prompt: str) -> str:
        # Standard constrained generate, but now the model is base + LoRA
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                temperature=0.0,
                early_stopping=False,
                use_cache=True,
                prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
            )

        # Only decode the completion, not the prompt
        completion_ids = gen_ids[0][input_len:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return extract_sql(text)


# --------------------------------------------------------------------
# Main evaluation loop
# --------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SQLCoder + LoRA + GBNF on Spider prompts.jsonl"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=SQLCODER_MODEL_NAME,
        help="Base model name or path (default: defog/sqlcoder-7b-2)",
    )
    parser.add_argument(
        "--lora_path",
        type=Path,
        required=True,
        help="Path to LoRA checkpoint (e.g., outputs/extra_copy/checkpoint-800)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"prompts.jsonl missing: {PROMPTS_PATH}")
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"questions_used.jsonl missing: {QUESTIONS_PATH}")

    prompts = list(load_jsonl(PROMPTS_PATH))
    questions = {row["question_id"]: row for row in load_jsonl(QUESTIONS_PATH)}

    print(f"[INFO] Loaded {len(prompts)} prompts")
    print(f"[INFO] Loaded {len(questions)} gold questions")

    sqlcoder = SQLCoderGBNFLoraWrapper(
        model_name=args.model_name,
        lora_path=args.lora_path,
    )

    total = 0
    strict_correct = 0
    lenient_correct = 0
    sql_success = 0
    jaccard_sum = 0.0

    syntax_error_count = 0
    runtime_error_count = 0

    with RESULTS_PATH.open("w", encoding="utf-8") as fout:
        for entry in prompts:
            total += 1
            qid = entry["question_id"]
            db_id = entry["db_id"]
            question = entry["question"]
            prompt = entry["prompt"]

            gold_record = questions.get(qid)
            if gold_record is None:
                print(f"[WARN] Missing gold record for question_id {qid}")
                continue

            gold_sql = gold_record["gold_sql"]

            # ---------------------------------------------------------
            # 1. Generate SQL via LoRA + GBNF
            # ---------------------------------------------------------
            pred_sql = sqlcoder.generate_sql(prompt)

            # ---------------------------------------------------------
            # 2. Execute gold + predicted
            # ---------------------------------------------------------
            gold_exec = execute_sql(db_id, gold_sql)
            pred_exec = execute_sql(db_id, pred_sql)

            # Syntax & runtime classification
            error_msg = pred_exec.get("error", "")
            has_syntax_error = ("syntax error" in error_msg.lower())
            has_runtime_error = (not pred_exec.get("ok") and not has_syntax_error)

            if pred_exec.get("ok"):
                sql_success += 1
            else:
                if has_syntax_error:
                    syntax_error_count += 1
                else:
                    runtime_error_count += 1

            # ---------------------------------------------------------
            # 3. Compute metrics
            # ---------------------------------------------------------
            strict = exec_match_strict(gold_exec, pred_exec)
            loose = exec_match_lenient(gold_exec, pred_exec)
            jacc = row_jaccard(gold_exec, pred_exec)
            jaccard_sum += jacc

            if strict:
                strict_correct += 1
            if loose:
                lenient_correct += 1

            # ---------------------------------------------------------
            # 4. Pretty print per example
            # ---------------------------------------------------------
            print("\n" + "=" * 80)
            print(f"QUESTION ID: {qid}")
            print(f"DB ID      : {db_id}")
            print(f"QUESTION   : {question}")
            print("-" * 80)
            print("GOLD SQL:")
            print(gold_sql)
            print("\nPRED SQL (LoRA + GBNF constrained):")
            print(pred_sql)
            print("\nSQL DIFF (gold vs pred):")
            print(diff_sql(gold_sql, pred_sql) or "(no diff)")
            print("\nGOLD EXECUTION RESULT:")
            print(gold_exec)
            print("\nPRED EXECUTION RESULT:")
            print(pred_exec)

            print("\nSYNTAX ERROR?       ", "❌ YES" if has_syntax_error else "✅ NO")
            print("RUNTIME ERROR?      ", "❌ YES" if has_runtime_error else "✅ NO")
            print("EXECUTION SUCCEEDED?", "✅ YES" if pred_exec.get("ok") else "❌ NO")

            print(f"\nSTRICT EXEC MATCH?   {'✅ YES' if strict else '❌ NO'}")
            print(f"LENIENT EXEC MATCH?  {'✅ YES' if loose else '❌ NO'}")
            print(f"ROW JACCARD SIMILARITY: {jacc:.3f}")
            print("=" * 80)

            # ---------------------------------------------------------
            # 5. Log results file
            # ---------------------------------------------------------
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
                "syntax_error": has_syntax_error,
                "runtime_error": has_runtime_error,
            }) + "\n")

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    strict_acc = strict_correct / total
    lenient_acc = lenient_correct / total
    exec_ok_rate = sql_success / total
    avg_jaccard = jaccard_sum / total

    print("\n" + "#" * 80)
    print("# SQLCoder + LoRA + GBNF Evaluation Summary")
    print("#" * 80)
    print(f"Total examples                  : {total}")
    print(f"Pred SQL exec succeeded         : {sql_success} ({exec_ok_rate:.3f})")
    print(f"STRICT exec matches gold        : {strict_correct} ({strict_acc:.3f})")
    print(f"LENIENT exec matches gold       : {lenient_correct} ({lenient_acc:.3f})")
    print(f"Average row Jaccard similarity  : {avg_jaccard:.3f}")
    print(f"Syntax errors                   : {syntax_error_count} ({syntax_error_count/total:.3f})")
    print(f"Runtime errors                  : {runtime_error_count} ({runtime_error_count/total:.3f})")
    print(f"Results saved to                : {RESULTS_PATH}")
    print("#" * 80)


if __name__ == "__main__":
    main()
