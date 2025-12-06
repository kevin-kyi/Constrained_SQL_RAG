from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from collections import Counter
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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
RESULTS_PATH = PROJECT_ROOT / "sqlcoder_results.jsonl"

# Choose your SQLCoder model here
SQLCODER_MODEL_NAME = "defog/sqlcoder-7b-2"  


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
            before, after = generated_text.split("```sql", 1)
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
    """
    Normalize rows as tuples so they can go into sets.
    """
    return [tuple(r) for r in rows]


def exec_match_strict(gold: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    """
    Strict execution match: both must succeed and rows must be identical.
    """
    if not gold.get("ok") or not pred.get("ok"):
        return False
    return gold["rows"] == pred["rows"]


def exec_match_lenient(gold: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    """
    Lenient execution match that ignores:
      - row order
      - duplicates
      - column order for single-row aggregates
      - extra rows in prediction (gold ⊆ pred)

    Returns True if any of the following holds:
      1) strict match
      2) sets of rows are equal (order + dup-insensitive)
      3) single-row, column-order-insensitive value match
      4) all gold rows are contained in predicted rows (gold ⊆ pred)
    """
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

    # 2) sets equal (order + dup-insensitive)
    if g_set == p_set:
        return True

    # 3) single-row aggregate: ignore column order (treat row as multiset)
    if len(g_rows) == 1 and len(p_rows) == 1:
        g_counter = Counter(g_rows[0])
        p_counter = Counter(p_rows[0])
        if g_counter == p_counter:
            return True

    # 4) gold subset of pred (all correct rows present, but extras allowed)
    if g_set.issubset(p_set):
        return True

    return False


def row_jaccard(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    """
    Jaccard similarity between gold and predicted result sets, treating rows as sets.

    J = |gold ∩ pred| / |gold ∪ pred|

    Returns 0.0 if either query fails or if both sets are empty.
    """
    if not gold.get("ok") or not pred.get("ok"):
        return 0.0

    g_norm = _normalize_rows(gold["rows"])
    p_norm = _normalize_rows(pred["rows"])

    g_set = set(g_norm)
    p_set = set(p_norm)

    if not g_set and not p_set:
        # both empty; you can argue this should be 1.0, but keep 0.0 to avoid inflating average
        return 0.0

    union = g_set | p_set
    inter = g_set & p_set

    if not union:
        return 0.0

    return len(inter) / len(union)


def diff_sql(gold_sql: str, pred_sql: str) -> str:
    """
    Unified diff of SQL strings for debugging.
    """
    diff_lines = unified_diff(
        gold_sql.splitlines(),
        pred_sql.splitlines(),
        fromfile="gold_sql",
        tofile="pred_sql",
        lineterm=""
    )
    return "\n".join(diff_lines)


# --------------------------------------------------------------------
# SQLCoder wrapper
# --------------------------------------------------------------------
@dataclass
class SQLCoderWrapper:
    model_name: str

    def __post_init__(self) -> None:
        import torch

        print(f"[INFO] Loading SQLCoder model in fp16 on GPU: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        # torch_dtype is technically deprecated in HF logs, but still works.
        # If you want to silence the warning, you can switch to dtype=torch.float16.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,   # fp16 to fit in GPU
            device_map="cuda:0",         # or "auto"
            low_cpu_mem_usage=True,
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=160,          # modest to save VRAM
            do_sample=False,
            temperature=0.0,
            num_beams=4,                 # beam search, as recommended
            early_stopping=True,
        )

    def generate_sql(self, prompt: str) -> str:
        """
        Run the model on the SQLCoder-style prompt, return extracted SQL string.
        """
        out = self.pipe(prompt)[0]["generated_text"]
        return extract_sql(out)


# --------------------------------------------------------------------
# Main evaluation loop
# --------------------------------------------------------------------
def main() -> None:
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"prompts.jsonl not found at {PROMPTS_PATH}")
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"questions_used.jsonl not found at {QUESTIONS_PATH}")

    prompts = list(load_jsonl(PROMPTS_PATH))
    questions = {row["question_id"]: row for row in load_jsonl(QUESTIONS_PATH)}

    print(f"[INFO] Loaded {len(prompts)} prompts")
    print(f"[INFO] Loaded {len(questions)} gold questions")

    sqlcoder = SQLCoderWrapper(SQLCODER_MODEL_NAME)

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

            # 4. Print clean summary to stdout
            print("\n" + "=" * 80)
            print(f"QUESTION ID: {qid}")
            print(f"DB ID      : {db_id}")
            print(f"QUESTION   : {question}")
            print("-" * 80)
            print("GOLD SQL:")
            print(gold_sql)
            print("\nPRED SQL:")
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
    print("# SQLCoder Evaluation Summary")
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
