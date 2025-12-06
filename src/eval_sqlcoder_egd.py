from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# NOTE: we import relatively, because this script lives inside src/
from execution_guidance.generators import generate_beam_candidates, BeamCandidate
from execution_guidance.egd_utils import apply_egd_reranking

# --------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

PROMPTS_PATH = PROJECT_ROOT / "prompts.jsonl"
QUESTIONS_PATH = PROJECT_ROOT / "questions_used.jsonl"
SPIDER_DB_ROOT = PROJECT_ROOT / "spider_dataset" / "spider_data" / "database"
RESULTS_PATH = PROJECT_ROOT / "sqlcoder_egd_results.jsonl"

# Adjust if you’re using a different SQLCoder variant
SQLCODER_MODEL_NAME = "defog/sqlcoder-7b-2"


# --------------------------------------------------------------------
# Basic I/O + SQL extraction
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

    # ```sql fenced block
    if "```sql" in lower:
        try:
            _, after = generated_text.split("```sql", 1)
            sql_body = after.split("```", 1)[0]
            return sql_body.strip()
        except Exception:
            pass

    # Any fenced block
    if "```" in generated_text:
        try:
            parts = generated_text.split("```")
            return parts[-1].strip()
        except Exception:
            pass

    # From last 'select '
    idx = lower.rfind("select ")
    if idx != -1:
        return generated_text[idx:].strip()

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
    """
    Strict execution match: both must succeed and rows must be identical.
    """
    if not gold.get("ok") or not pred.get("ok"):
        return False
    return gold["rows"] == pred["rows"]


def exec_match_lenient(gold: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    """
    Lenient execution match that ignores row order, duplicates,
    and treats single-row aggregates more flexibly.
    """
    if not gold.get("ok") or not pred.get("ok"):
        return False

    g_rows = gold["rows"]
    p_rows = pred["rows"]

    # Strict equality
    if g_rows == p_rows:
        return True

    g_norm = _normalize_rows(g_rows)
    p_norm = _normalize_rows(p_rows)

    g_set = set(g_norm)
    p_set = set(p_norm)

    # Sets equal (order/dup insensitive)
    if g_set == p_set:
        return True

    # Single-row aggregate: ignore column order
    if len(g_rows) == 1 and len(p_rows) == 1:
        g_counter = Counter(g_rows[0])
        p_counter = Counter(p_rows[0])
        if g_counter == p_counter:
            return True

    # Gold subset of pred (allow extra rows)
    if g_set.issubset(p_set):
        return True

    return False


def row_jaccard(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    """
    Jaccard similarity between gold and predicted result sets, treating rows as sets.
    """
    if not gold.get("ok") or not pred.get("ok"):
        return 0.0

    g_norm = _normalize_rows(gold["rows"])
    p_norm = _normalize_rows(pred["rows"])

    g_set = set(g_norm)
    p_set = set(p_norm)

    if not g_set and not p_set:
        # both empty; keep as 0.0 to avoid inflating avg
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
        lineterm="",
    )
    return "\n".join(diff_lines)


# --------------------------------------------------------------------
# SQLCoder wrapper
# --------------------------------------------------------------------
@dataclass
class SQLCoderWrapper:
    model_name: str

    def __post_init__(self) -> None:
        print(f"[INFO] Loading SQLCoder model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        # Baseline pipeline helper (not used directly in EGD, but handy to keep)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=160,
            do_sample=False,
            temperature=0.0,
            num_beams=4,
            early_stopping=True,
        )

    def generate_sql(self, prompt: str) -> str:
        out = self.pipe(prompt)[0]["generated_text"]
        return extract_sql(out)


# --------------------------------------------------------------------
# Main EGD evaluation
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
            qid = entry["question_id"]
            db_id = entry["db_id"]
            question = entry["question"]
            prompt = entry["prompt"]

            gold_record = questions.get(qid)
            if gold_record is None:
                print(f"[WARN] Missing gold record for question_id={qid}, skipping.")
                continue

            gold_sql = gold_record["gold_sql"]

            # 1) Generate beam candidates from SQLCoder (no EGD yet)
            beam_candidates: List[BeamCandidate] = generate_beam_candidates(
                model=sqlcoder.model,
                tokenizer=sqlcoder.tokenizer,
                prompt=prompt,
                num_beams=8,
                num_return_sequences=8,
                max_new_tokens=160,
            )

            # 2) Apply execution-guided reranking (cheap EGD)
            best_candidate, egd_candidates = apply_egd_reranking(
                db_id=db_id,
                raw_candidates=beam_candidates,
                extract_sql_fn=extract_sql,
                execute_sql_fn=execute_sql,
            )

            pred_sql = best_candidate.sql
            pred_exec = best_candidate.exec_result
            gold_exec = execute_sql(db_id, gold_sql)

            # 3) Metrics
            total += 1

            if pred_exec.get("ok"):
                sql_success += 1

            strict = exec_match_strict(gold_exec, pred_exec)
            loose = exec_match_lenient(gold_exec, pred_exec)
            jacc = row_jaccard(gold_exec, pred_exec)
            jaccard_sum += jacc

            if strict:
                strict_correct += 1
            if loose:
                lenient_correct += 1

            # 4) Pretty print with beam viz
            print("\n" + "=" * 80)
            print(f"QUESTION ID: {qid}")
            print(f"DB ID      : {db_id}")
            print(f"QUESTION   : {question}")
            print("-" * 80)
            print("GOLD SQL:")
            print(gold_sql)
            print("\nPRED SQL (best EGD candidate):")
            print(pred_sql)

            print("\nSQL DIFF (gold vs pred):")
            print(diff_sql(gold_sql, pred_sql) or "(no diff)")

            print("\nGOLD EXECUTION RESULT:")
            print(gold_exec)

            print("\nPRED EXECUTION RESULT (best beam):")
            print(pred_exec)

            print("\nALL CANDIDATE BEAMS (after EGD scoring):")
            for idx, cand in enumerate(egd_candidates):
                ok = cand.exec_result.get("ok", False)
                rows = cand.exec_result.get("rows", [])
                row_count = len(rows) if ok and isinstance(rows, list) else "N/A"
                print(
                    f"  Beam {idx}: "
                    f"score={cand.score:.3f}, "
                    f"prob={cand.prob:.4f}, "
                    f"logprob={cand.logprob:.2f}, "
                    f"ok={ok}, rows={row_count}"
                )
                print(f"    SQL: {cand.sql}")

            print(f"\nSTRICT EXEC MATCH?   {'✅ YES' if strict else '❌ NO'}")
            print(f"LENIENT EXEC MATCH?  {'✅ YES' if loose else '❌ NO'}")
            print(f"ROW JACCARD SIMILARITY: {jacc:.3f}")
            print("=" * 80)

            # 5) JSONL logging including all EGD candidates
            fout.write(
                json.dumps(
                    {
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
                        "egd_candidates": [
                            {
                                "sql": c.sql,
                                "text": getattr(c, "text", ""),
                                "logprob": c.logprob,
                                "prob": c.prob,
                                "exec_result": c.exec_result,
                                "score": c.score,
                            }
                            for c in egd_candidates
                        ],
                    }
                )
                + "\n"
            )
            fout.flush()

    # 6) Aggregate summary
    strict_acc = strict_correct / total if total > 0 else 0.0
    lenient_acc = lenient_correct / total if total > 0 else 0.0
    exec_ok_rate = sql_success / total if total > 0 else 0.0
    avg_jaccard = jaccard_sum / total if total > 0 else 0.0

    print("\n" + "#" * 80)
    print("# SQLCoder+EGD Evaluation Summary")
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
