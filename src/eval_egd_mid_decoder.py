# src/eval_sqlcoder_egd_mid.py

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

from execution_guidance.mid_egd_beam_search import (
    generate_sql_mid_egd,
    MidEGDConfig,
    MidEGDBeamResult,
)

# --------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

PROMPTS_PATH = PROJECT_ROOT / "prompts.jsonl"
QUESTIONS_PATH = PROJECT_ROOT / "questions_used.jsonl"
SPIDER_DB_ROOT = PROJECT_ROOT / "spider_dataset" / "spider_data" / "database"

RESULTS_PATH = PROJECT_ROOT / "mid_decoder_egd_results.jsonl"

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
    Extraction for plain SQLCoder.

    - Prefer fenced ```sql blocks if present.
    - Otherwise, grab from FIRST 'select ' onward, optionally stop at the
      first semicolon, and treat that as the SQL.
    - If nothing matches, return the whole string stripped.
    """
    text = generated_text.strip()
    lower = text.lower()

    # 1) fenced ```sql block (if you ever decide to use it)
    if "```sql" in lower:
        try:
            _, after = text.split("```sql", 1)
            sql_body = after.split("```", 1)[0]
            return sql_body.strip()
        except Exception:
            pass

    if "```" in text:
        try:
            parts = text.split("```")
            return parts[-1].strip()
        except Exception:
            pass

    # 2) take from FIRST 'select ' onwards
    idx = lower.find("select ")
    if idx != -1:
        sql = text[idx:]
        semi = sql.find(";")
        if semi != -1:
            return sql[: semi + 1].strip()
        return sql.strip()

    # 3) fallback: entire text
    return text


def execute_sql(db_id: str, sql: str) -> Dict[str, Any]:
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
# Metrics
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

    if g_rows == p_rows:
        return True

    g_norm = _normalize_rows(g_rows)
    p_norm = _normalize_rows(p_rows)

    g_set = set(g_norm)
    p_set = set(p_norm)

    if g_set == p_set:
        return True

    if len(g_rows) == 1 and len(p_rows) == 1:
        g_counter = Counter(g_rows[0])
        p_counter = Counter(p_rows[0])
        if g_counter == p_counter:
            return True

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
        # Baseline pipeline (not used for mid-EGD generation, but handy to keep)
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
# Main SQLCoder + Mid-EGD evaluation
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

    # Mid-decoding EGD settings
    mid_egd_cfg = MidEGDConfig(
        num_beams=4,
        max_new_tokens=160,
        egd_interval=16,
        min_tokens_for_egd=32,
        egd_top_k=2,
    )

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

            # 1) Run manual beam search with mid-decoding EGD
            pred_sql, beam_debug = generate_sql_mid_egd(
                model=sqlcoder.model,
                tokenizer=sqlcoder.tokenizer,
                prompt=prompt,
                db_id=db_id,
                execute_sql_fn=execute_sql,
                extract_sql_fn=extract_sql,
                config=mid_egd_cfg,
            )

            # 2) Execute gold + predicted SQL
            gold_exec = execute_sql(db_id, gold_sql)
            pred_exec = execute_sql(db_id, pred_sql)

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

            # 4) Pretty-print summary + beam debug
            print("\n" + "=" * 80)
            print(f"QUESTION ID: {qid}")
            print(f"DB ID      : {db_id}")
            print(f"QUESTION   : {question}")
            print("-" * 80)
            print("GOLD SQL:")
            print(gold_sql)
            print("\nPRED SQL (mid-EGD best beam):")
            print(pred_sql)

            print("\nSQL DIFF (gold vs pred):")
            print(diff_sql(gold_sql, pred_sql) or "(no diff)")

            print("\nGOLD EXECUTION RESULT:")
            print(gold_exec)

            print("\nPRED EXECUTION RESULT:")
            print(pred_exec)

            print("\nFINAL BEAMS (after mid-decoding EGD):")
            for idx, b in enumerate(beam_debug):
                print(
                    f"  Beam {idx}: "
                    f"logprob={b.logprob:.2f}, "
                    f"killed={b.killed}, "
                    f"egd_checks={b.num_egd_checks}, "
                    f"egd_failures={b.num_egd_failures}"
                )
                print(f"    SQL: {b.sql}")

            print(f"\nSTRICT EXEC MATCH?   {'✅ YES' if strict else '❌ NO'}")
            print(f"LENIENT EXEC MATCH?  {'✅ YES' if loose else '❌ NO'}")
            print(f"ROW JACCARD SIMILARITY: {jacc:.3f}")
            print("=" * 80)

            # 5) JSONL logging
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
                        "beams": [
                            {
                                "text": b.text,
                                "sql": b.sql,
                                "logprob": b.logprob,
                                "killed": b.killed,
                                "num_egd_checks": b.num_egd_checks,
                                "num_egd_failures": b.num_egd_failures,
                            }
                            for b in beam_debug
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
    print("# SQLCoder + Mid-Decoding EGD Evaluation Summary")
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
