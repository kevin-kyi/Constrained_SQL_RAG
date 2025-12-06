import json
import sqlite3
import torch
import re
from pathlib import Path
from typing import Dict, Any, Set
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = PROJECT_ROOT / "prompts.jsonl"
QUESTIONS_PATH = PROJECT_ROOT / "questions_used.jsonl"
RESULTS_PATH = PROJECT_ROOT / "sqlcoder_egd_col_results.jsonl"
SPIDER_DB_ROOT = PROJECT_ROOT / "spider_dataset" / "spider_data" / "database"
MODEL_NAME = "defog/sqlcoder-7b-2"

# -----------------------------------------------------------------------------
# PART 1: HELPERS
# -----------------------------------------------------------------------------
def get_db_path(db_id: str) -> Path:
    return SPIDER_DB_ROOT / db_id / f"{db_id}.sqlite"

def execute_sql(db_id: str, sql: str) -> Dict[str, Any]:
    db_path = get_db_path(db_id)
    if not db_path.exists():
        return {"ok": False, "error": f"DB not found"}
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return {"ok": True, "rows": rows}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def exec_match_strict(gold: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    if not gold.get("ok") or not pred.get("ok"):
        return False
    return gold["rows"] == pred["rows"]

def get_db_schema(db_path: Path) -> Dict[str, Set[str]]:
    """
    Returns a dict: { 'table_name': {'col1', 'col2', ...} }
    """
    if not db_path.exists():
        return {}
        
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0].lower() for r in cursor.fetchall()]
    
    schema = {}
    for t in tables:
        try:
            cursor.execute(f"PRAGMA table_info('{t}')")
            cols = {r[1].lower() for r in cursor.fetchall()}
            schema[t] = cols
        except:
            pass
            
    conn.close()
    return schema

# -----------------------------------------------------------------------------
# PART 2: COLUMN-AWARE EGD GENERATOR
# -----------------------------------------------------------------------------
class EGDGenerator:
    def __init__(self, model_name):
        print(f"[INFO] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.device = self.model.device

    def generate(self, prompt: str, db_id: str, max_new_tokens=150):
        db_path = get_db_path(db_id)
        # Load full schema: {table: {cols}}
        db_schema = get_db_schema(db_path)
        
        # Flatten for easy checking
        all_tables = set(db_schema.keys())
        all_columns = set().union(*db_schema.values())
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        curr_ids = inputs.input_ids.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(curr_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Top 10 candidates
            top_k_logits, top_k_indices = torch.topk(next_token_logits, 10, dim=1)
            
            # --- CONTEXT ANALYSIS ---
            current_text = self.tokenizer.decode(curr_ids[0], skip_special_tokens=True)
            
            # 1. Detect "Dot" Context (e.g. "student.")
            # We want to know if the character immediately before the cursor is a '.'
            is_dot_context = current_text.strip().endswith(".")
            
            # 2. Detect "Table" Context (e.g. "FROM ", "JOIN ")
            last_words = current_text.lower().replace(",", " ").split()
            last_word = last_words[-1] if last_words else ""
            is_table_context = last_word in ["from", "join", "update"]
            
            best_token_id = top_k_indices[0][0]
            
            # --- CONSTRAINT LOGIC ---
            
            # CASE A: We are typing a Column (after a dot)
            if is_dot_context:
                # Find the table alias/name before the dot
                # Heuristic: split by space, take last segment, split by dot
                # ex: "SELECT T1." -> last segment "T1." -> table "T1"
                prefix = current_text.strip().split()[-1][:-1].lower() 
                
                # Check if this prefix is a known table name
                # (Note: Handling aliases correctly requires a full parser, 
                #  so we use a fallback: if we can't find the table, allow ALL columns)
                allowed_cols = db_schema.get(prefix, all_columns)
                
                # Filter candidates
                for idx in top_k_indices[0]:
                    token_str = self.tokenizer.decode(idx).strip().lower()
                    # Check if token is a prefix of any valid column
                    if any(c.startswith(token_str) for c in allowed_cols):
                        best_token_id = idx
                        break
            
            # CASE B: We are typing a Table (after FROM/JOIN)
            elif is_table_context:
                for idx in top_k_indices[0]:
                    token_str = self.tokenizer.decode(idx).strip().lower()
                    if any(t.startswith(token_str) for t in all_tables):
                        best_token_id = idx
                        break
                        
            # ------------------------

            next_token = best_token_id.unsqueeze(0).unsqueeze(0)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            if best_token_id == self.tokenizer.eos_token_id:
                break
            if "```" in self.tokenizer.decode(curr_ids[0][-3:]):
                break

        full_text = self.tokenizer.decode(curr_ids[0], skip_special_tokens=True)
        return self._extract_sql(full_text)

    def _extract_sql(self, text):
        if "```sql" in text:
            return text.split("```sql")[1].split("```")[0].strip()
        if "SELECT" in text:
            return "SELECT" + text.split("SELECT", 1)[1]
        return text

# -----------------------------------------------------------------------------
# PART 3: MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    generator = EGDGenerator(MODEL_NAME)
    
    with open(PROMPTS_PATH, "r") as f:
        prompts = [json.loads(line) for line in f]
    
    with open(QUESTIONS_PATH, "r") as f:
        gold_data = {row['question_id']: row for row in (json.loads(line) for line in f)}

    print(f"Loaded {len(prompts)} prompts. Starting Column-Aware EGD...")
    
    results = []
    correct_count = 0
    
    for i, p in enumerate(prompts):
        qid = p['question_id']
        db_id = p['db_id']
        gold_info = gold_data.get(qid)
        
        if not gold_info: continue
            
        print(f"\n--- [{i+1}/{len(prompts)}] QID: {qid} DB: {db_id} ---")
        
        pred_sql = generator.generate(p['prompt'], db_id)
        
        gold_exec = execute_sql(db_id, gold_info['gold_sql'])
        pred_exec = execute_sql(db_id, pred_sql)
        
        is_match = exec_match_strict(gold_exec, pred_exec)
        if is_match: correct_count += 1
            
        # Verbose Reporting
        print(f"Pred: {pred_sql}")
        if not pred_exec['ok']:
             print(f"Exec Error: {pred_exec['error']}")
        else:
             print(f"Exec Rows: {str(pred_exec['rows'])[:100]}...")
        
        status = "✅ MATCH" if is_match else "❌ MISMATCH"
        print(f"Result: {status}")
        
        results.append({
            "question_id": qid,
            "pred_sql": pred_sql,
            "match": is_match
        })

    print("\n" + "="*40)
    print(f"Final Accuracy: {correct_count}/{len(prompts)} ({correct_count/len(prompts):.2%})")
    print("="*40)

if __name__ == "__main__":
    main()