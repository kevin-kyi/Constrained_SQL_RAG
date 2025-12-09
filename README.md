# Constrained_SQL_RAG
Schema aware Retrieval Augmented Generation + Picard(syntactical constraint)/Execution Guided Decoding(Semantic Constraint) Text-To-SQL Generator for Tabular Data

## Downloading Spider 1.0 natural language + SQL queries, and actual Spider 1.0 dataset
1. Download NL + SQL queries: run python3 src/download_spider.py
   This step will download the NL + SQL queries under Constrained_SQL_RAG/spider_dataset

2. Download all tables/DB and gold executions
Go to the Spider 1.0 website (https://yale-lily.github.io/spider), click "Spider Dataset" under Getting Started, then extract this zip(/spider_data) and place the database directory under Constrained_SQL_RAG/spider_dataset

Once you follow these step /spider_dataset should have /spider_dataset/spider which is the NL + SQL queries, and /spider_dataset/spider_data which is the actual database

## Retrieval usage
- Query tables and get JSON output of candidate tables:  
  `python scripts/query_tables.py --query "flights and airports" --top-k 3 --mode hybrid`  
  Outputs `{question, mode, candidate_tables: [{db_id, table_name, columns, score}], db_id (if single)}`.

## Grammar-constrained SQL generation
- SQL syntax is enforced with a SQLite-flavored GBNF grammar at `src/sql/grammars/sqlite.gbnf`.
- `src/sql_gbnf.py` exposes `build_sqlite_prefix_allowed_tokens_fn(tokenizer, grammar_text)` to plug into `transformers.generate`.
- The SQLCoder smoke test (`src/test_scripts/test_sqlcoder.py`) now uses the grammar by default; disable with `USE_GBNF=0` or point to a custom grammar via `SQLITE_GBNF_PATH`.

## LoRA fine-tuning SQLCoder on Spider
- Install deps: `pip install -r requirements.txt` (adds `datasets`, `peft`, `bitsandbytes`, `accelerate` for QLoRA).
- Run QLoRA training (requires a CUDA GPU):  
  `python src/train_sqlcoder_lora.py --output_dir outputs/sqlcoder-lora-spider --num_train_epochs 3 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --eval_steps 200 --save_steps 400`
- Flags: `--include_train_others` to append Spider `train_others.json`, `--max_train_samples`/`--max_eval_samples` to cap records, `--no_4bit` for full-precision LoRA (mandatory on Python 3.13+), `--lora_r`/`--lora_alpha`/`--lora_dropout` for adapter sizing.
- Outputs: `output_dir` stores the LoRA adapter weights and tokenizer; load with `peft.PeftModel.from_pretrained` alongside the base `defog/sqlcoder-7b-2`.
