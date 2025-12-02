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
