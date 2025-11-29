import json
import os
import random
from collections import defaultdict, deque
from pathlib import Path


# We store the actual dataset files here:
#   spider_dataset/spider_data/
SPIDER_ROOT = Path("spider_dataset") / "spider_data"
TABLES_JSON = SPIDER_ROOT / "tables.json"
TRAIN_JSON = SPIDER_ROOT / "train_spider.json"  # you can change to dev.json if you prefer


def _check_files_exist():
    """Ensure the expected Spider data files exist and give a helpful error if not."""
    missing = []
    for path in [SPIDER_ROOT, TABLES_JSON, TRAIN_JSON]:
        if path == SPIDER_ROOT:
            if not path.exists():
                missing.append(str(path))
        else:
            if not path.is_file():
                missing.append(str(path))

    if missing:
        msg_lines = [
            "[ERROR] Missing Spider data files.",
            "The following paths were not found:",
            *[f"  - {m}" for m in missing],
            "",
            "Make sure you have downloaded the official Spider dataset from:",
            "  https://yale-lily.github.io/spider",
            "and placed these inside spider_dataset/spider_data/:",
            "  - tables.json",
            "  - train_spider.json",
            "  - train_others.json",
            "  - dev.json",
            "  - database/   (directory of DBs)",
            "",
            "You can first run: python scripts/download_spider.py",
            "to set up the directory layout (spider_dataset/spider_data/).",
        ]
        raise SystemExit("\n".join(msg_lines))


def load_tables_schema():
    """Load tables.json and build convenient lookups."""
    with open(TABLES_JSON, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    # Build: db_id -> schema struct
    schema_by_db = {}
    for db in tables_data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]  # list[str]
        column_names = db["column_names_original"]  # list[[table_idx, col_name], ...]
        foreign_keys = db["foreign_keys"]  # list[[col_idx1, col_idx2], ...]

        # Map column index -> table index
        col_to_table = [tbl_idx for (tbl_idx, _col_name) in column_names]

        # Build table-level adjacency from foreign_keys
        adj = defaultdict(set)
        for col1, col2 in foreign_keys:
            t1 = col_to_table[col1]
            t2 = col_to_table[col2]
            if t1 == -1 or t2 == -1:
                continue  # skip special "all tables" index
            adj[t1].add(t2)
            adj[t2].add(t1)

        schema_by_db[db_id] = {
            "table_names": table_names,
            "col_to_table": col_to_table,
            "foreign_keys": foreign_keys,
            "table_adj": adj,  # dict[int -> set[int]]
        }

    return schema_by_db


def bfs_hops_from_table(adj, start_table_idx):
    """
    Given table adjacency (int -> set[int]) and a starting table index,
    return a dict: table_idx -> hop distance from start.
    """
    dist = {start_table_idx: 0}
    q = deque([start_table_idx])

    while q:
        cur = q.popleft()
        for nei in adj[cur]:
            if nei not in dist:
                dist[nei] = dist[cur] + 1
                q.append(nei)

    return dist


def sample_tables(schema_by_db, n_samples=5, seed=42):
    """Randomly sample (db_id, table_idx) pairs."""
    random.seed(seed)
    samples = []

    db_ids = list(schema_by_db.keys())
    while len(samples) < n_samples and db_ids:
        db_id = random.choice(db_ids)
        schema = schema_by_db[db_id]
        table_names = schema["table_names"]
        if not table_names:
            continue
        t_idx = random.randrange(len(table_names))
        samples.append((db_id, t_idx))

    return samples


def inspect_tables_and_fks(schema_by_db, n_samples=5):
    print("=== Table + FK hop inspection ===")
    samples = sample_tables(schema_by_db, n_samples=n_samples)

    for i, (db_id, t_idx) in enumerate(samples, 1):
        schema = schema_by_db[db_id]
        table_names = schema["table_names"]
        adj = schema["table_adj"]

        start_name = table_names[t_idx]
        print(f"\n[{i}] db_id = {db_id}, start_table = {start_name} (idx={t_idx})")

        dist = bfs_hops_from_table(adj, t_idx)
        hops = defaultdict(list)
        for tbl_idx, d in dist.items():
            hops[d].append(table_names[tbl_idx])

        for d in sorted(hops.keys()):
            print(f"  Hop {d}: {', '.join(hops[d])}")


def inspect_questions(n_samples=5, seed=123):
    print("\n=== Sample NL questions and SQL queries ===")
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(seed)
    samples = random.sample(data, k=min(n_samples, len(data)))

    for i, ex in enumerate(samples, 1):
        print(f"\n[{i}] db_id = {ex['db_id']}")
        print(f"Question: {ex['question']}")
        print(f"SQL:      {ex['query']}")


def main():
    _check_files_exist()
    schema_by_db = load_tables_schema()
    inspect_tables_and_fks(schema_by_db, n_samples=5)
    inspect_questions(n_samples=5)


if __name__ == "__main__":
    main()
