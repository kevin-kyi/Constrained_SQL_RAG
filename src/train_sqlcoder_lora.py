from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.table_retrieval.schemas import DatabaseSchema, load_spider_schemas


DEFAULT_SPIDER_DIR = PROJECT_ROOT / "spider_dataset" / "spider_data"


@dataclass
class SpiderExample:
    prompt: str
    sql: str
    db_id: str


def load_split(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_create_block(table_name: str, columns: Iterable[tuple[str, str]]) -> str:
    col_lines = [f"  {name} {ctype.upper()}" for name, ctype in columns]
    return f"CREATE TABLE {table_name} (\n" + ",\n".join(col_lines) + "\n);"


def format_schema(db: DatabaseSchema) -> tuple[str, str]:
    create_blocks = [
        build_create_block(table.name, [(col.name, col.type) for col in table.columns])
        for table in db.tables
    ]
    fk_pairs: list[str] = []
    for table in db.tables:
        for fk in table.foreign_keys:
            fk_pairs.append(f"{table.name}.{fk.source_column} = {fk.target_table}.{fk.target_column}")

    fk_block = "\n".join([f"-- {fk}" for fk in fk_pairs]) if fk_pairs else "-- No foreign key relationships"
    schema_block = "\n\n".join(create_blocks)
    return schema_block, fk_block


def build_prompt(question: str, schema_block: str, fk_block: str) -> str:
    return (
        "## Task\n"
        "Generate a SQLite SQL query to answer the following question:\n"
        f"{question}\n\n"
        "### Database Schema\n"
        "This query will run on a database whose schema is represented as:\n"
        f"{schema_block}\n\n"
        f"{fk_block}\n\n"
        "### SQL\n"
        "Given the database schema, Write only the valid SQLite SQL query that answers the question:\n"
    )


def make_training_examples(
    spider_dir: Path,
    schema_cache: Dict[str, tuple[str, str]],
    include_train_others: bool = False,
) -> list[SpiderExample]:
    train_path = spider_dir / "train_spider.json"
    other_path = spider_dir / "train_others.json"

    payload: list[dict] = load_split(train_path)
    if include_train_others and other_path.exists():
        payload += load_split(other_path)

    examples: list[SpiderExample] = []
    for row in payload:
        db_id = row["db_id"]
        question = row["question"]
        sql = row["query"]
        if db_id not in schema_cache:
            raise KeyError(f"Database {db_id} missing from schema cache.")
        schema_block, fk_block = schema_cache[db_id]
        prompt = build_prompt(question, schema_block, fk_block)
        examples.append(SpiderExample(prompt=prompt, sql=sql, db_id=db_id))
    return examples


def make_eval_examples(spider_dir: Path, schema_cache: Dict[str, tuple[str, str]]) -> list[SpiderExample]:
    dev_path = spider_dir / "dev.json"
    payload = load_split(dev_path)

    examples: list[SpiderExample] = []
    for row in payload:
        db_id = row["db_id"]
        question = row["question"]
        sql = row["query"]
        if db_id not in schema_cache:
            raise KeyError(f"Database {db_id} missing from schema cache.")
        schema_block, fk_block = schema_cache[db_id]
        prompt = build_prompt(question, schema_block, fk_block)
        examples.append(SpiderExample(prompt=prompt, sql=sql, db_id=db_id))
    return examples


def tokenize_example(example: dict, tokenizer: AutoTokenizer, max_length: int) -> dict:
    prompt: str = example["prompt"]
    sql: str = example["sql"]

    full_text = prompt + sql + tokenizer.eos_token
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
    )["input_ids"]

    labels = [-100] * len(prompt_tokens)
    labels += tokenized["input_ids"][len(prompt_tokens):]
    labels = labels[:max_length]

    return {
        "input_ids": tokenized["input_ids"][:max_length],
        "attention_mask": tokenized["attention_mask"][:max_length],
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for SQLCoder on Spider.")
    parser.add_argument("--model_name", type=str, default="defog/sqlcoder-7b-2")
    parser.add_argument("--spider_dir", type=Path, default=DEFAULT_SPIDER_DIR)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "sqlcoder-lora-spider")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=400)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=500)
    parser.add_argument("--include_train_others", action="store_true", help="Append Spider train_others.json to the training set.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 where supported.")
    parser.add_argument("--use_4bit", dest="use_4bit", action="store_true", help="Enable 4-bit QLoRA loading.")
    parser.add_argument("--no_4bit", dest="use_4bit", action="store_false", help="Load the base model in full precision.")
    parser.set_defaults(use_4bit=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.use_4bit and sys.version_info >= (3, 13):
        print("[WARN] Python 3.13 detected; torch.compile is unavailable, disabling 4-bit load. "
              "Re-run with --no_4bit to silence this.")
        args.use_4bit = False

    if args.use_4bit and not torch.cuda.is_available():
        raise EnvironmentError("4-bit loading requires a CUDA-enabled GPU.")

    spider_dir: Path = args.spider_dir
    tables_path = spider_dir / "tables.json"
    if not tables_path.exists():
        raise FileNotFoundError(f"Spider tables.json not found at {tables_path}")

    schema_cache: Dict[str, tuple[str, str]] = {}
    for db in load_spider_schemas(tables_path):
        schema_cache[db.db_id] = format_schema(db)

    train_examples = make_training_examples(
        spider_dir,
        schema_cache,
        include_train_others=args.include_train_others,
    )
    eval_examples = make_eval_examples(spider_dir, schema_cache)

    if args.max_train_samples:
        train_examples = train_examples[: args.max_train_samples]
    if args.max_eval_samples:
        eval_examples = eval_examples[: args.max_eval_samples]

    print(f"[INFO] Training examples: {len(train_examples)}")
    print(f"[INFO] Eval examples:     {len(eval_examples)}")

    train_dataset = Dataset.from_list([ex.__dict__ for ex in train_examples])
    eval_dataset = Dataset.from_list([ex.__dict__ for ex in eval_examples])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_length

    tokenize_fn = partial(tokenize_example, tokenizer=tokenizer, max_length=args.max_seq_length)
    train_tokenized = train_dataset.map(tokenize_fn, remove_columns=train_dataset.column_names, desc="Tokenizing train split")
    eval_tokenized = eval_dataset.map(tokenize_fn, remove_columns=eval_dataset.column_names, desc="Tokenizing eval split")

    compute_dtype = (
        torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else torch.float16
    )
    if not torch.cuda.is_available():
        compute_dtype = torch.float32
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

    if args.use_4bit:
        model_kwargs.update(
            load_in_4bit=True,
            torch_dtype=compute_dtype,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs.update(torch_dtype=compute_dtype)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    bf16_flag = bool(args.bf16 and torch.cuda.is_available())
    fp16_flag = bool(not bf16_flag and torch.cuda.is_available())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=bf16_flag,
        fp16=fp16_flag,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
