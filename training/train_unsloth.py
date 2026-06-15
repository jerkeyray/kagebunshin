#!/usr/bin/env python3
"""Fine-tune Kagebunshin with QLoRA using Unsloth."""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
DEFAULT_TRAIN_PATH = ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_VALID_PATH = ROOT / "data" / "processed" / "valid.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "adapters" / "kagebunshin-qwen3-4b-lora"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-seq-length", type=positive_int, default=1024)
    parser.add_argument("--epochs", type=positive_float, default=2.0)
    parser.add_argument("--learning-rate", type=positive_float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=positive_int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=8)
    parser.add_argument("--lora-r", type=positive_int, default=16)
    parser.add_argument("--lora-alpha", type=positive_int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-steps", type=positive_int, default=25)
    parser.add_argument("--save-steps", type=positive_int, default=50)
    parser.add_argument("--logging-steps", type=positive_int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def validate_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise SystemExit(f"{path} has no records")

    for index, record in enumerate(records[:10], start=1):
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise SystemExit(f"{path}: record {index} missing messages list")
        roles = [message.get("role") for message in messages]
        if roles != ["system", "user", "assistant"]:
            raise SystemExit(f"{path}: record {index} has unexpected roles: {roles}")


def preflight_dataset(train_path: Path, valid_path: Path) -> None:
    if not train_path.exists():
        raise SystemExit(f"missing train file: {train_path}")
    if not valid_path.exists():
        raise SystemExit(f"missing valid file: {valid_path}")

    train = load_jsonl(train_path)
    valid = load_jsonl(valid_path)
    validate_records(train_path, train)
    validate_records(valid_path, valid)
    print(f"train_examples={len(train)}")
    print(f"valid_examples={len(valid)}")


def format_example(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {"text": text}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    preflight_dataset(args.train_path, args.valid_path)

    import unsloth  # noqa: F401
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.train_path),
            "validation": str(args.valid_path),
        },
    )
    dataset = dataset.map(
        lambda example: format_example(example, tokenizer),
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to="none",
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "args": training_args,
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs.update(
            {
                "tokenizer": tokenizer,
                "dataset_text_field": "text",
                "max_seq_length": args.max_seq_length,
                "packing": False,
            }
        )

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"saved_adapter={args.output_dir}")


if __name__ == "__main__":
    main()
