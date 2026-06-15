#!/usr/bin/env python3
"""Inspect the heuristic instruction tuning dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_TRAIN_PATH = PROCESSED_DIR / "train_instruct.jsonl"
DEFAULT_VALID_PATH = PROCESSED_DIR / "valid_instruct.jsonl"
DEFAULT_MANIFEST_PATH = PROCESSED_DIR / "instruction_dataset_manifest.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def message(record: dict[str, Any], role: str) -> str:
    for item in record["messages"]:
        if item["role"] == role:
            return item["content"]
    return ""


def duplicate_values(records: list[dict[str, Any]], key: str) -> list[str]:
    counts = Counter(record["metadata"][key] for record in records)
    return [value for value, count in counts.items() if count > 1]


def print_examples(title: str, records: list[dict[str, Any]]) -> None:
    print(f"\n## {title}")
    for index, record in enumerate(records, start=1):
        metadata = record["metadata"]
        prompt = message(record, "user")
        answer = message(record, "assistant")
        print(
            f"{index:02d}. "
            f"{metadata['kind']} "
            f"likes={metadata['favorite_count']} "
            f"id={metadata['tweet_id']}\n"
            f"    prompt: {prompt}\n"
            f"    answer: {answer}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_jsonl(args.train_path)
    valid = load_jsonl(args.valid_path)
    manifest = load_jsonl(args.manifest_path)
    all_records = train + valid

    texts = [message(record, "assistant") for record in all_records]
    prompts = [message(record, "user") for record in all_records]
    duplicate_tweet_ids = duplicate_values(all_records, "tweet_id")
    duplicate_text_hashes = duplicate_values(all_records, "text_hash")
    kind_counts = Counter(record["metadata"]["kind"] for record in all_records)
    bad_texts = [
        text
        for text in texts
        if text.startswith("@") or "http://" in text or "https://" in text
    ]
    generic_prompts = [
        prompt for prompt in prompts if prompt == "Write a tweet in Aditya's style."
    ]

    print("# Instruction Dataset Summary")
    print(f"train={len(train)}")
    print(f"valid={len(valid)}")
    print(f"total={len(all_records)}")
    print(f"manifest={len(manifest)}")
    print(f"originals={kind_counts['original']}")
    print(f"replies={kind_counts['reply']}")
    print(f"duplicate_tweet_ids={len(duplicate_tweet_ids)}")
    print(f"duplicate_text_hashes={len(duplicate_text_hashes)}")
    print(f"bad_assistant_texts={len(bad_texts)}")
    print(f"generic_prompts={len(generic_prompts)}")

    random.seed(42)
    print_examples("Random 20", random.sample(all_records, min(20, len(all_records))))
    print_examples(
        "Shortest 20 Answers",
        sorted(all_records, key=lambda record: len(message(record, "assistant")))[:20],
    )
    print_examples(
        "Longest 20 Answers",
        sorted(
            all_records,
            key=lambda record: len(message(record, "assistant")),
            reverse=True,
        )[:20],
    )

    if (
        duplicate_tweet_ids
        or duplicate_text_hashes
        or bad_texts
        or generic_prompts
        or len(manifest) != len(all_records)
    ):
        raise SystemExit("Inspection failed. See counts above.")


if __name__ == "__main__":
    main()
