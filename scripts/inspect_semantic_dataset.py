#!/usr/bin/env python3
"""Inspect v3 semantic prompt dataset quality."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train_semantic.jsonl"
VALID_PATH = PROCESSED_DIR / "valid_semantic.jsonl"
MANIFEST_PATH = PROCESSED_DIR / "semantic_dataset_manifest.jsonl"
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*")


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


def words(text: str) -> set[str]:
    return {word for word in WORD_RE.findall(text.lower()) if len(word) >= 5}


def overlap(prompt: str, answer: str) -> float:
    prompt_words = words(prompt)
    answer_words = words(answer)
    if not prompt_words:
        return 0
    return len(prompt_words & answer_words) / len(prompt_words)


def print_examples(title: str, records: list[dict[str, Any]]) -> None:
    print(f"\n## {title}")
    for index, record in enumerate(records, start=1):
        metadata = record["metadata"]
        print(
            f"{index:02d}. {metadata['kind']} likes={metadata['favorite_count']} id={metadata['tweet_id']}\n"
            f"    prompt: {message(record, 'user')}\n"
            f"    answer: {message(record, 'assistant')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--valid-path", type=Path, default=VALID_PATH)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--max-overlap", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_jsonl(args.train_path)
    valid = load_jsonl(args.valid_path)
    manifest = load_jsonl(args.manifest_path)
    records = train + valid
    prompts = [message(record, "user") for record in records]
    answers = [message(record, "assistant") for record in records]
    kind_counts = Counter(record["metadata"]["kind"] for record in records)
    duplicate_ids = [
        value for value, count in Counter(record["metadata"]["tweet_id"] for record in records).items() if count > 1
    ]
    duplicate_hashes = [
        value for value, count in Counter(record["metadata"]["text_hash"] for record in records).items() if count > 1
    ]
    bad_answers = [
        answer for answer in answers if answer.startswith("@") or "http://" in answer or "https://" in answer
    ]
    high_overlap = [
        record
        for record in records
        if overlap(message(record, "user"), message(record, "assistant")) > args.max_overlap
    ]

    print("# Semantic Dataset Summary")
    print(f"train={len(train)}")
    print(f"valid={len(valid)}")
    print(f"total={len(records)}")
    print(f"manifest={len(manifest)}")
    print(f"originals={kind_counts['original']}")
    print(f"replies={kind_counts['reply']}")
    print(f"unique_prompts={len(set(prompts))}")
    print(f"duplicate_tweet_ids={len(duplicate_ids)}")
    print(f"duplicate_text_hashes={len(duplicate_hashes)}")
    print(f"bad_answers={len(bad_answers)}")
    print(f"high_prompt_answer_overlap={len(high_overlap)}")

    random.seed(42)
    print_examples("Random 20", random.sample(records, min(20, len(records))))
    print_examples("Highest Overlap 20", high_overlap[:20])

    if (
        len(manifest) != len(records)
        or duplicate_ids
        or duplicate_hashes
        or bad_answers
        or len(set(prompts)) < len(records) * 0.5
        or len(high_overlap) > len(records) * 0.03
    ):
        raise SystemExit("Inspection failed. See counts above.")


if __name__ == "__main__":
    main()
