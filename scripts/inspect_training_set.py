#!/usr/bin/env python3
"""Inspect chat-format training data."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
VALID_PATH = PROCESSED_DIR / "valid.jsonl"
MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def assistant_text(record: dict[str, Any]) -> str:
    for message in record["messages"]:
        if message["role"] == "assistant":
            return message["content"]
    return ""


def print_examples(title: str, records: list[dict[str, Any]]) -> None:
    print(f"\n## {title}")
    for index, record in enumerate(records, start=1):
        metadata = record["metadata"]
        text = assistant_text(record).replace("\n", " ")
        print(
            f"{index:02d}. "
            f"split={metadata['split']} "
            f"likes={metadata['favorite_count']} "
            f"id={metadata['tweet_id']} "
            f"text={text}"
        )


def duplicate_values(records: list[dict[str, Any]], key: str) -> list[str]:
    counts = Counter(record["metadata"][key] for record in records)
    return [value for value, count in counts.items() if count > 1]


def main() -> None:
    train = load_jsonl(TRAIN_PATH)
    valid = load_jsonl(VALID_PATH)
    manifest = load_jsonl(MANIFEST_PATH)
    all_records = train + valid
    texts = [assistant_text(record) for record in all_records]

    leading_mentions = [text for text in texts if text.startswith("@")]
    urls = [text for text in texts if "http://" in text or "https://" in text]
    duplicate_tweet_ids = duplicate_values(all_records, "tweet_id")
    duplicate_hashes = duplicate_values(all_records, "text_hash")

    print("# Training Set Summary")
    print(f"train={len(train)}")
    print(f"valid={len(valid)}")
    print(f"total={len(all_records)}")
    print(f"manifest={len(manifest)}")
    print(f"duplicate_tweet_ids={len(duplicate_tweet_ids)}")
    print(f"duplicate_text_hashes={len(duplicate_hashes)}")
    print(f"leading_mentions={len(leading_mentions)}")
    print(f"urls_in_assistant_text={len(urls)}")

    random.seed(42)
    random_sample = random.sample(all_records, min(20, len(all_records)))
    shortest = sorted(all_records, key=lambda record: len(assistant_text(record)))[:20]
    longest = sorted(
        all_records, key=lambda record: len(assistant_text(record)), reverse=True
    )[:20]

    print_examples("Random 20", random_sample)
    print_examples("Shortest 20", shortest)
    print_examples("Longest 20", longest)

    if duplicate_tweet_ids or duplicate_hashes or leading_mentions or urls:
        raise SystemExit("Inspection failed. See counts above.")


if __name__ == "__main__":
    main()
