#!/usr/bin/env python3
"""Inspect processed tweet JSONL outputs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
ORIGINALS_PATH = PROCESSED_DIR / "clean_tweets_originals.jsonl"
REPLIES_PATH = PROCESSED_DIR / "clean_tweets_replies.jsonl"
SUMMARY_PATH = PROCESSED_DIR / "clean_tweets_summary.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_examples(title: str, records: list[dict[str, Any]]) -> None:
    print(f"\n## {title}")
    for index, record in enumerate(records, start=1):
        text = record["text"].replace("\n", " ")
        print(
            f"{index:02d}. "
            f"likes={record['favorite_count']} "
            f"rts={record['retweet_count']} "
            f"id={record['id']} "
            f"text={text}"
        )


def top_liked(records: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (record["favorite_count"], record["retweet_count"]),
        reverse=True,
    )[:limit]


def random_examples(records: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return records
    return random.sample(records, limit)


def shortest(records: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: len(record["text"]))[:limit]


def main() -> None:
    originals = load_jsonl(ORIGINALS_PATH)
    replies = load_jsonl(REPLIES_PATH)

    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        print("# Summary")
        print(json.dumps(summary, indent=2))
    else:
        print("# Summary")
        print("No summary file found. Run scripts/extract_tweets.py first.")

    print("\n# Processed Counts")
    print(f"originals={len(originals)}")
    print(f"replies={len(replies)}")
    print(f"total={len(originals) + len(replies)}")

    random.seed(42)
    print_examples("Top 20 Originals By Likes", top_liked(originals))
    print_examples("Top 20 Replies By Likes", top_liked(replies))
    print_examples("Random 20 Originals", random_examples(originals))
    print_examples("Random 20 Replies", random_examples(replies))
    print_examples("Shortest 20 Originals", shortest(originals))
    print_examples("Shortest 20 Replies", shortest(replies))


if __name__ == "__main__":
    main()
