#!/usr/bin/env python3
"""Build chat-format training data from cleaned original tweets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
ORIGINALS_PATH = PROCESSED_DIR / "clean_tweets_originals.jsonl"
CANDIDATES_PATH = PROCESSED_DIR / "dataset_candidates.jsonl"
MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.jsonl"
TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
VALID_PATH = PROCESSED_DIR / "valid.jsonl"

SYSTEM_PROMPT = (
    "You are Kagebunshin, a private drafting assistant trained to write in "
    "Aditya's style."
)
USER_PROMPT = "Write a tweet in Aditya's style."
DATASET_VERSION = "v0-originals-all"
DEFAULT_VALID_RATIO = 0.1
DEFAULT_SEED = 42


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidate_record(tweet: dict[str, Any]) -> dict[str, Any]:
    text = tweet["text"]
    return {
        "tweet_id": tweet["id"],
        "text_hash": text_hash(text),
        "text": text,
        "kind": tweet["kind"],
        "created_at": tweet["created_at"],
        "favorite_count": tweet["favorite_count"],
        "retweet_count": tweet["retweet_count"],
        "has_media": tweet["has_media"],
        "has_urls": tweet["has_urls"],
        "source_file": "data/processed/clean_tweets_originals.jsonl",
        "dataset_version": DATASET_VERSION,
        "selection_status": "selected",
        "selected_reason": "all_clean_originals",
    }


def training_record(candidate: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": candidate["text"]},
        ],
        "metadata": {
            "tweet_id": candidate["tweet_id"],
            "text_hash": candidate["text_hash"],
            "kind": candidate["kind"],
            "split": split,
            "dataset_version": candidate["dataset_version"],
            "source_file": candidate["source_file"],
            "selected_reason": candidate["selected_reason"],
            "created_at": candidate["created_at"],
            "favorite_count": candidate["favorite_count"],
            "retweet_count": candidate["retweet_count"],
            "has_media": candidate["has_media"],
            "has_urls": candidate["has_urls"],
        },
    }


def manifest_record(candidate: dict[str, Any], split: str) -> dict[str, Any]:
    record = dict(candidate)
    record["split"] = split
    return record


def split_candidates(
    candidates: list[dict[str, Any]], valid_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(candidates)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    valid_count = max(1, round(len(shuffled) * valid_ratio))
    valid = shuffled[:valid_count]
    train = shuffled[valid_count:]
    return train, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not 0 < args.valid_ratio < 1:
        raise SystemExit("--valid-ratio must be between 0 and 1")

    originals = load_jsonl(ORIGINALS_PATH)
    candidates = [candidate_record(tweet) for tweet in originals]
    train_candidates, valid_candidates = split_candidates(
        candidates, args.valid_ratio, args.seed
    )

    train = [training_record(candidate, "train") for candidate in train_candidates]
    valid = [training_record(candidate, "valid") for candidate in valid_candidates]
    manifest = [
        manifest_record(candidate, "train") for candidate in train_candidates
    ] + [manifest_record(candidate, "valid") for candidate in valid_candidates]

    write_jsonl(CANDIDATES_PATH, candidates)
    write_jsonl(MANIFEST_PATH, manifest)
    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VALID_PATH, valid)

    summary = {
        "dataset_version": DATASET_VERSION,
        "source": str(ORIGINALS_PATH.relative_to(ROOT)),
        "candidates": len(candidates),
        "train": len(train),
        "valid": len(valid),
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "outputs": {
            "candidates": str(CANDIDATES_PATH.relative_to(ROOT)),
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "train": str(TRAIN_PATH.relative_to(ROOT)),
            "valid": str(VALID_PATH.relative_to(ROOT)),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
