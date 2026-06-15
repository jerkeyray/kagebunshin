#!/usr/bin/env python3
"""Build v3 train/valid files from generated semantic prompt pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PAIRS_PATH = PROCESSED_DIR / "semantic_prompt_pairs.jsonl"
TRAIN_PATH = PROCESSED_DIR / "train_semantic.jsonl"
VALID_PATH = PROCESSED_DIR / "valid_semantic.jsonl"
MANIFEST_PATH = PROCESSED_DIR / "semantic_dataset_manifest.jsonl"

SYSTEM_PROMPT = (
    "You are Kagebunshin, a private drafting assistant trained to write in "
    "Aditya's style. Generate drafts only. Do not claim to be Aditya."
)
DATASET_VERSION = "v3-semantic-prompts"
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


def training_record(pair: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["text"]},
        ],
        "metadata": {
            "tweet_id": pair["tweet_id"],
            "text_hash": pair["text_hash"],
            "prompt_hash": pair["prompt_hash"],
            "kind": pair["kind"],
            "split": split,
            "dataset_version": DATASET_VERSION,
            "source_file": pair["source_file"],
            "selected_reason": "semantic_prompt_keep",
            "prompt_provider": pair["provider"],
            "created_at": pair["created_at"],
            "favorite_count": pair["favorite_count"],
            "retweet_count": pair["retweet_count"],
            "has_media": pair["has_media"],
            "has_urls": pair["has_urls"],
        },
    }


def split_pairs(
    pairs: list[dict[str, Any]], valid_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_kind: dict[str, list[dict[str, Any]]] = {"original": [], "reply": []}
    for pair in pairs:
        by_kind[pair["kind"]].append(pair)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for records in by_kind.values():
        shuffled = list(records)
        rng.shuffle(shuffled)
        valid_count = max(1, round(len(shuffled) * valid_ratio))
        valid.extend(shuffled[:valid_count])
        train.extend(shuffled[valid_count:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-path", type=Path, default=PAIRS_PATH)
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--valid-path", type=Path, default=VALID_PATH)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-kept", type=int, default=1500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = [pair for pair in load_jsonl(args.pairs_path) if pair.get("keep")]
    if len(pairs) < args.min_kept:
        raise SystemExit(f"only {len(pairs)} kept pairs; expected at least {args.min_kept}")

    seen_hashes: set[str] = set()
    unique_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        if pair["text_hash"] in seen_hashes:
            continue
        seen_hashes.add(pair["text_hash"])
        unique_pairs.append(pair)

    train_pairs, valid_pairs = split_pairs(unique_pairs, args.valid_ratio, args.seed)
    train = [training_record(pair, "train") for pair in train_pairs]
    valid = [training_record(pair, "valid") for pair in valid_pairs]
    manifest = [{**pair, "split": "train"} for pair in train_pairs] + [
        {**pair, "split": "valid"} for pair in valid_pairs
    ]

    write_jsonl(args.train_path, train)
    write_jsonl(args.valid_path, valid)
    write_jsonl(args.manifest_path, manifest)
    print(
        json.dumps(
            {
                "dataset_version": DATASET_VERSION,
                "pairs": len(pairs),
                "unique_pairs": len(unique_pairs),
                "train": len(train),
                "valid": len(valid),
                "outputs": {
                    "train": display_path(args.train_path),
                    "valid": display_path(args.valid_path),
                    "manifest": display_path(args.manifest_path),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
