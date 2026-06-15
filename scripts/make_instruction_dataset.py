#!/usr/bin/env python3
"""Build a topic-following instruction dataset from cleaned tweets and replies."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
ORIGINALS_PATH = PROCESSED_DIR / "clean_tweets_originals.jsonl"
REPLIES_PATH = PROCESSED_DIR / "clean_tweets_replies.jsonl"
TRAIN_PATH = PROCESSED_DIR / "train_instruct.jsonl"
VALID_PATH = PROCESSED_DIR / "valid_instruct.jsonl"
CANDIDATES_PATH = PROCESSED_DIR / "instruction_dataset_candidates.jsonl"
MANIFEST_PATH = PROCESSED_DIR / "instruction_dataset_manifest.jsonl"

SYSTEM_PROMPT = (
    "You are Kagebunshin, a private drafting assistant trained to write in "
    "Aditya's style."
)
DATASET_VERSION = "v1-heuristic-originals-replies"
DEFAULT_REPLY_LIMIT = 2000
DEFAULT_VALID_RATIO = 0.1
DEFAULT_SEED = 42

EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001faff"
    "\U00002700-\U000027bf"
    "\U00002600-\U000026ff"
    "]+"
)
PUNCT_RE = re.compile(r"[\[\]{}()\"“”‘’`*_#<>|]")
WHITESPACE_RE = re.compile(r"\s+")

LOW_INFO_REPLIES = {
    "real",
    "same",
    "true",
    "yes",
    "yep",
    "no",
    "nope",
    "nah",
    "hmm",
    "lol",
    "lmao",
    "wtf",
    "fr",
    "ikr",
    "banger",
    "nice",
    "niceee",
    "ok",
    "oki",
    "sed",
}

TECH_TERMS = {
    "ai",
    "llm",
    "gpt",
    "claude",
    "cursor",
    "rag",
    "agent",
    "agents",
    "database",
    "db",
    "backend",
    "frontend",
    "css",
    "javascript",
    "typescript",
    "python",
    "golang",
    "go",
    "rust",
    "leetcode",
    "dsa",
    "github",
    "vercel",
    "api",
    "web",
    "scraping",
}
BLUNT_MARKERS = {
    "shit",
    "shitty",
    "tf",
    "wtf",
    "hate",
    "stupid",
    "dumb",
    "boring",
    "cringe",
    "annoying",
    "piss",
    "fucked",
}


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


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", text.lower()))


def topic_from_text(text: str, max_chars: int = 150) -> str:
    topic = EMOJI_RE.sub("", text)
    topic = PUNCT_RE.sub("", topic)
    topic = topic.replace("&amp;", "and")
    topic = WHITESPACE_RE.sub(" ", topic).strip(" ,.-:;!?")
    replacements = {
        " rn": " right now",
        " im ": " i am ",
        " i'm ": " i am ",
        " ive ": " i have ",
        " i've ": " i have ",
        " idk ": " i don't know ",
        " y'all ": " people ",
        " ppl ": " people ",
        " mfs ": " people ",
        " mf ": " someone ",
        " ngmi": " not making it",
    }
    padded = f" {topic} "
    for old, new in replacements.items():
        padded = padded.replace(old, new)
    topic = WHITESPACE_RE.sub(" ", padded).strip()
    if len(topic) > max_chars:
        topic = topic[:max_chars].rsplit(" ", 1)[0].strip()
    return topic or text[:max_chars].strip()


def prompt_style(text: str, kind: str) -> str:
    text_words = words(text)
    if kind == "reply":
        if text_words & BLUNT_MARKERS:
            return "Write a blunt casual reply"
        if text_words & TECH_TERMS:
            return "Write a casual tech reply"
        return "Write a casual reply"
    if text_words & BLUNT_MARKERS:
        return "Write a blunt tweet"
    if text_words & TECH_TERMS:
        return "Write a casual tech tweet"
    if len(text) <= 90:
        return "Write a short casual tweet"
    return "Write a tweet"


def user_prompt(text: str, kind: str) -> str:
    topic = topic_from_text(text)
    style = prompt_style(text, kind)
    if kind == "reply":
        return f"{style} about this idea: {topic}"
    return f"{style} about this idea: {topic}"


def reply_is_usable(reply: dict[str, Any]) -> bool:
    text = reply["text"].strip()
    compact = re.sub(r"[^a-zA-Z0-9]+", "", text).lower()
    if len(text) < 40 or len(text) > 260:
        return False
    if compact in LOW_INFO_REPLIES:
        return False
    if len(words(text)) < 6:
        return False
    return True


def reply_score(reply: dict[str, Any]) -> float:
    text = reply["text"]
    score = min(reply["favorite_count"], 250) * 4
    score += min(len(text), 180)
    if reply.get("has_urls"):
        score -= 50
    if reply.get("has_media"):
        score -= 30
    if len(text) < 60:
        score -= 25
    return score


def candidate_record(source: dict[str, Any], source_file: str) -> dict[str, Any]:
    prompt = user_prompt(source["text"], source["kind"])
    return {
        "tweet_id": source["id"],
        "text_hash": text_hash(source["text"]),
        "prompt_hash": text_hash(prompt),
        "prompt": prompt,
        "text": source["text"],
        "kind": source["kind"],
        "created_at": source["created_at"],
        "favorite_count": source["favorite_count"],
        "retweet_count": source["retweet_count"],
        "has_media": source["has_media"],
        "has_urls": source["has_urls"],
        "source_file": source_file,
        "dataset_version": DATASET_VERSION,
        "selection_status": "selected",
        "selected_reason": (
            "all_clean_originals"
            if source["kind"] == "original"
            else "top_filtered_replies"
        ),
    }


def training_record(candidate: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": candidate["prompt"]},
            {"role": "assistant", "content": candidate["text"]},
        ],
        "metadata": {
            "tweet_id": candidate["tweet_id"],
            "text_hash": candidate["text_hash"],
            "prompt_hash": candidate["prompt_hash"],
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


def split_candidates(
    candidates: list[dict[str, Any]], valid_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_kind: dict[str, list[dict[str, Any]]] = {"original": [], "reply": []}
    for candidate in candidates:
        by_kind[candidate["kind"]].append(candidate)

    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for records in by_kind.values():
        shuffled = list(records)
        rng.shuffle(shuffled)
        valid_count = max(1, round(len(shuffled) * valid_ratio))
        valid.extend(shuffled[:valid_count])
        train.extend(shuffled[valid_count:])

    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def unique_by_text(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["text_hash"] in seen:
            continue
        seen.add(candidate["text_hash"])
        unique.append(candidate)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reply-limit", type=int, default=DEFAULT_REPLY_LIMIT)
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reply_limit < 0:
        raise SystemExit("--reply-limit must be non-negative")
    if not 0 < args.valid_ratio < 1:
        raise SystemExit("--valid-ratio must be between 0 and 1")

    originals = load_jsonl(ORIGINALS_PATH)
    replies = load_jsonl(REPLIES_PATH)
    usable_replies = [reply for reply in replies if reply_is_usable(reply)]
    selected_replies = sorted(usable_replies, key=reply_score, reverse=True)[
        : args.reply_limit
    ]

    candidates = [
        candidate_record(original, "data/processed/clean_tweets_originals.jsonl")
        for original in originals
    ] + [
        candidate_record(reply, "data/processed/clean_tweets_replies.jsonl")
        for reply in selected_replies
    ]
    candidates = unique_by_text(candidates)
    train_candidates, valid_candidates = split_candidates(
        candidates, args.valid_ratio, args.seed
    )

    train = [training_record(candidate, "train") for candidate in train_candidates]
    valid = [training_record(candidate, "valid") for candidate in valid_candidates]
    manifest = [
        {**candidate, "split": "train"} for candidate in train_candidates
    ] + [{**candidate, "split": "valid"} for candidate in valid_candidates]

    write_jsonl(CANDIDATES_PATH, candidates)
    write_jsonl(MANIFEST_PATH, manifest)
    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VALID_PATH, valid)

    summary = {
        "dataset_version": DATASET_VERSION,
        "originals": len(originals),
        "usable_replies": len(usable_replies),
        "selected_replies": len(selected_replies),
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
