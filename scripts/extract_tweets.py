#!/usr/bin/env python3
"""Extract and clean tweets from a Twitter/X archive."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "raw" / "twitter-archive" / "tweets.js"
PROCESSED_DIR = ROOT / "data" / "processed"
ORIGINALS_PATH = PROCESSED_DIR / "clean_tweets_originals.jsonl"
REPLIES_PATH = PROCESSED_DIR / "clean_tweets_replies.jsonl"
SUMMARY_PATH = PROCESSED_DIR / "clean_tweets_summary.json"

ARCHIVE_PREFIX = "window.YTD.tweets.part0 = "
URL_RE = re.compile(r"https?://\S+")
LEADING_MENTION_RE = re.compile(r"^(?:@\w+\s+)+")
WHITESPACE_RE = re.compile(r"\s+")


def load_archive(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if raw.startswith(ARCHIVE_PREFIX):
        raw = raw[len(ARCHIVE_PREFIX) :]
    return json.loads(raw)


def parse_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def normalize_text(text: str) -> str:
    text = URL_RE.sub("", text)
    text = LEADING_MENTION_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def has_urls(tweet: dict[str, Any], raw_text: str) -> bool:
    entities = tweet.get("entities") or {}
    urls = entities.get("urls") or []
    media = entities.get("media") or []
    return bool(urls or media or URL_RE.search(raw_text))


def has_media(tweet: dict[str, Any]) -> bool:
    entities = tweet.get("entities") or {}
    extended = tweet.get("extended_entities") or {}
    return bool((entities.get("media") or []) or (extended.get("media") or []))


def is_reply(tweet: dict[str, Any]) -> bool:
    return bool(tweet.get("in_reply_to_status_id_str"))


def is_retweet(tweet: dict[str, Any], raw_text: str) -> bool:
    return bool(
        tweet.get("retweeted") is True
        or raw_text.startswith("RT @")
        or tweet.get("retweeted_status")
    )


def drop_reason(tweet: dict[str, Any], raw_text: str, clean_text: str) -> str | None:
    if is_retweet(tweet, raw_text):
        return "retweet"
    if not clean_text:
        return "empty_after_cleaning"
    if len(clean_text.replace(" ", "")) < 3:
        return "too_short"
    if clean_text.startswith("@") and all(part.startswith("@") for part in clean_text.split()):
        return "mentions_only"
    return None


def build_record(tweet: dict[str, Any], raw_text: str, clean_text: str) -> dict[str, Any]:
    kind = "reply" if is_reply(tweet) else "original"
    return {
        "id": tweet.get("id_str") or "",
        "created_at": tweet.get("created_at") or "",
        "text": clean_text,
        "raw_text": raw_text,
        "kind": kind,
        "favorite_count": parse_int(tweet.get("favorite_count")),
        "retweet_count": parse_int(tweet.get("retweet_count")),
        "has_media": has_media(tweet),
        "has_urls": has_urls(tweet, raw_text),
        "source": tweet.get("source") or "",
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    archive = load_archive(INPUT_PATH)
    originals: list[dict[str, Any]] = []
    replies: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    seen_texts: set[str] = set()

    for item in archive:
        tweet = item.get("tweet") or {}
        raw_text = tweet.get("full_text") or ""
        clean_text = normalize_text(raw_text)

        reason = drop_reason(tweet, raw_text, clean_text)
        if reason:
            dropped[reason] += 1
            continue

        duplicate_key = clean_text.casefold()
        if duplicate_key in seen_texts:
            dropped["duplicate_clean_text"] += 1
            continue
        seen_texts.add(duplicate_key)

        record = build_record(tweet, raw_text, clean_text)
        if record["kind"] == "reply":
            replies.append(record)
        else:
            originals.append(record)

    write_jsonl(ORIGINALS_PATH, originals)
    write_jsonl(REPLIES_PATH, replies)

    summary = {
        "input": str(INPUT_PATH.relative_to(ROOT)),
        "raw_tweet_count": len(archive),
        "kept_original_count": len(originals),
        "kept_reply_count": len(replies),
        "kept_total_count": len(originals) + len(replies),
        "dropped_total_count": sum(dropped.values()),
        "dropped_by_reason": dict(sorted(dropped.items())),
        "outputs": {
            "originals": str(ORIGINALS_PATH.relative_to(ROOT)),
            "replies": str(REPLIES_PATH.relative_to(ROOT)),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
