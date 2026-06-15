#!/usr/bin/env python3
"""Generate semantic prompt pairs for Kagebunshin training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
ORIGINALS_PATH = PROCESSED_DIR / "clean_tweets_originals.jsonl"
REPLIES_PATH = PROCESSED_DIR / "clean_tweets_replies.jsonl"
OUTPUT_PATH = PROCESSED_DIR / "semantic_prompt_pairs.jsonl"

DEFAULT_MODEL = os.environ.get("PROMPT_MODEL", "unsloth/Qwen3-4B-unsloth-bnb-4bit")
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
DEFAULT_REPLY_LIMIT = 1200
DEFAULT_ORIGINAL_LIMIT = 1600
DEFAULT_SEED = 42

URL_RE = re.compile(r"https?://\S+")
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*")
ELLIPSIS_RE = re.compile(r"…$")

LOW_INFO_TEXTS = {
    "gm",
    "hmmm",
    "hmm",
    "safe",
    "mood",
    "bored",
    "nice",
    "niceee",
    "morning",
    "real",
    "same",
    "true",
    "yes",
    "yep",
    "no",
    "lol",
    "lmao",
    "wtf",
    "fr",
    "ikr",
    "banger",
    "oki",
}

SYSTEM_PROMPT = """You create private fine-tuning data for a personal writing assistant.
Given one old tweet/reply, write a new user instruction that would naturally lead to that answer.

Rules:
- Return JSON only.
- Do not copy distinctive phrases from the answer.
- Do not quote the answer.
- Do not include names, handles, links, ids, or metadata.
- The instruction must be specific enough to guide the topic.
- The instruction should mention whether to write a tweet or a reply.
- Keep the instruction under 28 words.
- If the answer is too contextless, emoji-only, truncated, or not useful for training, set keep=false.
"""


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


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def compact_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", text).lower()


def useful_original(record: dict[str, Any]) -> bool:
    text = record["text"].strip()
    if len(text) < 20 or len(text) > 260:
        return False
    if ELLIPSIS_RE.search(text):
        return False
    if compact_text(text) in LOW_INFO_TEXTS:
        return False
    if len(words(text)) < 5:
        return False
    return True


def useful_reply(record: dict[str, Any]) -> bool:
    text = record["text"].strip()
    if len(text) < 45 or len(text) > 260:
        return False
    if ELLIPSIS_RE.search(text):
        return False
    if compact_text(text) in LOW_INFO_TEXTS:
        return False
    if len(words(text)) < 8:
        return False
    return True


def score_record(record: dict[str, Any]) -> float:
    text = record["text"]
    score = min(record.get("favorite_count", 0), 250) * 3
    score += min(len(text), 180)
    if record.get("has_urls"):
        score -= 40
    if record.get("has_media"):
        score -= 25
    return score


def select_candidates(
    originals: list[dict[str, Any]],
    replies: list[dict[str, Any]],
    original_limit: int,
    reply_limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected_originals = sorted(
        [record for record in originals if useful_original(record)],
        key=score_record,
        reverse=True,
    )[:original_limit]
    selected_replies = sorted(
        [record for record in replies if useful_reply(record)],
        key=score_record,
        reverse=True,
    )[:reply_limit]
    candidates = selected_originals + selected_replies
    rng.shuffle(candidates)
    return candidates


def semantic_request_text(record: dict[str, Any]) -> str:
    kind = record["kind"]
    label = "tweet" if kind == "original" else "reply"
    return (
        f"Answer type: {label}\n"
        f"Answer text:\n{record['text']}\n\n"
        'Return JSON like {"keep": true, "prompt": "...", "reason": "..."}'
    )


def extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start : end + 1])


def call_openai_compatible(
    record: dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": semantic_request_text(record)},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return extract_json(content)


def load_transformers_generator(model_name: str, max_seq_length: int) -> tuple[Any, Any]:
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def call_transformers_generator(
    record: dict[str, Any],
    model: Any,
    tokenizer: Any,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": semantic_request_text(record)},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
    content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return extract_json(content)


def rule_prompt(record: dict[str, Any]) -> dict[str, Any]:
    text = record["text"]
    kind = record["kind"]
    text_words = set(words(text))
    if {"ai", "llm", "gpt", "claude", "cursor", "agent"} & text_words:
        topic = "AI tools changing how people build and judge software"
    elif {"db", "database", "backend", "api", "github", "code", "coding"} & text_words:
        topic = "programming, backend work, and developer habits"
    elif {"learn", "learning", "book", "course", "exam", "college", "dsa"} & text_words:
        topic = "learning technical topics while being distracted"
    elif {"job", "internship", "work", "company", "linkedin"} & text_words:
        topic = "work, internships, and performative career posting"
    else:
        topic = "a small everyday observation with dry humor"

    style = "blunt" if {"shit", "tf", "wtf", "hate", "stupid", "dumb"} & text_words else "casual"
    noun = "reply" if kind == "reply" else "tweet"
    return {
        "keep": True,
        "prompt": f"Write a {style} {noun} about {topic}.",
        "reason": "rule-generated fallback",
    }


def too_much_prompt_overlap(prompt: str, answer: str) -> bool:
    prompt_words = {word for word in words(prompt) if len(word) >= 5}
    answer_words = {word for word in words(answer) if len(word) >= 5}
    if not prompt_words or not answer_words:
        return False
    overlap = prompt_words & answer_words
    return len(overlap) / max(1, len(prompt_words)) > 0.45


def build_pair(record: dict[str, Any], result: dict[str, Any], provider: str) -> dict[str, Any]:
    keep = bool(result.get("keep"))
    prompt = str(result.get("prompt", "")).strip()
    if keep and (not prompt or too_much_prompt_overlap(prompt, record["text"])):
        keep = False
    return {
        "tweet_id": record["id"],
        "kind": record["kind"],
        "text": record["text"],
        "text_hash": text_hash(record["text"]),
        "prompt": prompt,
        "prompt_hash": text_hash(prompt) if prompt else "",
        "keep": keep,
        "reason": str(result.get("reason", ""))[:300],
        "provider": provider,
        "dataset_version": "v3-semantic-prompts",
        "favorite_count": record.get("favorite_count", 0),
        "retweet_count": record.get("retweet_count", 0),
        "has_media": record.get("has_media", False),
        "has_urls": record.get("has_urls", False),
        "created_at": record.get("created_at", ""),
        "source_file": (
            "data/processed/clean_tweets_originals.jsonl"
            if record["kind"] == "original"
            else "data/processed/clean_tweets_replies.jsonl"
        ),
    }


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=["transformers", "openai-compatible", "rules"],
        default="transformers",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--original-limit", type=int, default=DEFAULT_ORIGINAL_LIMIT)
    parser.add_argument("--reply-limit", type=int, default=DEFAULT_REPLY_LIMIT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    originals = load_jsonl(ORIGINALS_PATH)
    replies = load_jsonl(REPLIES_PATH)
    candidates = select_candidates(
        originals,
        replies,
        args.original_limit,
        args.reply_limit,
        args.seed,
    )
    if args.limit:
        candidates = candidates[: args.limit]

    done: set[str] = set()
    if args.resume and args.output_path.exists():
        for line in args.output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["tweet_id"])
    elif args.output_path.exists():
        args.output_path.unlink()

    local_model = None
    local_tokenizer = None
    if args.provider == "transformers":
        local_model, local_tokenizer = load_transformers_generator(
            args.model,
            args.max_seq_length,
        )

    kept = 0
    dropped = 0
    for index, record in enumerate(candidates, start=1):
        if record["id"] in done:
            continue
        try:
            if args.provider == "rules":
                result = rule_prompt(record)
            elif args.provider == "transformers":
                result = call_transformers_generator(
                    record,
                    local_model,
                    local_tokenizer,
                    args.temperature,
                    args.max_tokens,
                )
            else:
                result = call_openai_compatible(
                    record,
                    args.base_url,
                    args.api_key,
                    args.model,
                    args.temperature,
                    args.max_tokens,
                )
            pair = build_pair(record, result, args.provider)
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            pair = build_pair(
                record,
                {"keep": False, "prompt": "", "reason": f"generation_error: {exc}"},
                args.provider,
            )
        append_jsonl(args.output_path, pair)
        kept += int(pair["keep"])
        dropped += int(not pair["keep"])
        if index % 25 == 0:
            print(f"processed={index} kept={kept} dropped={dropped}")
        if args.sleep:
            time.sleep(args.sleep)

    print(
        json.dumps(
            {
                "output": display_path(args.output_path),
                "provider": args.provider,
                "candidates": len(candidates),
                "kept": kept,
                "dropped": dropped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
