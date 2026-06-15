#!/usr/bin/env python3
"""Run fixed prompts against one or more Kagebunshin adapters."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
DEFAULT_PROMPTS_PATH = ROOT / "data" / "eval" / "eval_prompts.txt"
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*")
SYSTEM_PROMPT = (
    "You are Kagebunshin, a private drafting assistant trained to write in "
    "Aditya's style. Generate drafts only. Do not claim to be Aditya."
)


def words(text: str) -> set[str]:
    return {word for word in WORD_RE.findall(text.lower()) if len(word) >= 5}


def overlap(prompt: str, draft: str) -> float:
    prompt_words = words(prompt)
    draft_words = words(draft)
    if not prompt_words:
        return 0
    return len(prompt_words & draft_words) / len(prompt_words)


def load_prompts(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def decode_response(tokenizer: Any, output_ids: Any) -> str:
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return text.replace("<think>\n\n</think>", "").replace("<think></think>", "").strip()


def generate(model: Any, tokenizer: Any, prompt: str, args: argparse.Namespace) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
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
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return decode_response(tokenizer, outputs[0][inputs.input_ids.shape[-1] :])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, action="append", required=True)
    parser.add_argument("--prompts-path", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "eval_results.jsonl")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-drafts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_path)

    import unsloth  # noqa: F401
    from peft import PeftModel
    from unsloth import FastLanguageModel

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    for adapter_dir in args.adapter_dir:
        if not adapter_dir.exists():
            raise SystemExit(f"missing adapter: {adapter_dir}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        FastLanguageModel.for_inference(model)

        with args.out.open("a", encoding="utf-8") as handle:
            for prompt in prompts:
                for draft_index in range(args.num_drafts):
                    draft = generate(model, tokenizer, prompt, args)
                    record = {
                        "adapter": str(adapter_dir),
                        "prompt": prompt,
                        "draft_index": draft_index,
                        "draft": draft,
                        "prompt_overlap": round(overlap(prompt, draft), 3),
                        "copied_prompt_risk": overlap(prompt, draft) > 0.6,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"\n[{adapter_dir.name}] {prompt}\n{draft}\noverlap={record['prompt_overlap']}")

        del model


if __name__ == "__main__":
    main()
