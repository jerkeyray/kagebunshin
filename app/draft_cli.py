#!/usr/bin/env python3
"""Interactive draft CLI for a trained Kagebunshin adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
DEFAULT_ADAPTER_DIR = ROOT / "adapters" / "kagebunshin-qwen3-4b-lora-v1"
SYSTEM_PROMPT = (
    "You are Kagebunshin, a private drafting assistant trained to write in "
    "Aditya's style. Generate drafts only. Do not claim to be Aditya."
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--max-seq-length", type=positive_int, default=1024)
    parser.add_argument("--max-new-tokens", type=positive_int, default=160)
    parser.add_argument("--temperature", type=positive_float, default=0.8)
    parser.add_argument("--top-p", type=positive_float, default=0.9)
    parser.add_argument("--num-drafts", type=positive_int, default=1)
    return parser.parse_args()


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
    generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
    return decode_response(tokenizer, generated_ids)


def main() -> None:
    args = parse_args()
    if not args.adapter_dir.exists():
        raise SystemExit(f"missing adapter directory: {args.adapter_dir}")

    import unsloth  # noqa: F401
    from peft import PeftModel
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    FastLanguageModel.for_inference(model)

    print("Kagebunshin draft CLI. Type a prompt, or /quit to exit.")
    while True:
        try:
            prompt = input("\n> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt in {"/q", "/quit", "quit", "exit"}:
            break
        for index in range(args.num_drafts):
            draft = generate(model, tokenizer, prompt, args)
            if args.num_drafts > 1:
                print(f"\n[{index + 1}] {draft}")
            else:
                print(f"\n{draft}")


if __name__ == "__main__":
    main()
