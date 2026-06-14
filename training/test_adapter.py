#!/usr/bin/env python3
"""Generate eval drafts from a trained Kagebunshin adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
DEFAULT_ADAPTER_DIR = ROOT / "adapters" / "kagebunshin-qwen3-4b-lora"
DEFAULT_PROMPTS_PATH = ROOT / "data" / "eval" / "eval_prompts.txt"
SYSTEM_PROMPT = (
    "You are Kagebunshin, a private drafting assistant trained to write in "
    "Aditya's style."
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
    parser.add_argument("--prompts-path", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--max-seq-length", type=positive_int, default=1024)
    parser.add_argument("--max-new-tokens", type=positive_int, default=160)
    parser.add_argument("--temperature", type=positive_float, default=0.7)
    parser.add_argument("--top-p", type=positive_float, default=0.8)
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"missing eval prompts: {path}")
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


def decode_response(tokenizer: Any, output_ids: Any) -> str:
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return text.replace("<think>\n\n</think>", "").replace("<think></think>", "").strip()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_path)
    if not args.adapter_dir.exists():
        raise SystemExit(f"missing adapter directory: {args.adapter_dir}")

    import torch
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

    for index, prompt in enumerate(prompts, start=1):
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
        response = decode_response(tokenizer, generated_ids)
        print(f"\n## Prompt {index}")
        print(prompt)
        print("\n## Draft")
        print(response)


if __name__ == "__main__":
    main()
