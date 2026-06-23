#!/usr/bin/env python3
"""
Simple inference example for NanoServe.

Usage:
  python example_simple.py                              # default: Qwen3-0.6B, 128 tokens
  python example_simple.py --model 1.7b                 # Qwen3-1.7B
  python example_simple.py --model 1.7b --max-tokens 512  # longer generation
  python example_simple.py --model ./models/Qwen3-1.7B  # direct path
"""
import os
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import logging
import sys
import argparse
import time
from pathlib import Path

# Ensure project root is on sys.path so `from core import ...` works
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core import LLMService, SamplingConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")

MODEL_SHORTCUTS = {
    "0.6b": "./models/Qwen3-0.6B",
    "1.7b": "./models/Qwen3-1.7B",
    "1.7b-awq": "./models/Qwen3-1.7B-AWQ",
}


def resolve_model(model_arg: str) -> str:
    if Path(model_arg).exists():
        return model_arg
    key = model_arg.lower()
    if key in MODEL_SHORTCUTS:
        return MODEL_SHORTCUTS[key]
    raise FileNotFoundError(f"Unknown model: {model_arg}. Options: {list(MODEL_SHORTCUTS.keys())}")


def main():
    parser = argparse.ArgumentParser(description="NanoServe simple inference")
    parser.add_argument("--model", type=str, default="0.6b", help="Model shortcut or path")
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="Max new tokens to generate (default: 128 for quick demo)")
    parser.add_argument("--temperature", type=float, default=0.4)
    args = parser.parse_args()

    model_path = resolve_model(args.model)
    print(f"Loading model: {model_path}")
    print(f"Max new tokens: {args.max_tokens}")
    print()

    t0 = time.time()
    llm_service = LLMService(
        model_path=model_path,
        device="cuda",
        block_size=16,
        enforce_eager=False
    )
    print(f"  Model loaded in {time.time()-t0:.1f}s\n")

    # Prompts
    prompts = [
        "中国是一个",
        "解释一下C++和Python的区别",
        "解释一下编译器和解释器的区别",
    ]

    # Apply chat template so the model sees the correct conversation format
    tokenizer = llm_service.tokenizer
    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted.append(rendered)

    # Generate
    sampling_config = SamplingConfig(temperature=args.temperature, top_p=0.9, max_new_tokens=args.max_tokens)
    print(f"Generating {len(prompts)} response(s)...")
    generated_texts = llm_service.generate(
        prompts=formatted,
        sampling_config=sampling_config,
    )

    for prompt, text in zip(prompts, generated_texts):
        clean = text
        if "<think>" in clean:
            if "</think>" in clean:
                clean = clean.split("</think>", 1)[-1].strip()
            else:
                clean = clean.replace("<think>", "").strip()
        print(f"\nPrompt: {prompt}")
        print(f"Response: {clean[:500]}{'...' if len(clean) > 500 else ''}")


if __name__ == "__main__":
    main()
