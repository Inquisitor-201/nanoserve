#!/usr/bin/env python3
import os
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import logging
from core import LLMService, SamplingConfig, EngineArgs

logging.basicConfig(level=logging.INFO)


def main():
    engine_args = EngineArgs(
        model_path="./models/Qwen3-0.6B",
        device="cuda",
        block_size=16,
    )

    llm_service = LLMService.from_engine_args(engine_args)

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
    sampling_config = SamplingConfig(temperature=0.4, max_new_tokens=400)
    generated_texts = llm_service.generate(
        prompts=formatted,
        sampling_config=sampling_config,
    )

    for prompt, text in zip(prompts, generated_texts):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {text}")


if __name__ == "__main__":
    main()
