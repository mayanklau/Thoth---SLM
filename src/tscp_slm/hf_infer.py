from __future__ import annotations

import argparse

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from .formatting import build_instruction


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a TSCP SLM adapter.")
    parser.add_argument("--adapter-dir", required=True, help="Path to the trained adapter directory.")
    parser.add_argument("--text", required=True, help="Text to classify.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_dir,
        torch_dtype="auto",
        device_map="auto",
    )

    prompt = build_instruction(args.text)
    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**tokens, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
