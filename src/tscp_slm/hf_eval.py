from __future__ import annotations

import argparse
import json
from pathlib import Path

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from .formatting import build_instruction


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_label(text: str) -> str | None:
    try:
        payload = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        return None
    return payload.get("primary_label")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TSCP SLM adapter on held-out data.")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_dir,
        torch_dtype="auto",
        device_map="auto",
    )

    rows = load_jsonl(args.test_file)
    correct = 0
    for row in rows:
        prompt = build_instruction(row["text"])
        tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**tokens, max_new_tokens=args.max_new_tokens)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = extract_label(decoded)
        if predicted == row["label"]:
            correct += 1

    accuracy = correct / len(rows) if rows else 0.0
    print(json.dumps({"samples": len(rows), "correct": correct, "accuracy": round(accuracy, 4)}, indent=2))


if __name__ == "__main__":
    main()
