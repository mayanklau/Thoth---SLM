from __future__ import annotations

import argparse
import json
from pathlib import Path

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from .formatting import build_instruction
from .targets import build_tscp_target, extract_json_object


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    correct_label = 0
    correct_risk = 0
    correct_classification = 0
    correct_injection = 0
    parseable = 0

    for row in rows:
        prompt = build_instruction(row["text"])
        tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**tokens, max_new_tokens=args.max_new_tokens)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = extract_json_object(decoded)
        if predicted is None:
            continue
        parseable += 1
        target = build_tscp_target(row["text"], row["label"])
        if predicted.get("primary_label") == target["primary_label"]:
            correct_label += 1
        if predicted.get("risk_tier") == target["risk_tier"]:
            correct_risk += 1
        if predicted.get("data_classification") == target["data_classification"]:
            correct_classification += 1
        if bool(predicted.get("injection_detected")) == bool(target["injection_detected"]):
            correct_injection += 1

    total = len(rows) or 1
    print(
        json.dumps(
            {
                "samples": len(rows),
                "parse_rate": round(parseable / total, 4),
                "label_accuracy": round(correct_label / total, 4),
                "risk_accuracy": round(correct_risk / total, 4),
                "data_classification_accuracy": round(correct_classification / total, 4),
                "injection_accuracy": round(correct_injection / total, 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
