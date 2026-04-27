from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .model import TSCPMiniModel


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the TSCP mini model.")
    parser.add_argument("--data", required=True, help="Path to labeled JSONL.")
    parser.add_argument("--model", required=True, help="Path to model artifact.")
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    model = TSCPMiniModel.load(args.model)

    gold = []
    pred = []
    for row in rows:
        output = model.predict(row["text"])
        gold.append(row["label"])
        pred.append(output.label)

    total = len(rows)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    accuracy = correct / total if total else 0.0

    labels = sorted(set(gold))
    confusion: dict[str, Counter[str]] = {label: Counter() for label in labels}
    for g, p in zip(gold, pred):
        confusion[g][p] += 1

    print(json.dumps(
        {
            "samples": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "labels": labels,
            "confusion": {label: dict(counts) for label, counts in confusion.items()},
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
