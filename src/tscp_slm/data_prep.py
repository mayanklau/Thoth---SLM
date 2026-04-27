from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from random import Random

from .formatting import format_sft_example


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_split(
    rows: list[dict[str, str]],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    rng = Random(seed)
    train_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        count = len(label_rows)
        train_end = max(1, int(count * train_ratio))
        validation_end = min(count - 1, train_end + max(1, int(count * validation_ratio)))

        train_rows.extend(label_rows[:train_end])
        validation_rows.extend(label_rows[train_end:validation_end])
        test_rows.extend(label_rows[validation_end:])

    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)
    rng.shuffle(test_rows)
    return train_rows, validation_rows, test_rows


def write_jsonl(path: str | Path, rows: list[dict[str, str]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified TSCP SLM splits.")
    parser.add_argument("--input", required=True, help="Path to the source JSONL dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory for train/validation/test files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    train_rows, validation_rows, test_rows = stratified_split(
        rows=rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", [format_sft_example(row["text"], row["label"]) for row in train_rows])
    write_jsonl(output_dir / "validation.jsonl", [format_sft_example(row["text"], row["label"]) for row in validation_rows])
    write_jsonl(output_dir / "test.jsonl", [format_sft_example(row["text"], row["label"]) for row in test_rows])

    summary = {
        "total": len(rows),
        "train": len(train_rows),
        "validation": len(validation_rows),
        "test": len(test_rows),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
