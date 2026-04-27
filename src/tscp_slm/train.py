from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import TSCPMiniModel


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TSCP mini model.")
    parser.add_argument("--data", required=True, help="Path to training JSONL.")
    parser.add_argument("--output", required=True, help="Path to model artifact JSON.")
    args = parser.parse_args()

    samples = load_jsonl(args.data)
    model = TSCPMiniModel()
    model.train(samples)
    model.save(args.output)
    print(f"trained {len(samples)} samples -> {args.output}")


if __name__ == "__main__":
    main()
