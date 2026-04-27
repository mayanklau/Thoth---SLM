from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .model import TSCPMiniModel
from .targets import build_tscp_target


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_label_metrics(gold: list[str], pred: list[str]) -> dict[str, object]:
    labels = sorted(set(gold) | set(pred))
    metrics: dict[str, dict[str, float]] = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": gold.count(label),
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    label_count = len(labels) or 1
    return {
        "per_label": metrics,
        "macro_precision": round(macro_precision / label_count, 4),
        "macro_recall": round(macro_recall / label_count, 4),
        "macro_f1": round(macro_f1 / label_count, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the TSCP mini model.")
    parser.add_argument("--data", required=True, help="Path to labeled JSONL.")
    parser.add_argument("--model", required=True, help="Path to model artifact.")
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    model = TSCPMiniModel.load(args.model)

    gold_labels: list[str] = []
    pred_labels: list[str] = []
    gold_risk: list[str] = []
    pred_risk: list[str] = []
    gold_classification: list[str] = []
    pred_classification: list[str] = []
    gold_injection: list[bool] = []
    pred_injection: list[bool] = []

    for row in rows:
        output = model.predict(row["text"])
        target = build_tscp_target(row["text"], row["label"])
        gold_labels.append(row["label"])
        pred_labels.append(output.label)
        gold_risk.append(str(target["risk_tier"]))
        pred_risk.append(output.risk_tier)
        gold_classification.append(str(target["data_classification"]))
        pred_classification.append(output.data_classification)
        gold_injection.append(bool(target["injection_detected"]))
        pred_injection.append(output.injection_detected)

    total = len(rows)
    correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
    accuracy = correct / total if total else 0.0
    risk_accuracy = sum(1 for g, p in zip(gold_risk, pred_risk) if g == p) / total if total else 0.0
    classification_accuracy = (
        sum(1 for g, p in zip(gold_classification, pred_classification) if g == p) / total if total else 0.0
    )
    injection_accuracy = sum(1 for g, p in zip(gold_injection, pred_injection) if g == p) / total if total else 0.0

    confusion: dict[str, Counter[str]] = {label: Counter() for label in sorted(set(gold_labels))}
    for g, p in zip(gold_labels, pred_labels):
        confusion[g][p] += 1

    report = {
        "samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "risk_accuracy": round(risk_accuracy, 4),
        "data_classification_accuracy": round(classification_accuracy, 4),
        "injection_accuracy": round(injection_accuracy, 4),
        "metrics": compute_label_metrics(gold_labels, pred_labels),
        "confusion": {label: dict(counts) for label, counts in confusion.items()},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
