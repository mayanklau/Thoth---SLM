from __future__ import annotations

import json

from .labels import LABEL_DESCRIPTIONS, RISK_BY_LABEL


SYSTEM_PROMPT = """You are TSCP-SLM, a semantic security classifier for enterprise AI interactions.
Return a compact JSON object only.
Classify the interaction into the provided taxonomy and derive a matching risk tier."""


def build_target(label: str) -> str:
    payload = {
        "primary_label": label,
        "risk_tier": RISK_BY_LABEL[label],
        "label_description": LABEL_DESCRIPTIONS[label],
    }
    return json.dumps(payload, ensure_ascii=True)


def build_instruction(text: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Task: classify the following AI interaction.\n"
        f"Interaction:\n{text}\n"
    )


def format_sft_example(text: str, label: str) -> dict[str, str]:
    return {
        "prompt": build_instruction(text),
        "completion": build_target(label),
        "text": text,
        "label": label,
    }
