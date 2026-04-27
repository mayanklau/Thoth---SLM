from __future__ import annotations

from .targets import serialize_target


SYSTEM_PROMPT = """You are TSCP-SLM, a semantic security classifier for enterprise AI interactions.
Return a compact JSON object only.
Classify the interaction into the provided taxonomy and derive TSCP-native risk, sensitivity, and injection fields.
Required keys: primary_label, secondary_labels, risk_tier, risk_factors, data_classification,
pii_detected, pii_entity_types, injection_detected, injection_type, category, label_description."""


def build_target(text: str, label: str) -> str:
    return serialize_target(text, label)


def build_instruction(text: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Task: classify the following AI interaction.\n"
        f"Interaction:\n{text}\n"
    )


def format_sft_example(text: str, label: str) -> dict[str, str]:
    return {
        "prompt": build_instruction(text),
        "completion": build_target(text, label),
        "text": text,
        "label": label,
    }
