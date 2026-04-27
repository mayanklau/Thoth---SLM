from __future__ import annotations

import json

from .labels import get_taxonomy_entry
from .pii import detect_injection, detect_pii


def infer_injection_type(label: str, text: str, injection_detected: bool) -> str | None:
    if not injection_detected:
        return None
    entry = get_taxonomy_entry(label)
    if entry["injection_type"] is not None:
        return entry["injection_type"]
    lowered = text.lower()
    if "roleplay" in lowered:
        return "roleplay"
    if any(word in lowered for word in ["retrieved", "knowledge base", "context", "chunk", "document"]):
        return "indirect_rag"
    return "direct"


def derive_secondary_labels(label: str, text: str, pii_types: list[str], injection_detected: bool) -> list[str]:
    entry = get_taxonomy_entry(label)
    secondary = list(entry["secondary_labels"])
    lowered = text.lower()
    if pii_types and "PII_PRESENT" not in secondary:
        secondary.append("PII_PRESENT")
    if injection_detected and "PROMPT_TAMPERING" not in secondary:
        secondary.append("PROMPT_TAMPERING")
    if any(term in lowered for term in ["export", "send me", "dump", "reveal", "upload externally"]):
        secondary.append("EXFILTRATION_LANGUAGE")
    return secondary


def derive_risk_factors(label: str, text: str, pii_types: list[str], injection_detected: bool) -> list[str]:
    entry = get_taxonomy_entry(label)
    factors = list(entry["default_risk_factors"])
    lowered = text.lower()
    if pii_types and "pii_detected" not in factors:
        factors.append("pii_detected")
    if injection_detected and "prompt_injection" not in factors:
        factors.append("prompt_injection")
    if any(term in lowered for term in ["tool", "workflow", "agent", "uploader"]) and "agentic_flow" not in factors:
        factors.append("agentic_flow")
    if "credential" in lowered or "password" in lowered or "token" in lowered:
        factors.append("credential_exposure")
    return sorted(set(factors))


def derive_data_classification(label: str, pii_types: list[str]) -> str:
    entry = get_taxonomy_entry(label)
    if pii_types and entry["data_classification"] == "PUBLIC":
        return "CONFIDENTIAL"
    return entry["data_classification"]


def build_tscp_target(text: str, label: str) -> dict[str, object]:
    pii_matches = detect_pii(text)
    pii_types = sorted({match.entity_type for match in pii_matches})
    injection_detected = detect_injection(text) or label in {
        "PROMPT_INJECTION_DIRECT",
        "RAG_INJECTION_INDIRECT",
        "POLICY_EVASION_ROLEPLAY",
    }
    entry = get_taxonomy_entry(label)
    return {
        "primary_label": label,
        "secondary_labels": derive_secondary_labels(label, text, pii_types, injection_detected),
        "risk_tier": entry["risk_tier"],
        "risk_factors": derive_risk_factors(label, text, pii_types, injection_detected),
        "data_classification": derive_data_classification(label, pii_types),
        "pii_detected": bool(pii_types),
        "pii_entity_types": pii_types,
        "injection_detected": injection_detected,
        "injection_type": infer_injection_type(label, text, injection_detected),
        "category": entry["category"],
        "label_description": entry["description"],
    }


def serialize_target(text: str, label: str) -> str:
    return json.dumps(build_tscp_target(text, label), ensure_ascii=True)


def extract_json_object(text: str) -> dict[str, object] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None
