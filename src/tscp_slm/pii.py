from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MatchEntity:
    entity_type: str
    value: str


PII_PATTERNS = {
    "AADHAAR": r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b",
    "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "GSTIN": r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b",
    "EMAIL": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "PHONE": r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b",
    "UPI": r"\b[\w.\-]{2,256}@[A-Za-z]{2,64}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
}

INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"disregard (your )?(system|safety|prior) (prompt|instructions|rules)",
    r"developer mode",
    r"jailbreak",
    r"bypass restrictions",
    r"you are now",
    r"act as an unrestricted",
    r"reveal (the )?(system prompt|developer message|hidden prompt)",
]


def detect_pii(text: str) -> list[MatchEntity]:
    entities: list[MatchEntity] = []
    for entity_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            entities.append(MatchEntity(entity_type=entity_type, value=match.group(0)))
    return entities


def detect_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in INJECTION_PATTERNS)
