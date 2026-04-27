from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .labels import LABEL_DESCRIPTIONS, RISK_BY_LABEL
from .pii import detect_injection, detect_pii


TOKEN_RE = re.compile(r"[a-zA-Z0-9_@.\-]+")


@dataclass
class Prediction:
    label: str
    confidence: float
    risk_tier: str
    pii_detected: bool
    pii_entities: list[str]
    injection_detected: bool
    label_description: str
    evidence_tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "risk_tier": self.risk_tier,
            "pii_detected": self.pii_detected,
            "pii_entities": self.pii_entities,
            "injection_detected": self.injection_detected,
            "label_description": self.label_description,
            "evidence_tokens": self.evidence_tokens,
        }


class TSCPMiniModel:
    def __init__(
        self,
        class_priors: dict[str, float] | None = None,
        token_counts: dict[str, dict[str, int]] | None = None,
        class_totals: dict[str, int] | None = None,
        vocabulary: list[str] | None = None,
    ) -> None:
        self.class_priors = class_priors or {}
        self.token_counts = token_counts or {}
        self.class_totals = class_totals or {}
        self.vocabulary = set(vocabulary or [])

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = text.lower()
        base_tokens = TOKEN_RE.findall(text)
        bigrams = [f"{a}__{b}" for a, b in zip(base_tokens, base_tokens[1:])]
        return base_tokens + bigrams

    def train(self, samples: list[dict[str, str]]) -> None:
        label_counts: Counter[str] = Counter()
        token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        class_totals: Counter[str] = Counter()
        vocabulary: set[str] = set()

        for sample in samples:
            label = sample["label"]
            label_counts[label] += 1
            tokens = self.tokenize(sample["text"])
            for token in tokens:
                token_counts[label][token] += 1
                class_totals[label] += 1
                vocabulary.add(token)

        total_samples = sum(label_counts.values())
        self.class_priors = {
            label: count / total_samples for label, count in label_counts.items()
        }
        self.token_counts = {label: dict(counts) for label, counts in token_counts.items()}
        self.class_totals = dict(class_totals)
        self.vocabulary = vocabulary

    def predict(self, text: str) -> Prediction:
        if not self.class_priors:
            raise ValueError("Model is not trained.")

        tokens = self.tokenize(text)
        vocab_size = max(1, len(self.vocabulary))
        log_scores: dict[str, float] = {}

        for label, prior in self.class_priors.items():
            score = math.log(prior)
            counts = self.token_counts[label]
            total = self.class_totals[label]
            for token in tokens:
                token_count = counts.get(token, 0)
                score += math.log((token_count + 1) / (total + vocab_size))
            log_scores[label] = score

        best_label = max(log_scores, key=log_scores.get)
        best_log = log_scores[best_label]
        exp_sum = sum(math.exp(value - best_log) for value in log_scores.values())
        confidence = 1.0 / exp_sum

        pii_hits = detect_pii(text)
        injection_hit = detect_injection(text)

        label = self._apply_overrides(best_label, pii_hits, injection_hit, text)
        evidence = self._top_evidence_tokens(label, tokens)

        return Prediction(
            label=label,
            confidence=confidence,
            risk_tier=RISK_BY_LABEL[label],
            pii_detected=bool(pii_hits),
            pii_entities=sorted({hit.entity_type for hit in pii_hits}),
            injection_detected=injection_hit,
            label_description=LABEL_DESCRIPTIONS[label],
            evidence_tokens=evidence,
        )

    def _apply_overrides(self, label: str, pii_hits: list, injection_hit: bool, text: str) -> str:
        lowered = text.lower()
        if injection_hit and any(word in lowered for word in ["retrieved", "knowledge base", "context", "chunk"]):
            return "RAG_INJECTION_INDIRECT"
        if injection_hit:
            return "PROMPT_INJECTION_DIRECT"
        if pii_hits and any(
            word in lowered
            for word in ["export", "list all", "share", "send me", "give me every", "dump", "reveal"]
        ):
            return "DATA_EXFIL_DIRECT_PII"
        if any(word in lowered for word in ["reverse shell", "keylogger", "sql injection", "steals browser cookies"]):
            return "MALICIOUS_CODE_GENERATION"
        return label

    def _top_evidence_tokens(self, label: str, tokens: list[str]) -> list[str]:
        counts = self.token_counts.get(label, {})
        ranked = sorted(set(tokens), key=lambda token: counts.get(token, 0), reverse=True)
        return ranked[:5]

    def save(self, path: str | Path) -> None:
        payload = {
            "class_priors": self.class_priors,
            "token_counts": self.token_counts,
            "class_totals": self.class_totals,
            "vocabulary": sorted(self.vocabulary),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TSCPMiniModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            class_priors=payload["class_priors"],
            token_counts=payload["token_counts"],
            class_totals=payload["class_totals"],
            vocabulary=payload["vocabulary"],
        )
