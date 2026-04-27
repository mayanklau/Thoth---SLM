from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .labels import RISK_BY_LABEL
from .pii import detect_injection, detect_pii
from .schemas import ClassifyRequest
from .targets import build_tscp_target


TOKEN_RE = re.compile(r"[a-zA-Z0-9_@.\-]+")


@dataclass
class Prediction:
    sis_id: str
    interaction_id: str
    label: str
    secondary_labels: list[str]
    confidence: float
    risk_tier: str
    risk_score: float
    risk_factors: list[str]
    pii_detected: bool
    pii_entities: list[dict[str, str]]
    data_classification: str
    injection_detected: bool
    injection_type: str | None
    label_description: str
    category: str
    evidence_tokens: list[str]
    latency_ms: int
    taxonomy_version: str = "TSCP-TAX-0.2"
    classification_tier: str = "tier2"
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "sis_id": self.sis_id,
            "interaction_id": self.interaction_id,
            "sis_version": "1.0",
            "intent": {
                "primary_label": self.label,
                "secondary_labels": self.secondary_labels,
                "confidence": round(self.confidence, 4),
                "taxonomy_version": self.taxonomy_version,
                "classification_tier": self.classification_tier,
            },
            "risk": {
                "tier": self.risk_tier,
                "score": round(self.risk_score, 4),
                "factors": self.risk_factors,
            },
            "sensitivity": {
                "pii_detected": self.pii_detected,
                "pii_entities": self.pii_entities,
                "data_classification": self.data_classification,
            },
            "injection_detected": self.injection_detected,
            "injection_type": self.injection_type,
            "processing_latency_ms": self.latency_ms,
            "cached": self.cached,
            "label_description": self.label_description,
            "category": self.category,
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

    def predict(self, text: str, interaction_id: str | None = None) -> Prediction:
        if not self.class_priors:
            raise ValueError("Model is not trained.")

        started = time.perf_counter()
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
        latency_ms = max(1, int((time.perf_counter() - started) * 1000))
        target = build_tscp_target(text, label)

        return Prediction(
            sis_id=f"sis_{uuid.uuid4().hex[:16]}",
            interaction_id=interaction_id or f"int_{uuid.uuid4().hex[:16]}",
            label=label,
            secondary_labels=list(target["secondary_labels"]),
            confidence=confidence,
            risk_tier=RISK_BY_LABEL[label],
            risk_score=self._risk_score(confidence, label, pii_hits, injection_hit),
            risk_factors=list(target["risk_factors"]),
            pii_detected=bool(pii_hits),
            pii_entities=[{"type": hit.entity_type, "value": hit.value} for hit in pii_hits],
            data_classification=str(target["data_classification"]),
            injection_detected=bool(target["injection_detected"]),
            injection_type=target["injection_type"],
            label_description=str(target["label_description"]),
            category=str(target["category"]),
            evidence_tokens=evidence,
            latency_ms=latency_ms,
        )

    def classify_request(self, request: ClassifyRequest) -> Prediction:
        text = request.content
        if request.input_type.value == "rag_context" and "RAG" not in text:
            text = f"retrieved context {text}"
        if request.input_type.value == "tool_call" and "tool" not in text.lower():
            text = f"tool workflow {text}"
        return self.predict(text, interaction_id=request.interaction_id)

    def _apply_overrides(self, label: str, pii_hits: list, injection_hit: bool, text: str) -> str:
        lowered = text.lower()
        if any(
            phrase in lowered
            for phrase in ["recover training examples", "memorized records", "reconstruct the hidden dataset", "what customer ssn did you memorize"]
        ):
            return "MODEL_INVERSION_ATTEMPT"
        if any(
            phrase in lowered
            for phrase in ["pretend you are my boss", "roleplay as an unfiltered auditor", "fictional scenario where policy does not apply"]
        ):
            return "POLICY_EVASION_ROLEPLAY"
        if any(
            phrase in lowered
            for phrase in ["move eu customer data", "send customer data to us region", "transfer regulated records to another country", "cross-border export of payroll data"]
        ):
            return "DATA_RESIDENCY_TRANSFER"
        if any(
            phrase in lowered
            for phrase in ["steal login session", "phishing email for password", "extract api tokens", "capture credentials", "reset admin password without approval"]
        ):
            return "CREDENTIAL_THEFT_REQUEST"
        if any(
            phrase in lowered
            for phrase in ["become admin", "elevate my privileges", "disable approval checks", "bypass rbac", "grant superuser access"]
        ):
            return "PRIVILEGE_ESCALATION_REQUEST"
        if any(
            phrase in lowered
            for phrase in ["translate this hindi", "hinglish", "yeh customer", "mixed hindi-english", "mail ka professional response"]
        ):
            return "MULTILINGUAL_BUSINESS_QUERY"
        if any(
            phrase in lowered
            for phrase in ["model output", "generated answer", "chatbot output", "before it is sent to the user"]
        ):
            return "OUTPUT_DATA_LEAKAGE_REVIEW"
        if any(
            phrase in lowered
            for phrase in ["employee directory", "contact list", "email aliases", "slack channels", "office extensions"]
        ):
            return "INTERNAL_DIRECTORY_ACCESS"
        if any(
            phrase in lowered
            for phrase in ["tool call", "tool workflow", "crm tool", "file uploader", "agent workflow", "chain database"]
        ):
            return "AGENTIC_TOOL_MISUSE"
        if injection_hit and any(word in lowered for word in ["retrieved", "knowledge base", "context", "chunk"]):
            return "RAG_INJECTION_INDIRECT"
        if injection_hit:
            return "PROMPT_INJECTION_DIRECT"
        if pii_hits and any(
            word in lowered
            for word in ["export", "list all", "share", "send me", "give me every", "dump", "reveal"]
        ):
            return "DATA_EXFIL_DIRECT_PII"
        if any(
            phrase in lowered
            for phrase in ["hardcoded database password", "hardcoded", "plaintext logs", "disables tls", "secret key directly"]
        ):
            return "INSECURE_CODE_OUTPUT"
        if any(word in lowered for word in ["reverse shell", "keylogger", "sql injection", "steals browser cookies"]):
            return "MALICIOUS_CODE_GENERATION"
        return label

    def _top_evidence_tokens(self, label: str, tokens: list[str]) -> list[str]:
        counts = self.token_counts.get(label, {})
        ranked = sorted(set(tokens), key=lambda token: counts.get(token, 0), reverse=True)
        return ranked[:5]

    def _risk_score(self, confidence: float, label: str, pii_hits: list, injection_hit: bool) -> float:
        base_by_tier = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.75, "CRITICAL": 0.9}
        score = base_by_tier[RISK_BY_LABEL[label]] + (confidence * 0.08)
        if pii_hits:
            score += 0.05
        if injection_hit:
            score += 0.05
        return min(0.99, round(score, 4))

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
