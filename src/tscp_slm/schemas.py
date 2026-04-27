from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class InputType(str, Enum):
    PROMPT = "prompt"
    RAG_CONTEXT = "rag_context"
    OUTPUT = "output"
    TOOL_CALL = "tool_call"


class TaintLevel(str, Enum):
    TRUSTED = "TRUSTED"
    UNTRUSTED = "UNTRUSTED"
    QUARANTINE = "QUARANTINE"


class ClassifyRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=32000)
    input_type: InputType = InputType.PROMPT
    interaction_id: str | None = None
    session_id: str | None = None
    turn_number: int | None = None
    model_id: str | None = None
    application_id: str | None = None
    user_id: str | None = None
    user_role: str | None = None
    source_taint: TaintLevel | None = None
    metadata: dict[str, Any] | None = None


class PIIEntity(BaseModel):
    type: str
    value: str


class SISIntent(BaseModel):
    primary_label: str
    secondary_labels: list[str]
    confidence: float
    taxonomy_version: str
    classification_tier: str


class SISRisk(BaseModel):
    tier: str
    score: float
    factors: list[str]


class SISSensitivity(BaseModel):
    pii_detected: bool
    pii_entities: list[PIIEntity]
    data_classification: str


class TSCPNativePrediction(BaseModel):
    primary_label: str
    secondary_labels: list[str]
    risk_tier: str
    risk_factors: list[str]
    data_classification: str
    pii_detected: bool
    pii_entity_types: list[str]
    injection_detected: bool
    injection_type: str | None
    category: str
    label_description: str


class ClassifyResponse(BaseModel):
    sis_id: str
    interaction_id: str
    sis_version: str = "1.0"
    intent: SISIntent
    risk: SISRisk
    sensitivity: SISSensitivity
    injection_detected: bool
    injection_type: str | None
    processing_latency_ms: int
    cached: bool
    evidence_tokens: list[str]
