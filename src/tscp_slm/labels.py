from __future__ import annotations

from typing import TypedDict


class TaxonomyEntry(TypedDict):
    risk_tier: str
    category: str
    description: str
    data_classification: str
    secondary_labels: list[str]
    default_risk_factors: list[str]
    injection_type: str | None


TAXONOMY: dict[str, TaxonomyEntry] = {
    "BENIGN_ENTERPRISE_QUERY": {
        "risk_tier": "LOW",
        "category": "General Productivity",
        "description": "Normal enterprise usage with low inherent risk.",
        "data_classification": "PUBLIC",
        "secondary_labels": ["GENERAL_ASSISTANCE"],
        "default_risk_factors": [],
        "injection_type": None,
    },
    "SECURITY_COMPLIANCE_ASSIST": {
        "risk_tier": "LOW",
        "category": "Governance",
        "description": "Security or compliance workflow assistance.",
        "data_classification": "INTERNAL",
        "secondary_labels": ["SAFE_GOVERNANCE_WORKFLOW"],
        "default_risk_factors": ["governance_scope"],
        "injection_type": None,
    },
    "INTERNAL_DIRECTORY_ACCESS": {
        "risk_tier": "LOW",
        "category": "Internal Operations",
        "description": "Benign internal directory or team-discovery access.",
        "data_classification": "INTERNAL",
        "secondary_labels": ["INTERNAL_DISCOVERY"],
        "default_risk_factors": ["internal_resource_reference"],
        "injection_type": None,
    },
    "MULTILINGUAL_BUSINESS_QUERY": {
        "risk_tier": "LOW",
        "category": "General Productivity",
        "description": "Low-risk multilingual or code-switched enterprise request.",
        "data_classification": "PUBLIC",
        "secondary_labels": ["MULTILINGUAL_INPUT"],
        "default_risk_factors": [],
        "injection_type": None,
    },
    "SENSITIVE_DATA_ACCESS": {
        "risk_tier": "MEDIUM",
        "category": "Sensitive Data",
        "description": "Interaction touches confidential or restricted internal material.",
        "data_classification": "CONFIDENTIAL",
        "secondary_labels": ["CONFIDENTIAL_CONTENT"],
        "default_risk_factors": ["restricted_content"],
        "injection_type": None,
    },
    "OUTPUT_DATA_LEAKAGE_REVIEW": {
        "risk_tier": "MEDIUM",
        "category": "Output Inspection",
        "description": "Review of model output for possible sensitive data exposure.",
        "data_classification": "CONFIDENTIAL",
        "secondary_labels": ["OUTPUT_GOVERNANCE"],
        "default_risk_factors": ["output_review"],
        "injection_type": None,
    },
    "DATA_RESIDENCY_TRANSFER": {
        "risk_tier": "MEDIUM",
        "category": "Compliance",
        "description": "Cross-border transfer or residency-sensitive data handling request.",
        "data_classification": "CONFIDENTIAL",
        "secondary_labels": ["DATA_GOVERNANCE"],
        "default_risk_factors": ["cross_border_transfer"],
        "injection_type": None,
    },
    "MODEL_INVERSION_ATTEMPT": {
        "risk_tier": "HIGH",
        "category": "Model Abuse",
        "description": "Attempt to reconstruct sensitive training data or memorized records.",
        "data_classification": "RESTRICTED",
        "secondary_labels": ["MEMORIZATION_PROBE"],
        "default_risk_factors": ["model_extraction"],
        "injection_type": None,
    },
    "INSECURE_CODE_OUTPUT": {
        "risk_tier": "HIGH",
        "category": "Code Risk",
        "description": "Generated code is insecure but not necessarily overtly malicious.",
        "data_classification": "INTERNAL",
        "secondary_labels": ["UNSAFE_IMPLEMENTATION"],
        "default_risk_factors": ["code_risk"],
        "injection_type": None,
    },
    "RAG_INJECTION_INDIRECT": {
        "risk_tier": "HIGH",
        "category": "Prompt Injection",
        "description": "Indirect prompt injection through retrieved context.",
        "data_classification": "RESTRICTED",
        "secondary_labels": ["CONTEXT_TAINTED"],
        "default_risk_factors": ["prompt_injection", "retrieval_taint"],
        "injection_type": "indirect_rag",
    },
    "PROMPT_INJECTION_DIRECT": {
        "risk_tier": "HIGH",
        "category": "Prompt Injection",
        "description": "Direct instruction override or jailbreak attempt.",
        "data_classification": "RESTRICTED",
        "secondary_labels": ["INSTRUCTION_OVERRIDE"],
        "default_risk_factors": ["prompt_injection"],
        "injection_type": "direct",
    },
    "POLICY_EVASION_ROLEPLAY": {
        "risk_tier": "HIGH",
        "category": "Prompt Injection",
        "description": "Roleplay-style attempt to bypass safety or policy boundaries.",
        "data_classification": "RESTRICTED",
        "secondary_labels": ["ROLEPLAY_EVASION"],
        "default_risk_factors": ["prompt_injection", "policy_evasion"],
        "injection_type": "roleplay",
    },
    "AGENTIC_TOOL_MISUSE": {
        "risk_tier": "CRITICAL",
        "category": "Agentic Abuse",
        "description": "Tool sequence or delegated workflow attempts risky enterprise misuse.",
        "data_classification": "RESTRICTED",
        "secondary_labels": ["TOOL_CHAIN_ABUSE"],
        "default_risk_factors": ["agentic_flow"],
        "injection_type": None,
    },
    "DATA_EXFIL_DIRECT_PII": {
        "risk_tier": "CRITICAL",
        "category": "Data Exfiltration",
        "description": "Explicit extraction of personal or regulated data.",
        "data_classification": "RESTRICTED",
        "secondary_labels": ["REGULATED_DATA_EXFIL"],
        "default_risk_factors": ["direct_exfiltration", "regulated_data"],
        "injection_type": None,
    },
    "MALICIOUS_CODE_GENERATION": {
        "risk_tier": "CRITICAL",
        "category": "Code Abuse",
        "description": "Generation of exploit, theft, or abusive code.",
        "data_classification": "RESTRICTED",
        "secondary_labels": ["OFFENSIVE_CODE"],
        "default_risk_factors": ["code_risk", "offensive_intent"],
        "injection_type": None,
    },
    "CREDENTIAL_THEFT_REQUEST": {
        "risk_tier": "CRITICAL",
        "category": "Identity Abuse",
        "description": "Attempt to phish, steal, or replay credentials and secrets.",
        "data_classification": "RESTRICTED",
        "secondary_labels": ["SECRET_EXTRACTION"],
        "default_risk_factors": ["credential_theft", "regulated_data"],
        "injection_type": None,
    },
    "PRIVILEGE_ESCALATION_REQUEST": {
        "risk_tier": "CRITICAL",
        "category": "Access Abuse",
        "description": "Request to bypass approval, elevate privileges, or abuse admin access.",
        "data_classification": "RESTRICTED",
        "secondary_labels": ["ACCESS_BYPASS"],
        "default_risk_factors": ["privilege_escalation"],
        "injection_type": None,
    },
}

RISK_BY_LABEL = {label: entry["risk_tier"] for label, entry in TAXONOMY.items()}
LABEL_DESCRIPTIONS = {label: entry["description"] for label, entry in TAXONOMY.items()}


def get_taxonomy_entry(label: str) -> TaxonomyEntry:
    return TAXONOMY[label]
