RISK_BY_LABEL = {
    "BENIGN_ENTERPRISE_QUERY": "LOW",
    "SECURITY_COMPLIANCE_ASSIST": "LOW",
    "INTERNAL_DIRECTORY_ACCESS": "LOW",
    "MULTILINGUAL_BUSINESS_QUERY": "LOW",
    "SENSITIVE_DATA_ACCESS": "MEDIUM",
    "OUTPUT_DATA_LEAKAGE_REVIEW": "MEDIUM",
    "INSECURE_CODE_OUTPUT": "HIGH",
    "RAG_INJECTION_INDIRECT": "HIGH",
    "PROMPT_INJECTION_DIRECT": "HIGH",
    "AGENTIC_TOOL_MISUSE": "CRITICAL",
    "DATA_EXFIL_DIRECT_PII": "CRITICAL",
    "MALICIOUS_CODE_GENERATION": "CRITICAL",
}

LABEL_DESCRIPTIONS = {
    "BENIGN_ENTERPRISE_QUERY": "Normal enterprise usage with low inherent risk.",
    "SECURITY_COMPLIANCE_ASSIST": "Security or compliance workflow assistance.",
    "INTERNAL_DIRECTORY_ACCESS": "Benign internal directory or team-discovery access.",
    "MULTILINGUAL_BUSINESS_QUERY": "Low-risk multilingual or code-switched enterprise request.",
    "SENSITIVE_DATA_ACCESS": "Interaction touches confidential or restricted internal material.",
    "OUTPUT_DATA_LEAKAGE_REVIEW": "Review of model output for possible sensitive data exposure.",
    "INSECURE_CODE_OUTPUT": "Generated code is insecure but not necessarily overtly malicious.",
    "RAG_INJECTION_INDIRECT": "Indirect prompt injection through retrieved context.",
    "PROMPT_INJECTION_DIRECT": "Direct instruction override or jailbreak attempt.",
    "AGENTIC_TOOL_MISUSE": "Tool sequence or delegated workflow attempts risky enterprise misuse.",
    "DATA_EXFIL_DIRECT_PII": "Explicit extraction of personal or regulated data.",
    "MALICIOUS_CODE_GENERATION": "Generation of exploit, theft, or abusive code.",
}
