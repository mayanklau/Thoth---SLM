RISK_BY_LABEL = {
    "BENIGN_ENTERPRISE_QUERY": "LOW",
    "SECURITY_COMPLIANCE_ASSIST": "LOW",
    "SENSITIVE_DATA_ACCESS": "MEDIUM",
    "RAG_INJECTION_INDIRECT": "HIGH",
    "PROMPT_INJECTION_DIRECT": "HIGH",
    "DATA_EXFIL_DIRECT_PII": "CRITICAL",
    "MALICIOUS_CODE_GENERATION": "CRITICAL",
}

LABEL_DESCRIPTIONS = {
    "BENIGN_ENTERPRISE_QUERY": "Normal enterprise usage with low inherent risk.",
    "SECURITY_COMPLIANCE_ASSIST": "Security or compliance workflow assistance.",
    "SENSITIVE_DATA_ACCESS": "Interaction touches confidential or restricted internal material.",
    "RAG_INJECTION_INDIRECT": "Indirect prompt injection through retrieved context.",
    "PROMPT_INJECTION_DIRECT": "Direct instruction override or jailbreak attempt.",
    "DATA_EXFIL_DIRECT_PII": "Explicit extraction of personal or regulated data.",
    "MALICIOUS_CODE_GENERATION": "Generation of exploit, theft, or abusive code.",
}
