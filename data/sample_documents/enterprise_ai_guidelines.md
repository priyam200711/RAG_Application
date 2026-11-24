# Enterprise AI Architecture and Governance Guidelines

## 1. Overview and Core Principles
This policy sets the engineering and compliance requirements for developing, deploying, and operating Generative AI systems across the organization. All Large Language Model (LLM) applications must adhere to the following principles:
- **Traceability and Grounding**: Every generative answer served to customers or employees must be grounded in verified documentation via Retrieval-Augmented Generation (RAG).
- **Zero Data Retention with Third Parties**: Direct API calls to commercial LLM providers must enforce zero data retention flags (`data_retention=False`).
- **Least Privilege Access**: Vector indexes and retrieval systems must respect underlying database Role-Based Access Controls (RBAC).

## 2. Model Selection and Routing Criteria
The platform employs a tiered model routing strategy:
- **Tier 1 (Fast / Edge / LPU)**: Ultra-fast models running on Groq (such as `openai/gpt-oss-120`, `llama-3.3-70b-versatile`) delivering sub-200ms first-token latency for real-time RAG streaming.
- **Tier 2 (Primary Reasoning)**: Models such as `gpt-4o` or `gemini-1.5-pro` for deep synthesis, complex domain analysis, and multi-document consolidation.
- **Latency SLO**: 95th percentile latency (p95) for generative endpoints must remain under 1,500 milliseconds for streaming first-token and under 3,000 milliseconds for non-streamed responses.

## 3. Data Privacy and PII Masking
Before any text is passed to an embedding generator or LLM:
- All Personally Identifiable Information (PII) including Social Security Numbers, corporate credit card numbers, and patient medical identifiers must be stripped or masked using cryptographic tokens.
- Vector database partitions must be encrypted at rest with AES-256 and in transit via TLS 1.3.

## 4. API Key Management and Security
- API keys must never be hardcoded or checked into Git repositories.
- All secrets must be dynamically injected via environment variables or secret vaults (e.g., HashiCorp Vault, AWS Secrets Manager).
- Key rotation is mandatory every 90 days.
