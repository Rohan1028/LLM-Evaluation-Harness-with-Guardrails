# Product Overview

SentinelIQ ships with:
- Hybrid retrieval (sparse + dense) backed by a Chroma collection.
- Policy-aware LLM orchestrator with citation and toxicity guardrails.
- Configurable pipelines for summarization, Q&A, and blueprint generation.

The orchestrator requires bracketed citations like `[product:2]` or `[security:1]` tied to document chunks. Clients can extend guardrails via YAML or Guardrails.ai specs.

The product roadmap includes integrations with open source models via Ollama and enterprise deployments through Azure OpenAI.
