# Global AI Moderation System

## Project Overview

This repository implements a **production-oriented AI moderation decision system** used within a global asynchronous search platform.

The system is designed for real-world deployment, not as a research prototype or single-model demo.

Its primary goal is to provide controlled, explainable, and conservative moderation decisions across multiple risk domains and languages.

---

## Why a System (Not a Single Model)
![AI Moderation System Architecture](moderation_system_architecture.png)

The system is built as a modular AI service:
Content moderation cannot be reliably solved by a universal model.

Different risk domains exhibit:
- different linguistic patterns,
- different tolerance for false positives,
- different operational consequences.

This project follows a strict architectural principle:

- one model = one responsibility
- models produce signals, not decisions
- final decisions are made by policy logic

This separation allows independent model evolution, safer thresholds, and predictable system behavior.

---

## Problem Domains Covered

The system is designed to handle multiple moderation risks:

- Toxicity / harassment
- Hate speech (including protected groups)
- Sexual intent / solicitation
- Threats / violent intent
- Spam / scam behavior (planned)

Each domain is treated as an independent signal source.

---

## High-Level Architecture

### Preprocessing Layer

Responsible for:
- Unicode-safe normalization,
- language-agnostic cleanup,
- preparation for model inference.

This layer is deterministic and model-independent.

---

### Model Inference Layer

Contains multiple specialized ML models.

Key properties:
- one model per risk domain,
- no shared responsibilities,
- models output scores, not decisions,
- models are independently replaceable.

---

### Policy & Orchestration Layer

The policy layer is the core decision-making component.

Responsibilities:
- aggregation of model signals,
- application of conservative thresholds,
- enforcement of decision rules,
- resolution of conflicting signals.

The system produces exactly three outcomes:
- `allow`
- `review`
- `block`

---

### API Layer

The system is exposed via a FastAPI-based HTTP service.

The API layer:
- defines stable integration contracts,
- triggers orchestration,
- does not contain business logic.

---

## Decision Policy & Explainability

Explainability is a design constraint, not an afterthought.

The architecture ensures that every decision can be traced to:
- evaluated signals,
- applied thresholds,
- policy rules.

While the API response may remain minimal, the internal system supports auditability and policy iteration.

---

## Multilingual & Cultural Strategy

Multilingual moderation is treated as a first-class concern.

Design assumptions:
- languages are grouped into clusters,
- risk thresholds may vary by locale,
- policies can be language-aware.

This avoids applying uniform global thresholds to heterogeneous linguistic contexts.

---

## Production Considerations

The system is designed with production constraints in mind:

- strict separation of concerns,
- predictable latency via parallel inference,
- conservative defaults,
- readiness for monitoring and recalibration,
- extensibility without architectural rewrites.

---

## Current Limitations

Known limitations include:
- partial language coverage,
- limited policy complexity compared to large-scale platforms,
- absence of human-in-the-loop tooling in this repository,
- constrained training data for some domains.

These limitations are explicit and guide further development.

---

## Roadmap & Future Extensions

Planned extensions:
- additional risk-domain models,
- improved multilingual calibration,
- richer policy configuration,
- integration with human review workflows,
- expansion beyond text (e.g. images, metadata).

---

## Final Note

This project is intentionally built as an **AI decision system**, not an ML demo.

Machine learning provides signals.  
Policy logic controls outcomes.  
Safety and explainability take priority.
