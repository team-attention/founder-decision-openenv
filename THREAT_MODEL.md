# Threat model

Protected properties are schema integrity, deterministic replay, bounded resource arithmetic,
source provenance, split isolation, and the exclusion of YC source content.

| Threat | Control | Residual limitation |
|---|---|---|
| Extra/duplicate/malformed fields | Pydantic `extra=forbid`, strict duplicate-key decoder | Upstream clients may pre-normalize JSON before validation |
| Forged locator | Exact allowlist from versioned sidecar | URL ownership/content can change after review |
| Budget/time tampering | Submitted costs and post-state claims checked against frozen costs | Frozen costs are synthetic, not real estimates |
| Future leakage | Step equality, forbidden-token checks, hash-keyed hidden sidecar, split test | Semantic leakage beyond explicit tokens needs audit |
| Reward hacking | Five independent hard components and failure codes; LLM audit excluded | Valid but strategically poor actions can score 1.0 |
| Replay drift | Locked dependencies, frozen versions, canonical transcript equality test | Platform/container differences still require release CI |
| Source redistribution | Link-only manifest and absence tests/manual review | A contributor could later add prohibited content |
| OpenEnv REST state loss | Persistent WebSocket client documented and integration-tested | OpenEnv v0.4.1 REST endpoints remain stateless per call |
