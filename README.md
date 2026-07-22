---
title: Founder Decision OpenEnv
emoji: "🧭"
colorFrom: orange
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Founder Decision OpenEnv

A reproducible package for **constrained sequential decisions under a frozen synthetic
transition model and verifier `v0.1.0`**. It does not predict startup success, reproduce YC
content, or claim that a static dataset is an environment.

The package contains 24 original synthetic cases, a four-step interactive OpenEnv server,
strict deterministic reward components, deterministic replay, and seeded random plus
state-only rule baselines. YC material appears only as official URL/heading/timestamp locators.

## Frozen data contract

The dataset record uses the exact decoded `allenai/RLVR-GSM` schema at revision
`b14884519de816306cf8ce7fc74547284b9ac548`:

```text
messages: nullable list[nullable {content: nullable string, role: nullable string}]
ground_truth: nullable string
dataset: nullable string
```

No Team Attention field is added to those records. Case state, splits, locators, costs,
transitions, and rights flags live in `environment-metadata-v0.1.0.json`, keyed by the
canonical record SHA-256. The pinned Parquet marks every top-level field, list element, and
nested message member nullable; V0 preserves that even though the decoded train rows had no
top-level null values. See `schemas/rlvr-gsm-record.schema-snapshot.json`.

## Run

```bash
uv sync --all-extras --frozen
uv run pytest -q
uv run ycfd-demo
uv run ycfd-baselines --output artifacts/baselines-v0.1.0.json
uv run openenv validate . --json --verbose
uv run server
```

In another shell, validate the running adapter:

```bash
uv run openenv validate --url http://127.0.0.1:8000
```

OpenEnv v0.4.1 exposes `done` and scalar reward only. This package adds typed
`terminated`, `truncated`, and `reward_components`, while mapping
`done = terminated or truncated`. Use the persistent WebSocket client for an episode;
OpenEnv's stateless REST probes are not a trajectory client.

## Hard reward

Each component contributes `0.2`: action validity, exact/affordable budget-time use,
official source locator validity, state arithmetic, and future-leakage rejection.
Machine-readable failure codes accompany failures. The optional strategic-quality audit is
explicitly excluded from hard reward and is not calibrated to real outcomes.

## Boundaries

- Synthetic/derived status: all 24 founder cases are original synthetic fixtures.
- YC boundary: no scraping or redistribution of original text, transcript, audio, or video.
- License: Team Attention code/data is MIT. Third-party evidence retains upstream terms.
- Limitation: passing the verifier means contract compliance under a frozen simulator, not
  good business judgment or likely startup success.
- Non-endorsement: this independent synthetic benchmark is not affiliated with or endorsed by YC.
- Public surface: the code is published on GitHub; Hugging Face Dataset and Docker Space
  publication status and identifiers are recorded in `publication/APPROVAL_REQUIRED.md`.
