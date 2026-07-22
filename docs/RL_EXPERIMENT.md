# Observation-only agent and tabular RL experiment

This is a reproducible benchmark experiment over the package's frozen synthetic transition
model. Its numeric source of truth is the sealed JSON in `artifacts/rl/`; values below are
copied from those artifacts, not recomputed for this report.

## What can and cannot be trained

Trial 2 updates only a local tabular Q table. It does not fine-tune, gradient-update, or
otherwise alter Codex, Claude, Terra, or any other foundation-model weights. Codex Terra is
an inference policy in Trial 1. Claude Code could implement the same `ObservationPolicy`
adapter or produce decisions, but this recorded run uses Codex subagents with
`gpt-5.6-terra`.

Hard reward (`verifier-v0.1.0`) is a separate action-contract compliance measure. The Q
learner optimizes the deterministic `synthetic-utility-v0.1.0`; the two are deliberately
never scalarized into a combined reward.

## Information boundary

Every decision policy receives only serialized `FounderObservation` data:

- `done`, `reward`, `metadata`, `case_id`, `week`, `budget_cents`, `founder_hours`
- `active_users`, `interviews_completed`, `qualified_pipeline`, `mrr_cents`, `price_cents`
- `allowed_actions`, all six public `action_specs` (allowed flag and exact cost), and
  `source_locators`
- `reward_components`, `failure_codes`, `terminated`, and `truncated`

`build_trusted_action` derives spend, founder hours, and the state claim from that public
observation. Policies cannot inject them. Selection never receives a case sidecar, canonical
record labels, preference labels, future transition outcomes, Q tables (except frozen Q
evaluation), or oracle results. The Terra prompt is frozen as
`observation-only-v0.1.0`; it instructs the model to use only the supplied observation and
`AgentDecision` JSON schema, return JSON only, copy a visible locator exactly, and make no
model-weight update.

## Trial 1: Terra inference

The sealed ledger records eight held-out seeds (`16..23`) and 32 sequential four-step
decisions. Its policy provenance is exactly `codex-subagent` / `gpt-5.6-terra` / `inference`,
with `model_weight_updates: false`. It freezes the exact neutral instruction and exact
`AgentDecision` JSON schema in `request_contract`, including a recomputed
`decision_schema_sha256`. Each turn retains the public-observation SHA-256, the SHA-256 of
the stable request envelope `{instruction, observation, decision_schema}`, the raw response,
and parse-repair record. Each seed is mapped to one unique session task ID
`/root/terra_policy_seed16` through `/root/terra_policy_seed23`.

Those task IDs and the claimed model selection are **session-attested orchestration metadata**:
they are integrity-sealed in this repository but are not cryptographically provider-signed by
Codex. The validator rejects resealed ledgers whose model/prompt contract, schema hash,
request-envelope hash, seed mapping, or unique task IDs do not match. There were zero parse
repairs. The separately sealed episode artifact replays all eight episodes from fresh
environments.

Terra's mean total hard reward was `4.0` (all five per-step contract components averaged
`0.2`), and its mean total synthetic utility was `1.837813`. These figures are metrics of
the frozen benchmark, not startup-success estimates.

## Trial 2: Q-learning

The public state is `(4 - week, budget_cents // 1000, founder_hours, allowed_action_mask)`.
Actions flow through the same trusted factory. For each transition, the local table applies:

```text
Q <- Q + N(s,a)^-0.6 * (r + 0.95 max_a' Q(s',a') - Q)
```

where `r` is only `synthetic-utility-v0.1.0`. The five runs use RNG seeds `20260723` through
`20260727`, exactly 1,000 training episodes each, and training cases `0..15` only. Epsilon
at zero-based index `i` is `max(0.05, 0.30 * (1 - i / 999))`.

The first and final 50-episode mean returns, respectively, were:

| RNG seed | First 50 | Final 50 |
| --- | ---: | ---: |
| 20260723 | 0.645868 | 1.619650 |
| 20260724 | 0.637393 | 1.798234 |
| 20260725 | 0.720734 | 1.468725 |
| 20260726 | 0.815434 | 1.742675 |
| 20260727 | 0.664418 | 1.632900 |

## Trial 3: frozen held-out comparison

The comparison uses held-out seeds `16..23`. “Utility dispersion” is population standard
deviation of total synthetic utility across episodes for a deterministic policy and across
the five per-run means for random and learned Q. Oracle gap is the exhaustive oracle mean
utility minus the policy mean; it is not a deployable-policy score.

| Policy | Episodes | Mean hard reward | Mean synthetic utility | Utility dispersion | Oracle gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 40 | 4.0 | 0.795344 | 0.084439 | 1.382156 |
| Black-box rule | 8 | 4.0 | 0.937188 | 0.359730 | 1.240312 |
| Terra | 8 | 4.0 | 1.837813 | 0.850398 | 0.339687 |
| Learned Q | 40 | 4.0 | 1.462980 | 0.088292 | 0.714520 |
| Exhaustive oracle | 8 | 4.0 | 2.177500 | 0.818394 | 0.000000 |

On this held-out synthetic utility, learned Q beat random and the frozen black-box rule, but
did not beat Terra or the exhaustive oracle. All policies reached the same hard contract
score, so that measure does not distinguish their strategic behavior here.

## Reproduce

```bash
uv run ycfd-agent-trial \
  --decisions artifacts/rl/terra-held-out-decisions-v0.1.0.json \
  --output artifacts/rl/terra-agent-episodes-v0.1.0.json
uv run ycfd-q-learning --episodes 1000 --output artifacts/rl/q-learning-v0.1.0.json
uv run ycfd-rl-evaluate \
  --terra-decisions artifacts/rl/terra-held-out-decisions-v0.1.0.json \
  --q-artifact artifacts/rl/q-learning-v0.1.0.json \
  --output artifacts/rl/held-out-comparison-v0.1.0.json
uv run pytest -q
uv run ruff check .
uv run mypy src/yc_founder_decision_env
uv run python scripts/publication_dry_run.py
```

The decision ledger must be supplied as a sealed structured ledger; legacy unsigned or flat
decision shorthand is rejected. Re-running the Q and held-out commands byte-compares with
the committed artifacts.

## Artifact hashes

Embedded integrity hashes cover stable unsigned JSON payloads. Full-file SHA-256 values
below cover the complete pretty-printed files.

| Artifact | Embedded integrity SHA-256 | Full-file SHA-256 |
| --- | --- | --- |
| `terra-held-out-decisions-v0.1.0.json` | `fb4453d4058cbf714550b07b301f0275b3aa34084b9047ffe73d3abb7a056894` | `f3e7e8babe7dc7996a748f5f67f29a7b0119b5fd77532b61646f98e9aa6e96f6` |
| `terra-agent-episodes-v0.1.0.json` | `de40fc5cf0754deebad1ee80716de6e6bad8efa12abcc7b6353b6679183cbc3f` | `0cb88948e417ae16d5aefe9b3ba387c32c89b033034d81a44fb55edeb155a854` |
| `q-learning-v0.1.0.json` | `b995c94635e55b11cf7fcaa141531f53d1aa967912c0485c00e586cc1805ab33` | `3c9aa5c451afa5935beed1a031350cd84069edaae0565862b2043599e3f93cad` |
| `held-out-comparison-v0.1.0.json` | `1fdcadc4b5f7a5846567bc427d97d04d7ba9b3b0865883514befd187ca71a0ea` | `4e3241bf9beb5fe1e6d1f85bead27e9cca4c22ed218881638a070c4a1a091293` |

## Interpretation and limitations

This is a 24-case synthetic benchmark with a frozen transition model, shared action dynamics,
and a small tabular policy. The oracle gets a fresh-environment exhaustive-query advantage and
is explicitly non-deployable. Results do not demonstrate real-world startup success, general
business judgment, or foundation-model training. They only show these fixed policies under the
documented public-observation contract and synthetic utility.
