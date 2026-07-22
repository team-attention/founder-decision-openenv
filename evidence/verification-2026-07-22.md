# Verification evidence — 2026-07-22

Scope: reproducibility gates plus authorized publication. The GitHub code repository was
published after all gates passed. Hugging Face publication remains pending because the local
CLI has no authenticated account; a browser device-code attempt expired without authorization.

## G0 — Upstream and rights: PASS

Direct downloads and decoded record snapshots were produced with:

```bash
uv run --with pyarrow python scripts/fetch_upstream_evidence.py
```

Result: four candidates downloaded from pinned raw URLs; `candidate-manifest.json` contains
official URL, immutable revision, license locator, raw SHA-256, decoded record SHA-256, shape,
and top-level null counts.

| Candidate | Revision | License | Raw SHA-256 | Record SHA-256 | Result |
|---|---|---|---|---|---|
| allenai/RLVR-GSM | `b14884519de816306cf8ce7fc74547284b9ac548` | MIT | `a7e682...f2e1` | `ae7833...efa3` | selected |
| allenai/RLVR-MATH | `bd2a93551b503a395fadd1a740d957559cfe6f3c` | MIT | `2fb848...9422` | `53d6e8...f550` | nullable unrelated fields |
| tau2-bench retail | `f0927a4b9cdfa7b374269efaa29ff7fdac90d8fc` | MIT | `8e03eb...2e8` | `b346af...d40d` | domain-specific nested gold actions |
| WebArena | `dce04686a56253aefba7b18a4fa0937cf1dc987b` | Apache-2.0 | `7b5038...652e` | `fe3fda...52e0` | browser snapshot/auth coupling |

Official locator availability check:

```text
200 https://www.ycombinator.com/legal/#tou
200 https://www.ycombinator.com/blog/startup-school-videos
200 https://www.ycombinator.com/blog/startup-school-week-1-recap-kevin-hale-and-eric-migicovsky
200 https://www.ycombinator.com/blog/startup-school-2018-curriculum
200 https://huggingface.co/datasets/allenai/RLVR-GSM
200 https://github.com/huggingface/OpenEnv/tree/65c506ef94bb1f7279cb4359673b3ef81031d01f
```

`RIGHTS_MANIFEST.json` records YC Terms and a conservative link-only/no-scrape boundary.
The synthetic split contains no YC article text, transcript, audio, video, or image. License
compatibility result: PASS with `NOTICE`; final legal/publication review remains an approval item.

## G1 — Exact schema: PASS

Chosen schema at `allenai/RLVR-GSM@b148845...`:

```text
messages: nullable list<nullable struct<content: nullable string, role: nullable string>>
ground_truth: nullable string
dataset: nullable string
```

The pinned Arrow schema marks all top-level fields, list elements, and nested message members
nullable; the 7,473 decoded train rows happened to contain zero top-level nulls. All physical
nullable semantics are preserved. `tests/test_schema_gate.py` validates the actual record,
semantic decode/encode equality, field/type/nesting/nullability snapshot, nullable round-trip,
duplicate/malformed/extra field rejection,
and sidecar-only additions. Dataset hashes:

```text
fc5027f316131eb7a64159038bdb055c6e17acafa9a61e55d76a9ef46d893754 dataset-v0.1.0.jsonl
b2f4195b9268a1f20f2b9d0d0e116e9a798b01f85c048997f1c8618b26a6e996 environment-metadata-v0.1.0.json
e25f2b1f11cfe1b42a6cb880fd36c055ff15e0033e40a913e9eccdf56b258dd0 schema snapshot
```

## G2 — OpenEnv execution: PASS

OpenEnv is pinned to `0.4.1` (stable commit `65c506ef...`). Typed Action, Observation, and
State extend its Pydantic contract. `done = terminated or truncated` bridges OpenEnv's scalar
`done` interface; reward components remain typed house extensions.

```bash
uv run openenv validate . --json --verbose
# passed=true; 1/1 required local criterion, all reported modes true

uv run server
uv run openenv validate --url http://127.0.0.1:8000
# passed=true; runtime criteria 6/6
```

Persistent WebSocket client result:

```text
reset ycfd-008 False
step 1 1.0 False False
step 2 1.0 False False
step 3 1.0 False False
step 4 1.0 False True
{"step_count": 4, "terminated": false, "truncated": true}
```

`uv run ycfd-demo` reported `replay_equal=true`. OpenEnv's own validator does not test reset,
step, replay, reward vector, or terminal semantics, so project tests are the contract-level
supplement.

## G3 — Verifier: PASS

`uv run pytest -q` result: `19 passed`. Covered invalid action, disallowed action path, exact
cost, budget/time overspend, forged locator, state arithmetic, future leakage, malformed JSON,
duplicate/extra fields, side-effect-free hard failure, action-dependent transition, and LLM/audit
separation. Failures return stable codes including `ACTION_NOT_ALLOWED`, `COST_TAMPERING`,
`BUDGET_OR_TIME_OVERSPEND`, `FORGED_SOURCE_LOCATOR`, `STATE_ARITHMETIC_MISMATCH`, and
`FUTURE_LEAKAGE`.

## G4 — Baselines: PASS

```bash
uv run ycfd-baselines --output artifacts/baselines-v0.1.0.json
```

Both policies used the same eight-case held-out split and seed `20260722`; split leakage check
reported an empty shared-case list. Random: hard reward `1.0`, frozen preferred-action match
`0.375`, mean final synthetic MRR `46875`. State-only rule: hard reward `1.0`, match `0.125`,
mean final synthetic MRR `80000`. Equal hard reward is expected: it measures verifier contract
compliance, not strategy. Trajectories preserve every component and final state.

## G5 — Reproducibility: PASS

Clean environment commands:

```bash
clean_root=$(mktemp -d /tmp/ycfd-clean.XXXXXX)
UV_PROJECT_ENVIRONMENT="$clean_root/venv" uv sync --all-extras --frozen
UV_PROJECT_ENVIRONMENT="$clean_root/venv" uv run pytest -q
UV_PROJECT_ENVIRONMENT="$clean_root/venv" uv run ruff check .
UV_PROJECT_ENVIRONMENT="$clean_root/venv" uv run mypy
UV_PROJECT_ENVIRONMENT="$clean_root/venv" uv run python scripts/publication_dry_run.py
UV_PROJECT_ENVIRONMENT="$clean_root/venv" uv run ycfd-demo
```

Results: clean install PASS; 19 tests PASS; ruff PASS; mypy `Success: no issues found in 12
source files`; demo replay PASS; publication dry run PASS. `uv.lock` SHA-256 is
`c9efb772f73a5e96c6a88fea3d1fde741462c0309bf254256fef0082a71b6471`.

Container commands/results:

```bash
docker build -t yc-founder-decision-openenv:0.1.0 .
# PASS, final local image ID sha256:a39d353e796dae07bce8d1ff6ba4a2d66930b488011effa6ba8405b54dfd0a2b
docker run -d --rm --name ycfd-validation -p 8001:8000 yc-founder-decision-openenv:0.1.0
uv run openenv validate --url http://127.0.0.1:8001
# PASS, runtime criteria 6/6
docker stop ycfd-validation
```

Secret scan found no token/private-key patterns. `git diff --check` is part of the final gate.

## G6 — Publication dry run: PASS

```bash
uv run python scripts/publication_dry_run.py
```

Result:

```json
{"passed":true,"remote_mutation":false,"github_layout":true,"hf_dataset_layout":true,"hf_docker_space_layout":true,"records":24}
```

GitHub, HF Dataset, and Docker Space cards/layouts plus exact commands are in
`publication/APPROVAL_REQUIRED.md`. The public GitHub repository is
`https://github.com/team-attention/founder-decision-openenv` at initial publication commit
`b3b72317afa1563b844f7ebe22ace1f6199271a3`. Hugging Face Dataset and Space publication are
blocked only on local Hugging Face authentication, not on a failed technical gate.
