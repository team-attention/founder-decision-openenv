# Agent and Q-Learning Trials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three reproducible experiments: observation-only Codex Terra agent episodes, real tabular Q-learning against a separate synthetic utility, and a leakage-safe held-out comparison against random, a black-box rule, Terra, learned Q, and an exhaustive oracle.

**Architecture:** Keep the existing verifier reward unchanged as the hard contract-compliance signal. Extend `FounderObservation` with every public action's allowed flag and exact costs, then make all policies consume only serialized observations through one trusted action factory; environment sidecars, `preferred_action`, canonical `ground_truth`, and transition tables remain unavailable to action selection. Train tabular Q policies on seeds `0..15` using a versioned synthetic utility, freeze them, and evaluate on seeds `16..23`; stable JSON artifacts carry their own SHA-256 and are replayed through fresh environment instances.

**Tech Stack:** Python 3.11+, `uv`, Pydantic 2, OpenEnv 0.4.1, standard-library `random`/`hashlib`/`itertools`, pytest, Ruff, mypy, Codex subagents using `gpt-5.6-terra` for model execution.

## Global Constraints

- Planning/review may use the current high-reasoning planning agent; every actual language-model decision in Trial 1 and the Terra column of Trial 3 must be produced by a Codex subagent explicitly launched with model `gpt-5.6-terra`.
- A Terra subagent receives only `FounderObservation.model_dump(mode="json")`, the `AgentDecision` JSON schema, and neutral instructions. Never include the sidecar, `preferred_action`, canonical record `ground_truth`, transition maps, Q tables, oracle results, or later observations.
- Codex, Claude Code, and Terra are inference agents in this runtime. Their model weights cannot be gradient-updated here. “Training” in Trial 2 means updating a local tabular Q table only; do not claim Codex/Claude/Terra fine-tuning or weight updates.
- Preserve hard reward `verifier-v0.1.0` exactly and report it separately from `synthetic-utility-v0.1.0`. Never scalarize the two into one score.
- Synthetic utility is a deterministic benchmark objective under `frozen-v0.1.0`, not startup-success evidence.
- Training seeds are exactly `0..15`; held-out seeds are exactly `16..23`. No training, tuning, tie-break changes, or prompt changes may use held-out outcomes.
- Q-learning runs use RNG seeds `20260723`, `20260724`, `20260725`, `20260726`, and `20260727`; each run uses exactly `1000` training episodes, `gamma=0.95`, visit-dependent `alpha=1/N(s,a)^0.6`, and epsilon-greedy exploration.
- An epsilon schedule is frozen as `max(0.05, 0.30 * (1 - episode_index / 999))` for zero-based episode indices `0..999`.
- The canonical upstream record schema and sidecar assets are unchanged. New experiment metadata lives in code, documentation, or `artifacts/rl/`.
- Generated artifacts must be deterministic for fixed decisions/configuration, contain an `integrity_sha256`, and pass replay from a fresh environment.
- Do not push or publish during implementation unless the user separately authorizes that external mutation.

---

## File Structure

- Modify `src/yc_founder_decision_env/models.py`: public action spec and observation-only agent-decision types.
- Modify `src/yc_founder_decision_env/server/environment.py`: expose all six action specs, exact costs, and allowed flags in every observation.
- Create `src/yc_founder_decision_env/agent_trial.py`: observation-only policy protocol, trusted action factory, agent episode runner, stable artifact hashing, replay, and CLI ingestion of Terra decisions.
- Create `src/yc_founder_decision_env/rewards.py`: the isolated `synthetic-utility-v0.1.0` definition.
- Create `src/yc_founder_decision_env/q_learning.py`: public state encoder, epsilon-greedy training, visit-dependent Q updates, frozen policy serialization, and CLI.
- Create `src/yc_founder_decision_env/rl_evaluation.py`: random, black-box rule, frozen Q, Terra replay, exhaustive black-box oracle, held-out comparison, and CLI.
- Create `tests/test_agent_trial.py`: observation boundary, trusted factory, hashing, and replay tests.
- Create `tests/test_q_learning.py`: utility, state encoding, Q update, seed/config, determinism, and train-only tests.
- Create `tests/test_rl_evaluation.py`: policy boundary, oracle, held-out split, separate metrics, artifact integrity, and replay tests.
- Modify `pyproject.toml`: three command entry points.
- Modify `README.md`: concise trial entry points and model-weight boundary.
- Create `docs/RL_EXPERIMENT.md`: full protocol, formulas, execution commands, interpretation, and actual result table.
- Generate `artifacts/rl/terra-held-out-decisions-v0.1.0.json`: raw Terra decisions for held-out seeds `16..23`, with per-turn observation hashes.
- Generate `artifacts/rl/terra-agent-episodes-v0.1.0.json`: eight four-step replayable Terra episodes; seed 16 is the required concrete Trial 1 episode.
- Generate `artifacts/rl/q-learning-v0.1.0.json`: all five training runs, learning curves, visit counts, frozen Q tables, and representative update traces.
- Generate `artifacts/rl/held-out-comparison-v0.1.0.json`: frozen held-out trajectories and separate hard/utility results for all five policy families.

---

### Task 1: Make the environment observation self-sufficient

**Files:**
- Modify: `src/yc_founder_decision_env/models.py`
- Modify: `src/yc_founder_decision_env/server/environment.py`
- Modify: `tests/test_environment.py`

**Interfaces:**
- Produces: `PublicActionSpec(action_type, allowed, spend_cents, founder_hours)` and `FounderObservation.action_specs: list[PublicActionSpec]`.
- Invariant: `action_specs` contains all six `ActionName` values in the literal declaration order; `allowed_actions` remains for backward compatibility and equals the subset whose `allowed` flag is true.

- [ ] **Step 1: Write the failing public-contract tests**

Append these tests to `tests/test_environment.py`:

```python
def test_observation_exposes_every_action_cost_without_private_sidecar() -> None:
    env = FounderDecisionEnvironment()
    observation = env.reset(seed=0)
    specs = {spec.action_type: spec for spec in observation.action_specs}
    assert list(specs) == [
        "interview_users",
        "build_feature",
        "sell_pilot",
        "fundraise",
        "change_price",
        "abstain",
    ]
    assert specs["build_feature"].spend_cents == 60_000
    assert specs["build_feature"].founder_hours == 24
    assert specs["build_feature"].allowed == (
        "build_feature" in observation.allowed_actions
    )
    dumped = observation.model_dump(mode="json")
    serialized = json.dumps(dumped, sort_keys=True)
    assert "preferred_action" not in serialized
    assert "ground_truth" not in serialized
    assert "transitions" not in serialized


def test_public_action_specs_remain_available_after_step() -> None:
    env = FounderDecisionEnvironment()
    reset_observation = env.reset(seed=0)
    step_observation = env.step(valid_action(env, "abstain"))
    assert step_observation.action_specs == reset_observation.action_specs
```

Also add `import json` at the top of the test file.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
uv run pytest tests/test_environment.py::test_observation_exposes_every_action_cost_without_private_sidecar tests/test_environment.py::test_public_action_specs_remain_available_after_step -v
```

Expected: FAIL because `FounderObservation` has no `action_specs` attribute.

- [ ] **Step 3: Add the typed public action spec**

In `models.py`, define the type immediately after `ActionName` and add the field to `FounderObservation`:

```python
class PublicActionSpec(BaseModel):
    """Agent-visible action contract; contains no outcome or preference data."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action_type: ActionName
    allowed: bool
    spend_cents: int = Field(ge=0)
    founder_hours: int = Field(ge=0)


class FounderObservation(Observation):
    # Keep all existing fields.
    action_specs: list[PublicActionSpec]
```

The implementation must retain every existing `FounderObservation` field; the snippet shows only the new field insertion.

- [ ] **Step 4: Populate the specs from the sidecar only at the environment boundary**

Import `ActionName` and `PublicActionSpec` in `server/environment.py`, define the frozen order, and pass specs in `_observation`:

```python
ACTION_ORDER: tuple[ActionName, ...] = (
    "interview_users",
    "build_feature",
    "sell_pilot",
    "fundraise",
    "change_price",
    "abstain",
)


def _public_action_specs(self) -> list[PublicActionSpec]:
    assert self._sidecar is not None
    allowed = set(self._sidecar["allowed_actions"])
    return [
        PublicActionSpec(
            action_type=name,
            allowed=name in allowed,
            spend_cents=self._sidecar["action_costs"][name]["budget_cents"],
            founder_hours=self._sidecar["action_costs"][name]["hours"],
        )
        for name in ACTION_ORDER
    ]
```

Add `action_specs=self._public_action_specs()` to the `FounderObservation(...)` constructor. No transition values or `preferred_action` may be added to the observation or its metadata.

- [ ] **Step 5: Run focused and regression tests**

Run:

```bash
uv run pytest tests/test_environment.py -q
uv run pytest tests/test_schema_gate.py -q
```

Expected: all tests PASS; exact upstream schema tests remain unchanged.

- [ ] **Step 6: Commit the independently reviewable contract change**

```bash
git add src/yc_founder_decision_env/models.py src/yc_founder_decision_env/server/environment.py tests/test_environment.py
git commit -m "feat: expose public action contracts in observations"
```

---

### Task 2: Add an observation-only agent adapter and replayable artifact contract

**Files:**
- Modify: `src/yc_founder_decision_env/models.py`
- Create: `src/yc_founder_decision_env/agent_trial.py`
- Create: `tests/test_agent_trial.py`

**Interfaces:**
- Produces: `AgentDecision`, `ObservationPolicy`, `build_trusted_action(observation, decision)`, `run_decision_episode(seed, decisions, policy_metadata)`, `seal_artifact(payload)`, `verify_artifact(payload)`, and `replay_agent_artifact(payload)`.
- Consumes: `FounderObservation.action_specs` from Task 1.
- Security boundary: the runner accepts `FounderObservation` and agent decisions only; it never accepts a sidecar or canonical record.

- [ ] **Step 1: Write failing tests for the adapter boundary and trusted fields**

Create `tests/test_agent_trial.py`:

```python
import copy

import pytest

from yc_founder_decision_env.agent_trial import (
    build_trusted_action,
    replay_agent_artifact,
    run_decision_episode,
    seal_artifact,
    verify_artifact,
)
from yc_founder_decision_env.models import AgentDecision
from yc_founder_decision_env.server.environment import FounderDecisionEnvironment


def decision(action_type: str = "sell_pilot") -> AgentDecision:
    return AgentDecision.model_validate(
        {
            "action_type": action_type,
            "rationale": "Use the visible pipeline and resource constraints.",
            "source_locator": "https://www.ycombinator.com/blog/startup-school-videos",
        }
    )


def test_trusted_factory_derives_cost_and_claim_from_observation() -> None:
    observation = FounderDecisionEnvironment().reset(seed=0)
    action = build_trusted_action(observation, decision())
    spec = next(s for s in observation.action_specs if s.action_type == "sell_pilot")
    assert action.spend_cents == spec.spend_cents
    assert action.founder_hours == spec.founder_hours
    assert action.claim.budget_after_cents == observation.budget_cents - spec.spend_cents
    assert action.claim.founder_hours_after == observation.founder_hours - spec.founder_hours
    assert action.claim.observed_step == observation.week


def test_episode_artifact_contains_no_private_selection_fields() -> None:
    decisions = [decision("abstain") for _ in range(4)]
    artifact = run_decision_episode(
        16,
        decisions,
        {"provider": "codex", "model": "gpt-5.6-terra", "mode": "inference"},
    )
    rendered = str(artifact)
    assert "preferred_action" not in rendered
    assert "ground_truth" not in rendered
    assert "transitions" not in rendered
    assert artifact["episode"]["steps"][-1]["observation"]["done"] is True
    assert verify_artifact(artifact)
    assert replay_agent_artifact(artifact) == artifact["episode"]["trajectory_sha256"]


def test_artifact_hash_rejects_tampering() -> None:
    artifact = seal_artifact({"schema_version": "test", "value": 1})
    tampered = copy.deepcopy(artifact)
    tampered["value"] = 2
    assert verify_artifact(artifact)
    assert not verify_artifact(tampered)


def test_agent_decision_forbids_cost_and_claim_injection() -> None:
    with pytest.raises(Exception):
        AgentDecision.model_validate(
            {
                **decision().model_dump(),
                "spend_cents": 0,
                "claim": {"observed_step": 99},
            }
        )
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/test_agent_trial.py -v`

Expected: collection FAIL because `AgentDecision` and `agent_trial` do not exist.

- [ ] **Step 3: Define the minimal model-output type**

Add to `models.py`:

```python
class AgentDecision(BaseModel):
    """Only fields an inference agent may choose."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action_type: ActionName
    rationale: str = Field(min_length=1, max_length=1200)
    source_locator: str = Field(min_length=8, max_length=500)
```

- [ ] **Step 4: Implement stable hashing and the trusted action factory**

Create `agent_trial.py` with these exact core functions:

```python
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from .data import stable_json
from .models import AgentDecision, FounderAction, FounderObservation, StateClaim
from .server.environment import FounderDecisionEnvironment

ARTIFACT_SCHEMA_VERSION = "agent-episode-v0.1.0"


class ObservationPolicy(Protocol):
    def decide(self, observation: FounderObservation) -> AgentDecision: ...


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def seal_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(payload)
    sealed.pop("integrity_sha256", None)
    sealed["integrity_sha256"] = payload_sha256(sealed)
    return sealed


def verify_artifact(payload: dict[str, Any]) -> bool:
    expected = payload.get("integrity_sha256")
    unsigned = copy.deepcopy(payload)
    unsigned.pop("integrity_sha256", None)
    return isinstance(expected, str) and expected == payload_sha256(unsigned)


def build_trusted_action(
    observation: FounderObservation, decision: AgentDecision
) -> FounderAction:
    spec = next(s for s in observation.action_specs if s.action_type == decision.action_type)
    return FounderAction(
        action_type=decision.action_type,
        rationale=decision.rationale,
        source_locator=decision.source_locator,
        spend_cents=spec.spend_cents,
        founder_hours=spec.founder_hours,
        claim=StateClaim(
            budget_after_cents=max(0, observation.budget_cents - spec.spend_cents),
            founder_hours_after=max(0, observation.founder_hours - spec.founder_hours),
            observed_step=observation.week,
        ),
    )
```

The `max(0, ...)` is deliberate: it keeps a model's unaffordable choice representable so the environment can emit `BUDGET_OR_TIME_OVERSPEND`; it must not silently replace the choice with another action.

- [ ] **Step 5: Implement episode recording and replay**

Continue `agent_trial.py`:

```python
def _public_observation(observation: FounderObservation) -> dict[str, Any]:
    return observation.model_dump(mode="json")


def run_decision_episode(
    seed: int,
    decisions: list[AgentDecision],
    policy_metadata: dict[str, str],
) -> dict[str, Any]:
    if len(decisions) != 4:
        raise ValueError("EXACTLY_FOUR_DECISIONS_REQUIRED")
    env = FounderDecisionEnvironment()
    observation = env.reset(seed=seed)
    initial = _public_observation(observation)
    steps: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions):
        if observation.done:
            break
        action = build_trusted_action(observation, decision)
        before_hash = payload_sha256(_public_observation(observation))
        observation = env.step(action)
        steps.append(
            {
                "step": index,
                "input_observation_sha256": before_hash,
                "decision": decision.model_dump(mode="json"),
                "trusted_action": action.model_dump(mode="json", exclude={"metadata"}),
                "observation": _public_observation(observation),
                "hard_reward": observation.reward,
                "hard_reward_components": observation.reward_components.model_dump(mode="json"),
                "failure_codes": observation.failure_codes,
            }
        )
    trajectory = {"seed": seed, "initial_observation": initial, "steps": steps}
    trajectory["trajectory_sha256"] = payload_sha256(trajectory)
    return seal_artifact(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "policy": policy_metadata,
            "episode": trajectory,
            "claims": {
                "model_weight_updates": False,
                "observation_only": True,
                "hard_reward_is_contract_compliance": True,
            },
        }
    )


def replay_agent_artifact(artifact: dict[str, Any]) -> str:
    if not verify_artifact(artifact):
        raise ValueError("ARTIFACT_INTEGRITY_FAILURE")
    episode = artifact["episode"]
    decisions = [AgentDecision.model_validate(s["decision"]) for s in episode["steps"]]
    replayed = run_decision_episode(episode["seed"], decisions, artifact["policy"])
    if replayed["episode"]["trajectory_sha256"] != episode["trajectory_sha256"]:
        raise ValueError("DETERMINISTIC_REPLAY_FAILURE")
    return replayed["episode"]["trajectory_sha256"]
```

Add a CLI that reads a JSON object with keys `policy` and `episodes`, where each episode is `{"seed": int, "decisions": [AgentDecision x4]}`, seals every episode, verifies replay, and writes stable pretty JSON plus a final newline to `--output`. Required arguments are `--decisions` and `--output`; reject duplicate or missing seeds.

- [ ] **Step 6: Run tests and tighten types**

Run:

```bash
uv run pytest tests/test_agent_trial.py -q
uv run mypy src/yc_founder_decision_env/agent_trial.py src/yc_founder_decision_env/models.py
uv run ruff check src/yc_founder_decision_env/agent_trial.py tests/test_agent_trial.py
```

Expected: all PASS.

- [ ] **Step 7: Commit the adapter**

```bash
git add src/yc_founder_decision_env/models.py src/yc_founder_decision_env/agent_trial.py tests/test_agent_trial.py
git commit -m "feat: add observation-only agent trial adapter"
```

---

### Task 3: Define the separate synthetic utility

**Files:**
- Create: `src/yc_founder_decision_env/rewards.py`
- Create: `tests/test_q_learning.py`

**Interfaces:**
- Produces: `SYNTHETIC_UTILITY_VERSION = "synthetic-utility-v0.1.0"` and `synthetic_utility(before, after) -> float`.
- Consumes only consecutive public observations; no action preference or hidden transition lookup.

- [ ] **Step 1: Write failing exact-formula tests**

Create `tests/test_q_learning.py` with:

```python
from yc_founder_decision_env.agent_trial import build_trusted_action
from yc_founder_decision_env.models import AgentDecision
from yc_founder_decision_env.rewards import SYNTHETIC_UTILITY_VERSION, synthetic_utility
from yc_founder_decision_env.server.environment import FounderDecisionEnvironment


def take(seed: int, action_type: str):
    env = FounderDecisionEnvironment()
    before = env.reset(seed=seed)
    after = env.step(
        build_trusted_action(
            before,
            AgentDecision(
                action_type=action_type,
                rationale="Visible state test.",
                source_locator=before.source_locators[0],
            ),
        )
    )
    return before, after


def test_synthetic_utility_is_versioned_and_separate_from_hard_reward() -> None:
    before, after = take(0, "sell_pilot")
    assert SYNTHETIC_UTILITY_VERSION == "synthetic-utility-v0.1.0"
    assert after.reward == 1.0
    assert synthetic_utility(before, after) == 0.7125
    assert synthetic_utility(before, after) != after.reward


def test_rejected_action_receives_fixed_synthetic_penalty() -> None:
    before, after = take(0, "fundraise")  # seed 0 disallows fundraise
    assert "ACTION_NOT_ALLOWED" in after.failure_codes
    assert synthetic_utility(before, after) == -1.0


def test_utility_uses_public_state_deltas_only() -> None:
    before, after = take(0, "abstain")
    assert synthetic_utility(before, after) == -0.002083
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/test_q_learning.py -v`

Expected: collection FAIL because `rewards.py` does not exist.

- [ ] **Step 3: Implement the frozen formula**

Create `rewards.py`:

```python
"""Synthetic benchmark utility, strictly separate from hard verifier reward."""

from .models import FounderObservation

SYNTHETIC_UTILITY_VERSION = "synthetic-utility-v0.1.0"


def synthetic_utility(before: FounderObservation, after: FounderObservation) -> float:
    """Score public state change; this is not evidence of real startup quality."""
    if after.reward != 1.0 or after.failure_codes:
        return -1.0
    value = (
        0.45 * ((after.mrr_cents - before.mrr_cents) / 15_000)
        + 0.25 * ((after.qualified_pipeline - before.qualified_pipeline) / 2)
        + 0.15 * ((after.active_users - before.active_users) / 3)
        + 0.10 * ((after.interviews_completed - before.interviews_completed) / 4)
        + 0.05 * ((after.price_cents - before.price_cents) / 500)
        - 0.05 * ((before.budget_cents - after.budget_cents) / 60_000)
        - 0.05 * ((before.founder_hours - after.founder_hours) / 24)
    )
    return round(value, 6)
```

The normalized positive terms sum to 1.0 for a hypothetical step achieving every reference delta; the two resource penalties are each bounded relative to the largest single action cost. Do not add hard reward to this number.

- [ ] **Step 4: Verify RED-to-GREEN and regression**

Run:

```bash
uv run pytest tests/test_q_learning.py -q
uv run pytest tests/test_environment.py -q
uv run ruff check src/yc_founder_decision_env/rewards.py tests/test_q_learning.py
```

Expected: all PASS.

- [ ] **Step 5: Commit the utility contract**

```bash
git add src/yc_founder_decision_env/rewards.py tests/test_q_learning.py
git commit -m "feat: define frozen synthetic RL utility"
```

---

### Task 4: Implement real tabular Q-learning

**Files:**
- Create: `src/yc_founder_decision_env/q_learning.py`
- Modify: `tests/test_q_learning.py`

**Interfaces:**
- Produces: `EncodedState`, `encode_state`, `available_actions`, `q_update`, `train_q_learning`, `greedy_q_decision`, `serialize_q_table`, and `QTrainingConfig`.
- State key: `(remaining_steps, budget_thousands, founder_hours, allowed_mask)`. Case ID and outcome counters are intentionally excluded because, under the frozen transition/utility contract, they do not affect future feasibility or marginal reward.
- The Q-table is local learned state. No model API and no language-model weights are updated.

- [ ] **Step 1: Add failing encoder and update tests**

Append to `tests/test_q_learning.py`:

```python
import math

from yc_founder_decision_env.q_learning import (
    QTrainingConfig,
    available_actions,
    encode_state,
    q_update,
    train_q_learning,
)


def test_state_encoder_uses_only_public_markov_features() -> None:
    observation = FounderDecisionEnvironment().reset(seed=0)
    encoded = encode_state(observation)
    assert encoded == (4, 120, 64, encoded[3])
    assert encoded[3] > 0


def test_available_actions_are_allowed_and_affordable() -> None:
    observation = FounderDecisionEnvironment().reset(seed=0)
    names = available_actions(observation)
    assert "fundraise" not in names
    assert "abstain" in names


def test_q_update_uses_visit_dependent_alpha_and_gamma() -> None:
    q: dict[tuple[tuple[int, int, int, int], str], float] = {}
    visits: dict[tuple[tuple[int, int, int, int], str], int] = {}
    state = (4, 120, 64, 31)
    next_state = (3, 110, 50, 31)
    q[(next_state, "sell_pilot")] = 2.0
    trace = q_update(q, visits, state, "sell_pilot", 0.5, next_state, False)
    assert trace["visit"] == 1
    assert trace["alpha"] == 1.0
    assert trace["target"] == 2.4  # 0.5 + 0.95 * 2.0
    assert q[(state, "sell_pilot")] == 2.4
    second = q_update(q, visits, state, "sell_pilot", 0.5, next_state, True)
    assert second["visit"] == 2
    assert math.isclose(second["alpha"], 1 / (2**0.6))
    assert second["target"] == 0.5


def test_training_is_deterministic_and_train_only() -> None:
    config = QTrainingConfig(rng_seed=20260723, episodes=20)
    first = train_q_learning(config)
    second = train_q_learning(config)
    assert first == second
    assert first["training_seeds"] == list(range(16))
    assert first["held_out_seeds_seen"] == []
    assert len(first["episode_returns"]) == 20
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run pytest tests/test_q_learning.py::test_state_encoder_uses_only_public_markov_features tests/test_q_learning.py::test_q_update_uses_visit_dependent_alpha_and_gamma tests/test_q_learning.py::test_training_is_deterministic_and_train_only -v
```

Expected: collection FAIL because `q_learning.py` does not exist.

- [ ] **Step 3: Implement configuration, encoding, and public action filtering**

Create `q_learning.py` with:

```python
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias

from .agent_trial import build_trusted_action, seal_artifact
from .models import ActionName, AgentDecision, FounderObservation
from .rewards import SYNTHETIC_UTILITY_VERSION, synthetic_utility
from .server.environment import ACTION_ORDER, FounderDecisionEnvironment

EncodedState: TypeAlias = tuple[int, int, int, int]
QKey: TypeAlias = tuple[EncodedState, ActionName]


@dataclass(frozen=True)
class QTrainingConfig:
    rng_seed: int
    episodes: int = 1000
    gamma: float = 0.95
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    alpha_exponent: float = 0.6


def encode_state(observation: FounderObservation) -> EncodedState:
    allowed_mask = sum(
        (1 << index) for index, spec in enumerate(observation.action_specs) if spec.allowed
    )
    return (
        4 - observation.week,
        observation.budget_cents // 1000,
        observation.founder_hours,
        allowed_mask,
    )


def available_actions(observation: FounderObservation) -> list[ActionName]:
    return [
        spec.action_type
        for spec in observation.action_specs
        if spec.allowed
        and spec.spend_cents <= observation.budget_cents
        and spec.founder_hours <= observation.founder_hours
    ]


def epsilon_at(config: QTrainingConfig, episode_index: int) -> float:
    if config.episodes == 1:
        return config.epsilon_end
    progress = episode_index / (config.episodes - 1)
    return max(config.epsilon_end, config.epsilon_start * (1 - progress))
```

For the required 1000-episode run this evaluates exactly to `max(0.05, 0.30 * (1 - episode_index / 999))`.

- [ ] **Step 4: Implement the exact update and deterministic tie-breaking**

Continue `q_learning.py`:

```python
def q_update(
    q: dict[QKey, float],
    visits: dict[QKey, int],
    state: EncodedState,
    action: ActionName,
    reward: float,
    next_state: EncodedState,
    done: bool,
    *,
    gamma: float = 0.95,
    alpha_exponent: float = 0.6,
) -> dict[str, Any]:
    key = (state, action)
    visits[key] = visits.get(key, 0) + 1
    alpha = 1 / (visits[key] ** alpha_exponent)
    old = q.get(key, 0.0)
    next_values = [q.get((next_state, candidate), 0.0) for candidate in ACTION_ORDER]
    target = reward if done else reward + gamma * max(next_values)
    q[key] = old + alpha * (target - old)
    return {
        "state": list(state),
        "action": action,
        "reward": reward,
        "next_state": list(next_state),
        "done": done,
        "visit": visits[key],
        "alpha": alpha,
        "old_q": old,
        "target": target,
        "new_q": q[key],
    }


def greedy_action(
    q: dict[QKey, float], observation: FounderObservation
) -> ActionName:
    state = encode_state(observation)
    feasible = set(available_actions(observation))
    ranked = [name for name in ACTION_ORDER if name in feasible]
    if not ranked:
        return "abstain"
    return max(ranked, key=lambda name: (q.get((state, name), 0.0), -ACTION_ORDER.index(name)))
```

The `ACTION_ORDER` position is the only tie-break rule and is frozen before held-out evaluation.

- [ ] **Step 5: Implement the 1000-episode training loop and serialization**

Continue `q_learning.py`:

```python
def _decision(observation: FounderObservation, action: ActionName, label: str) -> AgentDecision:
    return AgentDecision(
        action_type=action,
        rationale=f"{label}; public observation only.",
        source_locator=observation.source_locators[0],
    )


def serialize_q_table(q: dict[QKey, float]) -> list[dict[str, Any]]:
    return [
        {"state": list(state), "action": action, "q": round(value, 12)}
        for (state, action), value in sorted(q.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def train_q_learning(config: QTrainingConfig) -> dict[str, Any]:
    rng = random.Random(config.rng_seed)
    q: dict[QKey, float] = {}
    visits: dict[QKey, int] = {}
    returns: list[float] = []
    trace: list[dict[str, Any]] = []
    for episode_index in range(config.episodes):
        seed = rng.choice(tuple(range(16)))
        env = FounderDecisionEnvironment()
        observation = env.reset(seed=seed)
        episode_return = 0.0
        while not observation.done:
            state = encode_state(observation)
            feasible = available_actions(observation)
            epsilon = epsilon_at(config, episode_index)
            if rng.random() < epsilon:
                action = rng.choice(feasible)
                mode = "explore"
            else:
                action = greedy_action(q, observation)
                mode = "greedy"
            after = env.step(build_trusted_action(observation, _decision(observation, action, mode)))
            utility = synthetic_utility(observation, after)
            update = q_update(
                q,
                visits,
                state,
                action,
                utility,
                encode_state(after),
                after.done,
                gamma=config.gamma,
                alpha_exponent=config.alpha_exponent,
            )
            if episode_index in {0, config.episodes - 1}:
                trace.append({"episode": episode_index, "seed": seed, **update})
            episode_return += utility
            observation = after
        returns.append(round(episode_return, 6))
    return {
        "config": asdict(config),
        "training_seeds": list(range(16)),
        "held_out_seeds_seen": [],
        "utility_version": SYNTHETIC_UTILITY_VERSION,
        "episode_returns": returns,
        "curve_mean_every_50": [
            round(sum(returns[i : i + 50]) / len(returns[i : i + 50]), 6)
            for i in range(0, len(returns), 50)
        ],
        "q_table": serialize_q_table(q),
        "visit_counts": [
            {"state": list(state), "action": action, "count": count}
            for (state, action), count in sorted(visits.items(), key=lambda item: item[0])
        ],
        "representative_updates": trace,
    }
```

Add `greedy_q_decision(observation, serialized_q_table)` to deserialize the table and return `_decision(..., "frozen-q")`. Add a CLI that trains all five required seeds, seals one `q-learning-artifact-v0.1.0` object, writes it to `--output`, and accepts `--episodes` only for tests/manual smoke runs (docs and committed artifact must use 1000).

- [ ] **Step 6: Verify unit and deterministic smoke tests**

Run:

```bash
uv run pytest tests/test_q_learning.py -q
uv run mypy src/yc_founder_decision_env/q_learning.py
uv run ruff check src/yc_founder_decision_env/q_learning.py tests/test_q_learning.py
uv run python -m yc_founder_decision_env.q_learning --episodes 20 --output /tmp/q-smoke.json
```

Expected: tests/typecheck/lint PASS and `/tmp/q-smoke.json` reports five runs, each with 20 returns and no held-out seeds.

- [ ] **Step 7: Commit the learner**

```bash
git add src/yc_founder_decision_env/q_learning.py tests/test_q_learning.py
git commit -m "feat: add deterministic tabular Q-learning"
```

---

### Task 5: Build leakage-safe policies and exhaustive held-out evaluation

**Files:**
- Create: `src/yc_founder_decision_env/rl_evaluation.py`
- Create: `tests/test_rl_evaluation.py`

**Interfaces:**
- Produces: `random_decision`, `black_box_rule_decision`, `exhaustive_oracle`, `evaluate_decisions`, and `build_held_out_report`.
- The exhaustive oracle may query/replay fresh environment instances and observe public observations. It may not read sidecars or import `load_bundle`.
- Terra decisions are immutable input, not generated inside this module.

- [ ] **Step 1: Write failing policy-boundary and comparison tests**

Create `tests/test_rl_evaluation.py`:

```python
import inspect
import random

from yc_founder_decision_env.rl_evaluation import (
    black_box_rule_decision,
    build_held_out_report,
    exhaustive_oracle,
    random_decision,
)
from yc_founder_decision_env.server.environment import FounderDecisionEnvironment


def test_rule_and_random_policies_accept_observation_not_sidecar() -> None:
    observation = FounderDecisionEnvironment().reset(seed=16)
    rule = black_box_rule_decision(observation)
    sampled = random_decision(observation, random.Random(7))
    assert rule.action_type in observation.allowed_actions
    assert sampled.action_type in observation.allowed_actions
    source = inspect.getsource(black_box_rule_decision)
    assert "preferred_action" not in source
    assert "ground_truth" not in source
    assert "load_bundle" not in source


def test_exhaustive_oracle_is_deterministic_and_replayable() -> None:
    first = exhaustive_oracle(16)
    second = exhaustive_oracle(16)
    assert first == second
    assert len(first["decisions"]) <= 4
    assert first["replay_equal"] is True


def test_report_uses_only_frozen_held_out_seeds_and_separate_scores() -> None:
    # Four abstentions per held-out case stand in for externally generated Terra input.
    terra = {
        str(seed): [
            {
                "action_type": "abstain",
                "rationale": "Public observation only.",
                "source_locator": FounderDecisionEnvironment().reset(seed=seed).source_locators[0],
            }
            for _ in range(4)
        ]
        for seed in range(16, 24)
    }
    q_artifact = {"runs": []}  # test fixture uses zero-valued frozen Q fallback
    report = build_held_out_report(terra, q_artifact, random_seeds=[20260723])
    assert report["held_out_seeds"] == list(range(16, 24))
    assert set(report["policies"]) == {
        "random",
        "black_box_rule",
        "terra",
        "learned_q",
        "exhaustive_oracle",
    }
    for result in report["policies"].values():
        assert "mean_hard_reward" in result
        assert "mean_synthetic_utility" in result
        assert "combined_reward" not in result
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/test_rl_evaluation.py -v`

Expected: collection FAIL because `rl_evaluation.py` does not exist.

- [ ] **Step 3: Implement public random and black-box rule policies**

Create `rl_evaluation.py` and use this frozen rule logic:

```python
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any, Callable

from .agent_trial import build_trusted_action, payload_sha256, seal_artifact
from .models import ActionName, AgentDecision, FounderObservation
from .q_learning import available_actions, greedy_q_decision
from .rewards import SYNTHETIC_UTILITY_VERSION, synthetic_utility
from .server.environment import ACTION_ORDER, FounderDecisionEnvironment

HELD_OUT_SEEDS = tuple(range(16, 24))


def _decision(observation: FounderObservation, action: ActionName, label: str) -> AgentDecision:
    return AgentDecision(
        action_type=action,
        rationale=f"{label}; public observation only.",
        source_locator=observation.source_locators[0],
    )


def random_decision(observation: FounderObservation, rng: random.Random) -> AgentDecision:
    return _decision(observation, rng.choice(available_actions(observation)), "seeded-random")


def black_box_rule_decision(observation: FounderObservation) -> AgentDecision:
    feasible = set(available_actions(observation))
    if observation.interviews_completed < 4 and "interview_users" in feasible:
        chosen: ActionName = "interview_users"
    elif observation.qualified_pipeline < 2 and "sell_pilot" in feasible:
        chosen = "sell_pilot"
    elif observation.mrr_cents < 15_000 and "sell_pilot" in feasible:
        chosen = "sell_pilot"
    elif observation.price_cents < 3_000 and "change_price" in feasible:
        chosen = "change_price"
    else:
        chosen = next(name for name in ACTION_ORDER if name in feasible)
    return _decision(observation, chosen, "frozen-black-box-rule-v0.1.0")
```

Do not import `.data` or read any asset in this module.

- [ ] **Step 4: Implement one common evaluator and the black-box exhaustive oracle**

Continue `rl_evaluation.py`:

```python
def evaluate_decisions(seed: int, decisions: list[AgentDecision]) -> dict[str, Any]:
    env = FounderDecisionEnvironment()
    observation = env.reset(seed=seed)
    steps: list[dict[str, Any]] = []
    total_utility = 0.0
    for decision in decisions:
        if observation.done:
            break
        before = observation
        observation = env.step(build_trusted_action(before, decision))
        utility = synthetic_utility(before, observation)
        total_utility += utility
        steps.append(
            {
                "decision": decision.model_dump(mode="json"),
                "observation_sha256": payload_sha256(observation.model_dump(mode="json")),
                "hard_reward": observation.reward,
                "hard_reward_components": observation.reward_components.model_dump(mode="json"),
                "synthetic_utility": utility,
                "failure_codes": observation.failure_codes,
            }
        )
    result = {
        "seed": seed,
        "steps": steps,
        "total_hard_reward": round(sum(s["hard_reward"] for s in steps), 6),
        "total_synthetic_utility": round(total_utility, 6),
        "final_state": env.state.model_dump(mode="json"),
    }
    result["trajectory_sha256"] = payload_sha256(result)
    return result


def exhaustive_oracle(seed: int) -> dict[str, Any]:
    best: tuple[float, tuple[int, ...], list[AgentDecision], dict[str, Any]] | None = None
    for indices in itertools.product(range(len(ACTION_ORDER)), repeat=4):
        env = FounderDecisionEnvironment()
        observation = env.reset(seed=seed)
        decisions: list[AgentDecision] = []
        valid_prefix = True
        for index in indices:
            action = ACTION_ORDER[index]
            if action not in available_actions(observation):
                valid_prefix = False
                break
            decision = _decision(observation, action, "exhaustive-black-box-oracle")
            decisions.append(decision)
            observation = env.step(build_trusted_action(observation, decision))
            if observation.done:
                break
        if not valid_prefix:
            continue
        result = evaluate_decisions(seed, decisions)
        rank = (result["total_synthetic_utility"], tuple(-i for i in indices))
        candidate = (rank[0], rank[1], decisions, result)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("ORACLE_FOUND_NO_TRAJECTORY")
    replay = evaluate_decisions(seed, best[2])
    return {
        "decisions": [d.model_dump(mode="json") for d in best[2]],
        "result": best[3],
        "replay_equal": replay["trajectory_sha256"] == best[3]["trajectory_sha256"],
    }
```

The oracle is a benchmark upper bound with simulator query access, not a deployable observation policy. Its search order is deterministic and its access advantage must be labeled in the report.

- [ ] **Step 5: Aggregate all five families without mixing rewards**

Implement `build_held_out_report(terra, q_artifact, random_seeds)` so that:

1. It asserts Terra keys are exactly strings `"16".."23"` and validates every decision with `AgentDecision`.
2. Random is evaluated for every combination of the eight held-out cases and five required RNG seeds; its aggregate includes `n_episodes=40` and per-RNG subaggregates.
3. Black-box rule, Terra, and exhaustive oracle each have `n_episodes=8`.
4. Learned Q evaluates every one of the five frozen Q tables on all eight cases (`n_episodes=40`) and provides per-training-seed subaggregates. If a test fixture has no runs, use an all-zero table and label it `test-zero-q`; production CLI rejects missing runs.
5. Each aggregate contains `mean_hard_reward`, `mean_synthetic_utility`, component means, failure-code counts, episode trajectories, and trajectory hashes as separate fields.
6. Top-level metadata records dataset revision `0.1.0`, transition model `frozen-v0.1.0`, verifier `v0.1.0`, utility version, exact split, and `combined_reward: null` with reason `hard reward and synthetic utility are intentionally not scalarized`.

Add `verify_held_out_replay(report)` that recreates every non-oracle trajectory from stored decisions and matches `trajectory_sha256`; for oracle, rerun search and compare the chosen trajectory. Add a CLI requiring `--terra-decisions`, `--q-artifact`, and `--output`.

- [ ] **Step 6: Run focused tests and a one-seed oracle smoke test**

Run:

```bash
uv run pytest tests/test_rl_evaluation.py -q
uv run python -c 'from yc_founder_decision_env.rl_evaluation import exhaustive_oracle; print(exhaustive_oracle(16)["replay_equal"])'
uv run mypy src/yc_founder_decision_env/rl_evaluation.py
uv run ruff check src/yc_founder_decision_env/rl_evaluation.py tests/test_rl_evaluation.py
```

Expected: all checks PASS and the smoke command prints `True`.

- [ ] **Step 7: Commit the evaluator**

```bash
git add src/yc_founder_decision_env/rl_evaluation.py tests/test_rl_evaluation.py
git commit -m "feat: add held-out RL policy comparison"
```

---

### Task 6: Run real Terra inference episodes without hidden context

**Files:**
- Generate: `artifacts/rl/terra-held-out-decisions-v0.1.0.json`
- Generate: `artifacts/rl/terra-agent-episodes-v0.1.0.json`

**Interfaces:**
- Consumes: serialized public observations and `AgentDecision.model_json_schema()`.
- Produces: four sequential decisions for each seed `16..23`, model provenance, per-turn input observation hashes, and replayable episode artifacts.

- [ ] **Step 1: Freeze the neutral Terra instruction**

Use this exact system/task instruction for every Terra subagent turn:

```text
You are selecting one action in a frozen four-step synthetic founder-decision benchmark.
Use only the JSON observation in this message. Choose exactly one action. Return JSON only,
matching the supplied AgentDecision schema. Do not infer or request preferred_action,
ground_truth, transition tables, future observations, Q values, or oracle output. Costs and
allowed flags in action_specs are authoritative. The source_locator must be copied exactly
from source_locators. This is inference only; no model weights are updated.
```

The turn payload is exactly:

```json
{
  "observation": "<FounderObservation.model_dump(mode='json') object>",
  "decision_schema": "<AgentDecision.model_json_schema() object>"
}
```

Here angle-bracket strings describe runtime substitutions; they must not be written into artifacts. No other repository content enters the prompt.

- [ ] **Step 2: Launch one Codex Terra subagent per held-out seed**

From the executing Codex session, launch eight subagents with explicit model `gpt-5.6-terra`, one for each seed `16..23`. Assign each subagent one persistent episode. At each of four turns:

1. Reset a local environment for that subagent's seed on turn 0, or use the observation returned by the previous accepted/rejected step.
2. Compute and record `payload_sha256(observation.model_dump(mode="json"))` before sending.
3. Send only the frozen instruction, observation, and JSON schema.
4. Parse the response with `AgentDecision.model_validate_json`; if parsing fails, send only the Pydantic error plus the same observation/schema for one repair attempt and record both raw responses.
5. Build the technical fields with `build_trusted_action` and step the environment. Never replace or improve the chosen semantic action.
6. Stop if `done`; otherwise continue with the new public observation.

Expected: at least seed 16 completes a real four-step episode unless resource termination occurs earlier. The artifact truthfully records early termination rather than padding fabricated steps.

- [ ] **Step 3: Save and validate the decision ledger**

Write `terra-held-out-decisions-v0.1.0.json` with:

```json
{
  "schema_version": "terra-decision-ledger-v0.1.0",
  "policy": {
    "provider": "codex-subagent",
    "model": "gpt-5.6-terra",
    "mode": "inference",
    "model_weight_updates": false,
    "prompt_version": "observation-only-v0.1.0"
  },
  "episodes": {
    "16": {"turns": []},
    "17": {"turns": []},
    "18": {"turns": []},
    "19": {"turns": []},
    "20": {"turns": []},
    "21": {"turns": []},
    "22": {"turns": []},
    "23": {"turns": []}
  },
  "integrity_sha256": "computed by seal_artifact"
}
```

Each turn contains `input_observation_sha256`, validated `decision`, `raw_response`, `repair_response` (`null` if unused), and `parse_attempts`. Seal the artifact and run `verify_artifact`.

- [ ] **Step 4: Materialize and replay agent episodes**

Transform each episode's validated decisions through `run_decision_episode`, then create the sealed `terra-agent-episodes-v0.1.0.json`. Run:

```bash
uv run python -m yc_founder_decision_env.agent_trial \
  --decisions artifacts/rl/terra-held-out-decisions-v0.1.0.json \
  --output artifacts/rl/terra-agent-episodes-v0.1.0.json
uv run python -c 'import json; from yc_founder_decision_env.agent_trial import replay_agent_artifact; p=json.load(open("artifacts/rl/terra-agent-episodes-v0.1.0.json")); [replay_agent_artifact(x) for x in p["episodes"].values()]; print("terra replay PASS")'
```

Expected: `terra replay PASS`. Check that the seed-16 episode contains the required real model provenance and up to four actual steps.

- [ ] **Step 5: Commit the immutable Terra evidence**

```bash
git add artifacts/rl/terra-held-out-decisions-v0.1.0.json artifacts/rl/terra-agent-episodes-v0.1.0.json
git commit -m "test: record observation-only Terra episodes"
```

---

### Task 7: Train five Q policies and run the frozen held-out comparison

**Files:**
- Generate: `artifacts/rl/q-learning-v0.1.0.json`
- Generate: `artifacts/rl/held-out-comparison-v0.1.0.json`
- Modify: `tests/test_q_learning.py`
- Modify: `tests/test_rl_evaluation.py`

**Interfaces:**
- Consumes: frozen code and Terra artifacts from prior tasks.
- Produces: deterministic training/evaluation artifacts that become the sole numeric source for documentation.

- [ ] **Step 1: Generate the full Q-learning artifact**

Run:

```bash
uv run python -m yc_founder_decision_env.q_learning \
  --episodes 1000 \
  --output artifacts/rl/q-learning-v0.1.0.json
```

Expected artifact assertions:

```python
assert [run["config"]["rng_seed"] for run in artifact["runs"]] == list(range(20260723, 20260728))
assert all(run["config"]["episodes"] == 1000 for run in artifact["runs"])
assert all(run["training_seeds"] == list(range(16)) for run in artifact["runs"])
assert all(run["held_out_seeds_seen"] == [] for run in artifact["runs"])
assert verify_artifact(artifact)
```

- [ ] **Step 2: Regenerate and byte-compare Q-learning output**

Run:

```bash
uv run python -m yc_founder_decision_env.q_learning --episodes 1000 --output /tmp/q-learning-replay.json
cmp artifacts/rl/q-learning-v0.1.0.json /tmp/q-learning-replay.json
shasum -a 256 artifacts/rl/q-learning-v0.1.0.json /tmp/q-learning-replay.json
```

Expected: `cmp` exits 0 and both SHA-256 values match.

- [ ] **Step 3: Run the full held-out comparison**

Run:

```bash
uv run python -m yc_founder_decision_env.rl_evaluation \
  --terra-decisions artifacts/rl/terra-held-out-decisions-v0.1.0.json \
  --q-artifact artifacts/rl/q-learning-v0.1.0.json \
  --output artifacts/rl/held-out-comparison-v0.1.0.json
```

Expected: seeds are exactly `16..23`; episode counts are random 40, black-box rule 8, Terra 8, learned Q 40, and exhaustive oracle 8; every trajectory replay passes; hard reward and synthetic utility are separate.

- [ ] **Step 4: Add committed-artifact regression tests**

Append tests that load both artifacts from repository root and assert their embedded hashes, exact configs, exact splits, episode counts, replay result, and no forbidden strings. The critical assertions are:

```python
assert "preferred_action" not in json.dumps(comparison)
assert "ground_truth" not in json.dumps(comparison)
assert q_artifact["integrity_sha256"]
assert comparison["integrity_sha256"]
assert comparison["combined_reward"] is None
assert comparison["replay"]["passed"] is True
```

Do not assert that learned Q must beat Terra or the rule; report the observed outcome without moving the gate after seeing held-out results.

- [ ] **Step 5: Run artifact tests and deterministic replay**

Run:

```bash
uv run pytest tests/test_q_learning.py tests/test_rl_evaluation.py -q
uv run python -m yc_founder_decision_env.rl_evaluation \
  --terra-decisions artifacts/rl/terra-held-out-decisions-v0.1.0.json \
  --q-artifact artifacts/rl/q-learning-v0.1.0.json \
  --output /tmp/held-out-replay.json
cmp artifacts/rl/held-out-comparison-v0.1.0.json /tmp/held-out-replay.json
```

Expected: tests PASS and `cmp` exits 0.

- [ ] **Step 6: Commit training and evaluation evidence**

```bash
git add artifacts/rl/q-learning-v0.1.0.json artifacts/rl/held-out-comparison-v0.1.0.json tests/test_q_learning.py tests/test_rl_evaluation.py
git commit -m "test: record Q-learning and held-out results"
```

---

### Task 8: Add commands, experiment documentation, and final verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Create: `docs/RL_EXPERIMENT.md`

**Interfaces:**
- Produces: `ycfd-agent-trial`, `ycfd-q-learning`, and `ycfd-rl-evaluate` commands plus a source-backed result report.
- Documentation reads numeric results from the committed artifacts; it must not hand-calculate or round inconsistently.

- [ ] **Step 1: Add CLI smoke tests before entry points**

Add a parametrized subprocess test to `tests/test_rl_evaluation.py` that invokes each module with `--help` and asserts exit code 0:

```python
@pytest.mark.parametrize(
    "module",
    [
        "yc_founder_decision_env.agent_trial",
        "yc_founder_decision_env.q_learning",
        "yc_founder_decision_env.rl_evaluation",
    ],
)
def test_rl_cli_help(module: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
```

Add imports for `subprocess`, `sys`, and `pytest`.

- [ ] **Step 2: Run the CLI smoke test and verify current module CLIs**

Run: `uv run pytest tests/test_rl_evaluation.py::test_rl_cli_help -v`

Expected: PASS only after Tasks 2, 4, and 5 supplied working `main()` functions.

- [ ] **Step 3: Register console scripts**

Add under `[project.scripts]` in `pyproject.toml`:

```toml
ycfd-agent-trial = "yc_founder_decision_env.agent_trial:main"
ycfd-q-learning = "yc_founder_decision_env.q_learning:main"
ycfd-rl-evaluate = "yc_founder_decision_env.rl_evaluation:main"
```

Run `uv lock` and `uv sync --all-extras --frozen`.

- [ ] **Step 4: Document the protocol and actual results**

Create `docs/RL_EXPERIMENT.md` with these sections and exact content sources:

1. **What can and cannot be trained** — tabular Q values are updated; Codex/Claude/Terra weights are not. Claude Code can implement the same `ObservationPolicy` adapter or produce decisions, but this run uses Codex Terra.
2. **Information boundary** — list every `FounderObservation` field and explicitly list forbidden sidecar/canonical fields.
3. **Trial 1: Terra inference** — prompt version, model string, held-out seeds, step count, parse repairs, hard reward/components, synthetic utility, and artifact SHA-256 from Terra artifacts.
4. **Trial 2: Q-learning** — state tuple, action factory, utility equation, Bellman update `Q <- Q + N(s,a)^-0.6 * (r + 0.95 max_a' Q(s',a') - Q)`, epsilon schedule, five RNG seeds, 1000 episodes, learning-curve start/end, and artifact SHA-256.
5. **Trial 3: held-out comparison** — a table with policy, number of episodes, mean hard reward, mean synthetic utility, dispersion across seeds where applicable, and oracle gap. Populate values by reading `held-out-comparison-v0.1.0.json` only.
6. **Reproduce** — exact commands from Tasks 6 and 7 plus the final test commands.
7. **Interpretation and limitations** — frozen synthetic transition model, 24 small cases, shared action dynamics, tabular policy, oracle query advantage, no real-world startup-success or foundation-model-training claim.

The result text must say whether Q beat each baseline based on actual artifact values; it must not describe a non-result as success.

- [ ] **Step 5: Add concise README commands and boundary**

Append a `## Agent and RL trials` section to `README.md`:

```markdown
## Agent and RL trials

The environment supports observation-only inference-agent episodes and real tabular
Q-learning. The learner updates a local Q table under `synthetic-utility-v0.1.0`; it does
not update Codex, Claude, Terra, or other model weights. Hard verifier reward remains a
separate contract-compliance metric.

```bash
uv run ycfd-q-learning --episodes 1000 --output artifacts/rl/q-learning-v0.1.0.json
uv run ycfd-rl-evaluate \
  --terra-decisions artifacts/rl/terra-held-out-decisions-v0.1.0.json \
  --q-artifact artifacts/rl/q-learning-v0.1.0.json \
  --output artifacts/rl/held-out-comparison-v0.1.0.json
```

See [`docs/RL_EXPERIMENT.md`](docs/RL_EXPERIMENT.md) for the frozen protocol, Terra
adapter boundary, results, hashes, and replay instructions.
```

- [ ] **Step 6: Run clean final gates**

Run:

```bash
uv sync --all-extras --frozen
uv run pytest -q
uv run ruff check .
uv run mypy src/yc_founder_decision_env
uv run ycfd-demo >/tmp/ycfd-demo.json
uv run ycfd-agent-trial --help
uv run ycfd-q-learning --help
uv run ycfd-rl-evaluate --help
git diff --check
```

Expected: every command exits 0, all tests PASS, and `git diff --check` emits no output.

- [ ] **Step 7: Perform the no-leakage and artifact audit**

Run:

```bash
rg -n 'preferred_action|ground_truth|"transitions"' \
  src/yc_founder_decision_env/agent_trial.py \
  src/yc_founder_decision_env/q_learning.py \
  src/yc_founder_decision_env/rl_evaluation.py \
  artifacts/rl
shasum -a 256 artifacts/rl/*.json
```

Expected: `rg` returns no matches (exit 1 is the expected “not found” result); record the four file hashes in `docs/RL_EXPERIMENT.md`. The hashes inside artifacts cover their unsigned stable JSON payloads, while `shasum` covers the complete files; label them distinctly.

- [ ] **Step 8: Commit documentation and command surface**

```bash
git add pyproject.toml uv.lock README.md docs/RL_EXPERIMENT.md tests/test_rl_evaluation.py
git commit -m "docs: publish reproducible agent and RL trials"
```

---

## Acceptance Checklist

- [ ] A real Codex `gpt-5.6-terra` subagent produced sequential decisions for seed 16 from public observations only, and its episode replay passes.
- [ ] Terra coverage for comparison includes every frozen held-out seed `16..23`; counts and early termination are reported truthfully.
- [ ] Observation exposes exact costs/allowed flags for all six actions without exposing transition outcomes, preference labels, or canonical ground truth.
- [ ] Every semantic policy choice flows through the trusted action factory; policies cannot inject cost or state-claim fields.
- [ ] Five tabular Q-learning runs each contain exactly 1000 train-only episodes and the required RNG/gamma/alpha/epsilon settings.
- [ ] The committed update trace visibly includes state, action, utility reward, next state, done, visit count, alpha, target, old Q, and new Q.
- [ ] Random, black-box rule, Terra, learned Q, and exhaustive oracle are compared on held-out seeds with hard reward and synthetic utility kept separate.
- [ ] No code or artifact used for action selection contains `preferred_action`, `ground_truth`, or hidden transitions.
- [ ] Every artifact validates its embedded integrity hash and deterministic replay; full-file SHA-256 values are documented.
- [ ] Documentation clearly states that Codex/Claude/Terra weights were not and cannot be gradient-updated in this runtime.
- [ ] Full pytest, Ruff, mypy, demo, CLI smoke tests, and `git diff --check` pass.

## Expected Interpretation

This implementation can demonstrate all of: an LLM agent acting through an OpenEnv-style contract, a genuine RL loop from state encoding through tabular policy updates, and a frozen held-out comparison. It cannot demonstrate foundation-model reinforcement learning because there is no model-training backend, optimizer, gradient access, or weight checkpoint path for Codex/Claude/Terra. A later foundation-model RL experiment would require a trainable open-weight policy (or a provider-supported fine-tuning/RL API), a tokenizer/chat adapter, rollout workers, and an algorithm such as PPO/GRPO; that is deliberately outside this plan.
