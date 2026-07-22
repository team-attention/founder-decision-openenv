"""Observation-only agent episode recording and deterministic replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, cast

from .data import stable_json
from .models import AgentDecision, FounderAction, FounderObservation, StateClaim
from .server.environment import FounderDecisionEnvironment

ARTIFACT_SCHEMA_VERSION = "agent-episode-v0.1.0"
TERRA_LEDGER_SCHEMA_VERSION = "terra-decision-ledger-v0.1.0"
TERRA_INSTRUCTION = (
    "You are selecting one action in a frozen four-step synthetic founder-decision benchmark.\n"
    "Use only the JSON observation in this message. Choose exactly one action. Return JSON only,\n"
    "matching the supplied AgentDecision schema. Do not infer or request preferred_action,\n"
    "ground_truth, transition tables, future observations, Q values, or oracle output. Costs and\n"
    "allowed flags in action_specs are authoritative. The source_locator must be copied exactly\n"
    "from source_locators. This is inference only; no model weights are updated."
)
TERRA_POLICY = {
    "provider": "codex-subagent",
    "model": "gpt-5.6-terra",
    "mode": "inference",
    "model_weight_updates": False,
    "prompt_version": "observation-only-v0.1.0",
}
TERRA_ORCHESTRATION_TASK_IDS = {
    str(seed): f"/root/terra_policy_seed{seed}" for seed in range(16, 24)
}
TERRA_ORCHESTRATION_ATTESTATION = (
    "session-attested orchestration metadata; not cryptographically provider-signed by Codex"
)


class ObservationPolicy(Protocol):
    """A policy which can inspect only the current public observation."""

    def decide(self, observation: FounderObservation) -> AgentDecision: ...


def payload_sha256(payload: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-serializable payload."""
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def terra_request_contract() -> dict[str, Any]:
    """Return the immutable, observation-only request contract for every Terra turn."""
    decision_schema = AgentDecision.model_json_schema()
    return {
        "instruction": TERRA_INSTRUCTION,
        "decision_schema": decision_schema,
        "decision_schema_sha256": payload_sha256(decision_schema),
    }


def request_envelope_sha256(
    observation: FounderObservation, request_contract: dict[str, Any]
) -> str:
    """Hash the exact model-visible request envelope using stable JSON."""
    return payload_sha256(
        {
            "instruction": request_contract["instruction"],
            "observation": observation.model_dump(mode="json"),
            "decision_schema": request_contract["decision_schema"],
        }
    )


def validate_terra_ledger_contract(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate the sealed exact Terra request/model/seed orchestration contract."""
    if raw.get("schema_version") != TERRA_LEDGER_SCHEMA_VERSION:
        raise ValueError("INVALID_TERRA_LEDGER_SCHEMA")
    if not verify_artifact(raw):
        raise ValueError("TERRA_LEDGER_INTEGRITY_FAILURE")
    if raw.get("policy") != TERRA_POLICY:
        raise ValueError("INVALID_TERRA_PROVENANCE")
    request_contract = raw.get("request_contract")
    if not isinstance(request_contract, dict):
        raise ValueError("TERRA_REQUEST_CONTRACT_REQUIRED")
    expected_contract = terra_request_contract()
    if request_contract.get("decision_schema_sha256") != payload_sha256(
        request_contract.get("decision_schema")
    ):
        raise ValueError("TERRA_DECISION_SCHEMA_HASH_MISMATCH")
    if request_contract != expected_contract:
        raise ValueError("INVALID_TERRA_REQUEST_CONTRACT")
    if raw.get("orchestration_attestation") != TERRA_ORCHESTRATION_ATTESTATION:
        raise ValueError("TERRA_ORCHESTRATION_ATTESTATION_REQUIRED")
    episodes = raw.get("episodes")
    if not isinstance(episodes, dict) or set(episodes) != set(TERRA_ORCHESTRATION_TASK_IDS):
        raise ValueError("TERRA_HELD_OUT_SPLIT_REQUIRED")
    task_ids: list[str] = []
    for seed_text, expected_task_id in TERRA_ORCHESTRATION_TASK_IDS.items():
        episode = episodes.get(seed_text)
        if (
            not isinstance(episode, dict)
            or episode.get("orchestration_task_id") != expected_task_id
        ):
            raise ValueError("TERRA_ORCHESTRATION_TASK_ID_MISMATCH")
        task_ids.append(expected_task_id)
    if len(set(task_ids)) != len(TERRA_ORCHESTRATION_TASK_IDS):
        raise ValueError("TERRA_ORCHESTRATION_TASK_IDS_NOT_UNIQUE")
    return request_contract


def seal_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    """Add an integrity digest without mutating the supplied payload."""
    sealed = copy.deepcopy(payload)
    sealed.pop("integrity_sha256", None)
    sealed["integrity_sha256"] = payload_sha256(sealed)
    return sealed


def verify_artifact(payload: dict[str, Any]) -> bool:
    """Check whether an artifact's integrity digest matches its contents."""
    expected = payload.get("integrity_sha256")
    unsigned = copy.deepcopy(payload)
    unsigned.pop("integrity_sha256", None)
    return isinstance(expected, str) and expected == payload_sha256(unsigned)


def build_trusted_action(
    observation: FounderObservation, decision: AgentDecision
) -> FounderAction:
    """Build verifier-required fields exclusively from a public observation."""
    spec = next(
        spec for spec in observation.action_specs if spec.action_type == decision.action_type
    )
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


def _public_observation(observation: FounderObservation) -> dict[str, Any]:
    return observation.model_dump(mode="json")


def run_decision_episode(
    seed: int,
    decisions: list[AgentDecision],
    policy_metadata: dict[str, str],
) -> dict[str, Any]:
    """Record up to four decisions, requiring four unless the episode terminates."""
    if len(decisions) > 4:
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
    if not observation.done and len(decisions) < 4:
        raise ValueError("EXACTLY_FOUR_DECISIONS_REQUIRED")
    trajectory: dict[str, Any] = {"seed": seed, "initial_observation": initial, "steps": steps}
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
    """Replay an integrity-checked artifact and return its trajectory digest."""
    if not verify_artifact(artifact):
        raise ValueError("ARTIFACT_INTEGRITY_FAILURE")
    episode = artifact["episode"]
    if not isinstance(episode, dict):
        raise ValueError("INVALID_EPISODE")
    seed = episode.get("seed")
    steps = episode.get("steps")
    policy = artifact.get("policy")
    if not isinstance(seed, int) or not isinstance(steps, list) or not isinstance(policy, dict):
        raise ValueError("INVALID_EPISODE")
    decisions = [AgentDecision.model_validate(step["decision"]) for step in steps]
    replayed = run_decision_episode(seed, decisions, cast(dict[str, str], policy))
    trajectory_sha256 = episode.get("trajectory_sha256")
    if replayed["episode"]["trajectory_sha256"] != trajectory_sha256:
        raise ValueError("DETERMINISTIC_REPLAY_FAILURE")
    return cast(str, replayed["episode"]["trajectory_sha256"])


def _read_decision_input(path: Path) -> tuple[dict[str, Any], dict[str, list[AgentDecision]]]:
    """Load only a sealed structured Terra ledger, never decision shorthand."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("DECISIONS_OBJECT_REQUIRED")
    request_contract = validate_terra_ledger_contract(raw)
    policy = raw.get("policy")
    episodes = raw.get("episodes")
    if not isinstance(policy, dict) or not isinstance(episodes, dict):
        raise ValueError("EPISODES_REQUIRED")
    parsed: dict[str, list[AgentDecision]] = {}
    for seed_text, episode in episodes.items():
        if (
            not isinstance(seed_text, str)
            or not seed_text.isdecimal()
            or not isinstance(episode, dict)
        ):
            raise ValueError("INVALID_EPISODE")
        seed = int(seed_text)
        turns = episode.get("turns")
        if not isinstance(turns, list) or not turns or len(turns) > 4:
            raise ValueError("INVALID_TERRA_TURNS")
        env = FounderDecisionEnvironment()
        observation = env.reset(seed=seed)
        decisions: list[AgentDecision] = []
        for turn in turns:
            if observation.done or not isinstance(turn, dict):
                raise ValueError("INVALID_TERRA_TURN")
            if turn.get("input_observation_sha256") != payload_sha256(
                observation.model_dump(mode="json")
            ):
                raise ValueError("TERRA_OBSERVATION_HASH_MISMATCH")
            if turn.get("request_envelope_sha256") != request_envelope_sha256(
                observation, request_contract
            ):
                raise ValueError("TERRA_REQUEST_ENVELOPE_HASH_MISMATCH")
            raw_response = turn.get("raw_response")
            repair_response = turn.get("repair_response")
            attempts = turn.get("parse_attempts")
            if (
                not isinstance(raw_response, str)
                or repair_response is not None and not isinstance(repair_response, str)
                or not isinstance(attempts, int)
                or isinstance(attempts, bool)
                or attempts < 1
            ):
                raise ValueError("INVALID_TERRA_RESPONSE")
            decision = AgentDecision.model_validate(turn.get("decision"))
            selected = raw_response if repair_response is None else repair_response
            if AgentDecision.model_validate_json(selected) != decision:
                raise ValueError("TERRA_RESPONSE_DECISION_MISMATCH")
            decisions.append(decision)
            observation = env.step(build_trusted_action(observation, decision))
        if not observation.done and len(decisions) < 4:
            raise ValueError("TERRA_INCOMPLETE_NONTERMINAL_EPISODE")
        parsed[seed_text] = decisions
    return cast(dict[str, Any], policy), parsed


def _seal_input_episodes(path: Path) -> dict[str, dict[str, Any]]:
    policy, episodes = _read_decision_input(path)
    artifacts: dict[str, dict[str, Any]] = {}
    for seed_text, decisions in episodes.items():
        artifact = run_decision_episode(int(seed_text), decisions, policy)
        replay_agent_artifact(artifact)
        artifacts[seed_text] = artifact
    return artifacts


def main() -> None:
    """Seal and replay decision episodes supplied as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    artifacts = seal_artifact(
        {
            "schema_version": "terra-agent-episodes-v0.1.0",
            "episodes": _seal_input_episodes(args.decisions),
        }
    )
    args.output.write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
