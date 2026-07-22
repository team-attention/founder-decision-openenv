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


class ObservationPolicy(Protocol):
    """A policy which can inspect only the current public observation."""

    def decide(self, observation: FounderObservation) -> AgentDecision: ...


def payload_sha256(payload: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-serializable payload."""
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


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


def _read_decision_input(path: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("DECISIONS_OBJECT_REQUIRED")
    policy = raw.get("policy")
    episodes = raw.get("episodes")
    if not isinstance(policy, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in policy.items()
    ):
        raise ValueError("POLICY_METADATA_REQUIRED")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("EPISODES_REQUIRED")
    parsed = [cast(dict[str, Any], item) for item in episodes if isinstance(item, dict)]
    if len(parsed) != len(episodes):
        raise ValueError("INVALID_EPISODE")
    return cast(dict[str, str], policy), parsed


def _seal_input_episodes(path: Path) -> list[dict[str, Any]]:
    policy, episodes = _read_decision_input(path)
    seen_seeds: set[int] = set()
    artifacts: list[dict[str, Any]] = []
    for item in episodes:
        seed = item.get("seed")
        decisions = item.get("decisions")
        if not isinstance(seed, int) or isinstance(seed, bool) or not isinstance(decisions, list):
            raise ValueError("EPISODE_SEED_AND_DECISIONS_REQUIRED")
        if seed in seen_seeds:
            raise ValueError("DUPLICATE_SEED")
        seen_seeds.add(seed)
        parsed_decisions = [AgentDecision.model_validate(value) for value in decisions]
        artifact = run_decision_episode(seed, parsed_decisions, policy)
        replay_agent_artifact(artifact)
        artifacts.append(artifact)
    return artifacts


def main() -> None:
    """Seal and replay decision episodes supplied as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    artifacts = _seal_input_episodes(args.decisions)
    args.output.write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
