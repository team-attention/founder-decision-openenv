import copy

import pytest
from pydantic import ValidationError

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
    with pytest.raises(ValidationError):
        AgentDecision.model_validate(
            {
                **decision().model_dump(),
                "spend_cents": 0,
                "claim": {"observed_step": 99},
            }
        )
