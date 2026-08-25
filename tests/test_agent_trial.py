import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from yc_founder_decision_env.agent_trial import (
    TERRA_ORCHESTRATION_ATTESTATION,
    TERRA_ORCHESTRATION_TASK_IDS,
    _seal_input_episodes,
    build_trusted_action,
    payload_sha256,
    replay_agent_artifact,
    request_envelope_sha256,
    run_decision_episode,
    seal_artifact,
    terra_request_contract,
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


def test_replay_accepts_an_episode_that_terminates_before_fourth_decision() -> None:
    artifact = run_decision_episode(
        20,
        [
            decision("build_feature"),
            decision("build_feature"),
            decision("abstain"),
            decision("abstain"),
        ],
        {"provider": "codex", "model": "gpt-5.6-terra", "mode": "inference"},
    )

    assert len(artifact["episode"]["steps"]) == 2
    assert artifact["episode"]["steps"][-1]["observation"]["done"] is True
    assert replay_agent_artifact(artifact) == artifact["episode"]["trajectory_sha256"]


def test_episode_rejects_insufficient_decisions_when_not_terminal() -> None:
    with pytest.raises(ValueError, match="EXACTLY_FOUR_DECISIONS_REQUIRED"):
        run_decision_episode(
            0,
            [decision("abstain") for _ in range(3)],
            {"provider": "codex", "model": "gpt-5.6-terra", "mode": "inference"},
        )


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


def test_sealed_terra_ledger_is_required_for_episode_materialization(tmp_path: Path) -> None:
    request_contract = terra_request_contract()
    decision_value = decision("abstain").model_dump(mode="json")
    episodes: dict[str, dict[str, object]] = {}
    for seed in range(16, 24):
        env = FounderDecisionEnvironment()
        observation = env.reset(seed=seed)
        turns: list[dict[str, object]] = []
        for _ in range(4):
            turns.append(
                {
                    "input_observation_sha256": payload_sha256(
                        observation.model_dump(mode="json")
                    ),
                    "request_envelope_sha256": request_envelope_sha256(
                        observation, request_contract
                    ),
                    "decision": decision_value,
                    "raw_response": json.dumps(
                        decision_value, ensure_ascii=False, separators=(",", ":")
                    ),
                    "repair_response": None,
                    "parse_attempts": 1,
                }
            )
            observation = env.step(
                build_trusted_action(observation, AgentDecision.model_validate(decision_value))
            )
        episodes[str(seed)] = {
            "orchestration_task_id": TERRA_ORCHESTRATION_TASK_IDS[str(seed)],
            "turns": turns,
        }
    ledger = seal_artifact(
        {
            "schema_version": "terra-decision-ledger-v0.1.0",
            "policy": {
                "provider": "codex-subagent",
                "model": "gpt-5.6-terra",
                "mode": "inference",
                "model_weight_updates": False,
                "prompt_version": "observation-only-v0.1.0",
            },
            "request_contract": request_contract,
            "orchestration_attestation": TERRA_ORCHESTRATION_ATTESTATION,
            "episodes": episodes,
        }
    )
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")

    artifacts = _seal_input_episodes(path)

    assert set(artifacts) == {str(seed) for seed in range(16, 24)}
    assert artifacts["16"]["policy"] == ledger["policy"]


def test_unsealed_terra_ledger_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unsealed-ledger.json"
    path.write_text('{"schema_version": "terra-decision-ledger-v0.1.0"}', encoding="utf-8")

    with pytest.raises(ValueError, match="TERRA_LEDGER_INTEGRITY_FAILURE"):
        _seal_input_episodes(path)


def test_terra_ledger_requires_request_contract_and_orchestration_task_id(
    tmp_path: Path,
) -> None:
    """A resealed legacy ledger cannot masquerade as an attested Terra run."""
    env = FounderDecisionEnvironment()
    observation = env.reset(seed=16)
    decision_value = decision("abstain").model_dump(mode="json")
    turns: list[dict[str, object]] = []
    for _ in range(4):
        turns.append(
            {
                "input_observation_sha256": payload_sha256(
                    observation.model_dump(mode="json")
                ),
                "decision": decision_value,
                "raw_response": json.dumps(decision_value),
                "repair_response": None,
                "parse_attempts": 1,
            }
        )
        observation = env.step(
            build_trusted_action(observation, AgentDecision.model_validate(decision_value))
        )
    ledger = seal_artifact(
        {
            "schema_version": "terra-decision-ledger-v0.1.0",
            "policy": {
                "provider": "codex-subagent",
                "model": "gpt-5.6-terra",
                "mode": "inference",
                "model_weight_updates": False,
                "prompt_version": "observation-only-v0.1.0",
            },
            "episodes": {"16": {"turns": turns}},
        }
    )
    path = tmp_path / "legacy-resealed-ledger.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(ValueError, match="TERRA_REQUEST_CONTRACT_REQUIRED"):
        _seal_input_episodes(path)
