import copy
import inspect
import json
import random
from pathlib import Path

import pytest

from yc_founder_decision_env.agent_trial import build_trusted_action, payload_sha256, seal_artifact
from yc_founder_decision_env.models import AgentDecision
from yc_founder_decision_env.rl_evaluation import (
    HELD_OUT_SEEDS,
    _aggregate,
    _load_terra_decision_ledger,
    black_box_rule_decision,
    build_held_out_report,
    evaluate_decisions,
    exhaustive_oracle,
    random_decision,
    verify_held_out_replay,
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
    report = build_held_out_report(
        _terra_ledger(), _frozen_q_artifact(), random_seeds=[20260723]
    )
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


def _decision(seed: int) -> dict[str, str]:
    observation = FounderDecisionEnvironment().reset(seed=seed)
    return {
        "action_type": "abstain",
        "rationale": "Public observation only.",
        "source_locator": observation.source_locators[0],
    }


def _terra_ledger(*, short_seed: int | None = None) -> dict[str, object]:
    episodes: dict[str, dict[str, object]] = {}
    for seed in HELD_OUT_SEEDS:
        env = FounderDecisionEnvironment()
        observation = env.reset(seed=seed)
        turns: list[dict[str, object]] = []
        for _ in range(4):
            decision = _decision(seed)
            turns.append(
                {
                    "input_observation_sha256": payload_sha256(observation.model_dump(mode="json")),
                    "decision": decision,
                    "raw_response": json.dumps(decision),
                    "repair_response": None,
                    "parse_attempts": 1,
                }
            )
            observation = env.step(
                build_trusted_action(
                    observation, AgentDecision.model_validate(turns[-1]["decision"])
                )
            )
        if short_seed == seed:
            turns.pop()
        episodes[str(seed)] = {"turns": turns}
    return seal_artifact(
        {
            "schema_version": "terra-decision-ledger-v0.1.0",
            "policy": {
                "provider": "codex-subagent",
                "model": "gpt-5.6-terra",
                "mode": "inference",
                "model_weight_updates": False,
                "prompt_version": "observation-only-v0.1.0",
            },
            "episodes": episodes,
        }
    )


def _frozen_q_artifact() -> dict[str, object]:
    runs = [
        {
            "config": {
                "rng_seed": rng_seed,
                "episodes": 1000,
                "gamma": 0.95,
                "epsilon_start": 0.30,
                "epsilon_end": 0.05,
                "alpha_exponent": 0.6,
            },
            "training_seeds": list(range(16)),
            "held_out_seeds_seen": [],
            "utility_version": "synthetic-utility-v0.1.0",
            "episode_returns": [0.0] * 1000,
            "curve_mean_every_50": [0.0] * 20,
            "q_table": [],
            "visit_counts": [],
            "representative_updates": [],
        }
        for rng_seed in range(20260723, 20260728)
    ]
    return seal_artifact(
        {
            "schema_version": "q-learning-artifact-v0.1.0",
            "training_protocol": {
                "rng_seeds": list(range(20260723, 20260728)),
                "episodes": 1000,
                "gamma": 0.95,
                "epsilon_schedule_version": "linear-floor-v0.1.0",
                "epsilon_start": 0.30,
                "epsilon_end": 0.05,
                "alpha_rule_version": "visit-count-power-v0.1.0",
                "alpha_exponent": 0.6,
                "training_seeds": list(range(16)),
                "held_out_seeds": [],
                "utility_version": "synthetic-utility-v0.1.0",
            },
            "runs": runs,
        }
    )


def test_terra_ledger_rejects_incomplete_nonterminal_episode() -> None:
    with pytest.raises(ValueError, match="TERRA_INCOMPLETE_NONTERMINAL_EPISODE"):
        build_held_out_report(_terra_ledger(short_seed=16), _frozen_q_artifact(), [20260723])


def test_report_preserves_verified_terra_ledger_provenance_and_turn_hashes() -> None:
    ledger = _terra_ledger()
    report = build_held_out_report(ledger, _frozen_q_artifact(), [20260723])

    assert report["terra_ledger"] == ledger


def test_loader_rejects_legacy_flat_terra_map(tmp_path: Path) -> None:
    legacy = {str(seed): [_decision(seed)] * 4 for seed in HELD_OUT_SEEDS}
    path = tmp_path / "legacy-terra.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(ValueError, match="INVALID_TERRA_LEDGER_SCHEMA"):
        _load_terra_decision_ledger(path)


def test_terra_turn_response_must_equal_stored_decision() -> None:
    ledger = _terra_ledger()
    turn = ledger["episodes"]["16"]["turns"][0]  # type: ignore[index]
    response_decision = copy.deepcopy(turn["decision"])
    response_decision["rationale"] = "Different model response."
    turn["repair_response"] = json.dumps(response_decision)
    turn["parse_attempts"] = 2
    ledger = seal_artifact(ledger)

    with pytest.raises(ValueError, match="TERRA_RESPONSE_DECISION_MISMATCH"):
        build_held_out_report(ledger, _frozen_q_artifact(), [20260723])


def test_q_artifact_requires_integrity_and_frozen_protocol() -> None:
    artifact = _frozen_q_artifact()
    artifact["runs"][0]["config"]["episodes"] = 999  # type: ignore[index]
    artifact = seal_artifact(artifact)
    with pytest.raises(ValueError, match="INVALID_Q_FROZEN_CONFIG"):
        build_held_out_report(_terra_ledger(), artifact, [20260723])


def test_report_rejects_empty_q_runs() -> None:
    with pytest.raises(ValueError, match="Q_RUNS_REQUIRED"):
        build_held_out_report(_terra_ledger(), {"runs": []}, [20260723])


def test_replay_recomputes_episode_digests_and_aggregates() -> None:
    report = build_held_out_report(_terra_ledger(), _frozen_q_artifact(), [20260723])
    tampered = report["policies"]["terra"]["episodes"][0]
    tampered["total_synthetic_utility"] = 999.0
    tampered["trajectory_sha256"] = payload_sha256(
        {key: value for key, value in tampered.items() if key != "trajectory_sha256"}
    )
    report = seal_artifact(report)

    assert verify_held_out_replay(report) is False


@pytest.fixture(scope="module")
def frozen_report() -> dict[str, object]:
    return build_held_out_report(_terra_ledger(), _frozen_q_artifact())


def test_replay_rejects_wrong_report_schema(frozen_report: dict[str, object]) -> None:
    report = copy.deepcopy(frozen_report)
    report["schema_version"] = "held-out-comparison-v9.9.9"

    assert verify_held_out_replay(seal_artifact(report)) is False


def test_replay_rejects_missing_policy_family(frozen_report: dict[str, object]) -> None:
    report = copy.deepcopy(frozen_report)
    del report["policies"]["terra"]  # type: ignore[index]

    assert verify_held_out_replay(seal_artifact(report)) is False


def test_replay_rejects_self_consistent_wrong_episode_count(
    frozen_report: dict[str, object],
) -> None:
    report = copy.deepcopy(frozen_report)
    random_aggregate = report["policies"]["random"]  # type: ignore[index]
    subaggregates = random_aggregate["per_rng_seed"]
    del subaggregates[next(iter(subaggregates))]
    episodes = [
        episode
        for subaggregate in subaggregates.values()
        for episode in subaggregate["episodes"]
    ]
    report["policies"]["random"] = {  # type: ignore[index]
        **_aggregate(episodes),
        "per_rng_seed": subaggregates,
    }

    assert verify_held_out_replay(seal_artifact(report)) is False


def test_replay_rejects_self_consistent_non_heldout_seed(
    frozen_report: dict[str, object],
) -> None:
    report = copy.deepcopy(frozen_report)
    rule_aggregate = report["policies"]["black_box_rule"]  # type: ignore[index]
    decisions = [
        AgentDecision.model_validate(step["decision"])
        for step in rule_aggregate["episodes"][0]["steps"]
    ]
    rule_aggregate["episodes"][0] = evaluate_decisions(15, decisions)
    report["policies"]["black_box_rule"] = _aggregate(rule_aggregate["episodes"])  # type: ignore[index]

    assert verify_held_out_replay(seal_artifact(report)) is False


def test_replay_requires_five_learned_q_subruns(frozen_report: dict[str, object]) -> None:
    report = copy.deepcopy(frozen_report)
    learned_aggregate = report["policies"]["learned_q"]  # type: ignore[index]
    subaggregates = learned_aggregate["per_training_seed"]
    del subaggregates[next(iter(subaggregates))]
    episodes = [
        episode
        for subaggregate in subaggregates.values()
        for episode in subaggregate["episodes"]
    ]
    report["policies"]["learned_q"] = {  # type: ignore[index]
        **_aggregate(episodes),
        "per_training_seed": subaggregates,
    }

    assert verify_held_out_replay(seal_artifact(report)) is False
