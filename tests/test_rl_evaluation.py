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
    terra = {
        str(seed): [
            {
                "action_type": "abstain",
                "rationale": "Public observation only.",
                "source_locator": FounderDecisionEnvironment()
                .reset(seed=seed)
                .source_locators[0],
            }
            for _ in range(4)
        ]
        for seed in range(16, 24)
    }
    q_artifact = {"runs": []}
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
