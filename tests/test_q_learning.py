import json
import math
from pathlib import Path

from yc_founder_decision_env.agent_trial import build_trusted_action, verify_artifact
from yc_founder_decision_env.models import AgentDecision
from yc_founder_decision_env.q_learning import (
    QTrainingConfig,
    available_actions,
    encode_state,
    q_update,
    train_q_learning,
)
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
    assert synthetic_utility(before, after) == 0.6625
    assert synthetic_utility(before, after) != after.reward


def test_rejected_action_receives_fixed_synthetic_penalty() -> None:
    before, after = take(0, "fundraise")  # seed 0 disallows fundraise
    assert "ACTION_NOT_ALLOWED" in after.failure_codes
    assert synthetic_utility(before, after) == -1.0


def test_utility_uses_public_state_deltas_only() -> None:
    before, after = take(0, "abstain")
    assert synthetic_utility(before, after) == -0.002083


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
    trace = q_update(
        q,
        visits,
        state,
        "sell_pilot",
        0.5,
        next_state,
        False,
        next_actions=["sell_pilot"],
    )
    assert trace["visit"] == 1
    assert trace["alpha"] == 1.0
    assert trace["target"] == 2.4  # 0.5 + 0.95 * 2.0
    assert q[(state, "sell_pilot")] == 2.4
    second = q_update(
        q,
        visits,
        state,
        "sell_pilot",
        0.5,
        next_state,
        True,
        next_actions=["sell_pilot"],
    )
    assert second["visit"] == 2
    assert math.isclose(second["alpha"], 1 / (2**0.6))
    assert second["target"] == 0.5


def test_q_update_ignores_infeasible_positive_next_value() -> None:
    q: dict[tuple[tuple[int, int, int, int], str], float] = {}
    visits: dict[tuple[tuple[int, int, int, int], str], int] = {}
    state = (4, 120, 64, 31)
    next_state = (3, 110, 50, 31)
    q[(next_state, "sell_pilot")] = -2.0
    q[(next_state, "fundraise")] = 10.0  # Infeasible continuation must not bootstrap.

    trace = q_update(
        q,
        visits,
        state,
        "sell_pilot",
        0.5,
        next_state,
        False,
        next_actions=["sell_pilot"],
    )

    assert trace["target"] == -1.4  # 0.5 + 0.95 * -2.0


def test_training_config_rejects_invalid_hyperparameters() -> None:
    for kwargs in (
        {"episodes": 0},
        {"gamma": float("nan")},
        {"gamma": 1.01},
        {"epsilon_start": float("inf")},
        {"epsilon_end": -0.01},
        {"alpha_exponent": 0.0},
    ):
        try:
            QTrainingConfig(rng_seed=1, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid config to fail: {kwargs}")


def test_training_is_deterministic_and_train_only() -> None:
    config = QTrainingConfig(rng_seed=20260723, episodes=20)
    first = train_q_learning(config)
    second = train_q_learning(config)
    assert first == second
    assert first["training_seeds"] == list(range(16))
    assert first["held_out_seeds_seen"] == []
    assert len(first["episode_returns"]) == 20


def test_committed_q_artifact_has_the_frozen_five_run_protocol() -> None:
    artifact_path = Path(__file__).parents[1] / "artifacts/rl/q-learning-v0.1.0.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert verify_artifact(artifact)
    assert [run["config"]["rng_seed"] for run in artifact["runs"]] == list(
        range(20260723, 20260728)
    )
    assert all(run["config"]["episodes"] == 1000 for run in artifact["runs"])
    assert all(run["training_seeds"] == list(range(16)) for run in artifact["runs"])
    assert all(run["held_out_seeds_seen"] == [] for run in artifact["runs"])
