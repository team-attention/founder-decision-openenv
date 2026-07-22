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
    assert synthetic_utility(before, after) == 0.6625
    assert synthetic_utility(before, after) != after.reward


def test_rejected_action_receives_fixed_synthetic_penalty() -> None:
    before, after = take(0, "fundraise")  # seed 0 disallows fundraise
    assert "ACTION_NOT_ALLOWED" in after.failure_codes
    assert synthetic_utility(before, after) == -1.0


def test_utility_uses_public_state_deltas_only() -> None:
    before, after = take(0, "abstain")
    assert synthetic_utility(before, after) == -0.002083
