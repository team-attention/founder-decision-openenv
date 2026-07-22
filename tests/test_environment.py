import pytest
from pydantic import ValidationError

from yc_founder_decision_env.data import load_bundle, record_sha256
from yc_founder_decision_env.demo import run
from yc_founder_decision_env.models import FounderAction, StateClaim
from yc_founder_decision_env.server.environment import FounderDecisionEnvironment


def context(seed: int = 0):
    records, sidecars = load_bundle()
    record = records[seed % len(records)]
    return sidecars[record_sha256(record.model_dump())]


def valid_action(env: FounderDecisionEnvironment, name: str | None = None) -> FounderAction:
    sidecar = context(env.state.seed)
    selected = name or sidecar["preferred_action"]
    if selected not in sidecar["allowed_actions"]:
        selected = sidecar["preferred_action"]
    cost = sidecar["action_costs"][selected]
    state = env.state
    return FounderAction(
        action_type=selected,
        rationale="Visible synthetic state only.",
        source_locator=sidecar["source_locators"][0]["url"],
        spend_cents=cost["budget_cents"],
        founder_hours=cost["hours"],
        claim=StateClaim(
            budget_after_cents=state.budget_cents - cost["budget_cents"],
            founder_hours_after=state.founder_hours - cost["hours"],
            observed_step=state.step_count,
        ),
    )


def test_reset_step_state_four_step_and_deterministic_replay() -> None:
    assert run(42) == run(42)
    env = FounderDecisionEnvironment()
    obs = env.reset(seed=0)
    assert env.state.step_count == 0 and not obs.done
    for expected in range(1, 5):
        obs = env.step(valid_action(env, "abstain"))
        assert env.state.step_count == expected
    assert obs.done and obs.truncated and not obs.terminated


def test_different_action_changes_state() -> None:
    first = FounderDecisionEnvironment()
    second = FounderDecisionEnvironment()
    first.reset(seed=0)
    second.reset(seed=0)
    first.step(valid_action(first, "interview_users"))
    second.step(valid_action(second, "sell_pilot"))
    assert first.state.model_dump() != second.state.model_dump()


def test_disallowed_action_is_rejected_without_business_side_effect() -> None:
    env = FounderDecisionEnvironment()
    env.reset(seed=0)
    before = env.state
    sidecar = context(0)
    disallowed = next(
        name for name in sidecar["action_costs"] if name not in sidecar["allowed_actions"]
    )
    cost = sidecar["action_costs"][disallowed]
    action = FounderAction(
        action_type=disallowed,
        rationale="Visible synthetic state only.",
        source_locator=sidecar["source_locators"][0]["url"],
        spend_cents=cost["budget_cents"],
        founder_hours=cost["hours"],
        claim=StateClaim(
            budget_after_cents=before.budget_cents - cost["budget_cents"],
            founder_hours_after=before.founder_hours - cost["hours"],
            observed_step=before.step_count,
        ),
    )
    observation = env.step(action)
    assert "ACTION_NOT_ALLOWED" in observation.failure_codes
    assert env.state.budget_cents == before.budget_cents


def test_resource_exhaustion_terminates_before_horizon() -> None:
    env = FounderDecisionEnvironment()
    env.reset(seed=0)
    first = env.step(valid_action(env, "build_feature"))
    assert not first.done
    second = env.step(valid_action(env, "build_feature"))
    assert second.done and second.terminated and not second.truncated


def test_invalid_action_and_extra_field_rejected_by_type() -> None:
    with pytest.raises(ValidationError):
        FounderAction.model_validate(
            {
                "action_type": "hire_everyone",
                "rationale": "x",
                "source_locator": "https://example.com",
                "spend_cents": 0,
                "founder_hours": 0,
                "claim": {"budget_after_cents": 0, "founder_hours_after": 0, "observed_step": 0},
            }
        )
    env = FounderDecisionEnvironment()
    env.reset(seed=0)
    with pytest.raises(ValidationError):
        FounderAction.model_validate({**valid_action(env).model_dump(), "extra": 1})


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("overspend", "BUDGET_OR_TIME_OVERSPEND"),
        ("forged", "FORGED_SOURCE_LOCATOR"),
        ("arithmetic", "STATE_ARITHMETIC_MISMATCH"),
        ("future", "FUTURE_LEAKAGE"),
        ("cost", "COST_TAMPERING"),
    ],
)
def test_verifier_failures_are_machine_readable_and_side_effect_free(
    mutation: str, code: str
) -> None:
    env = FounderDecisionEnvironment()
    env.reset(seed=0)
    before = env.state
    action = valid_action(env)
    if mutation == "overspend":
        action.spend_cents = before.budget_cents + 1
    elif mutation == "forged":
        action.source_locator = "https://evil.invalid/forged"
    elif mutation == "arithmetic":
        action.claim.budget_after_cents += 1
    elif mutation == "future":
        action.rationale = "I used the ground_truth from a future_outcome."
    else:
        action.founder_hours += 1
    obs = env.step(action)
    assert code in obs.failure_codes
    assert obs.reward < 1.0
    after = env.state
    assert after.budget_cents == before.budget_cents
    assert after.active_users == before.active_users


def test_optional_strategic_audit_is_absent_from_hard_components() -> None:
    env = FounderDecisionEnvironment()
    env.reset(seed=0)
    obs = env.step(valid_action(env))
    assert "strategic_quality" not in obs.reward_components.model_dump()
