"""Deterministic hard verifier and separate optional strategic audit."""

from dataclasses import dataclass
from typing import Any

from .models import FounderAction, FounderState, RewardComponents


@dataclass(frozen=True)
class Verification:
    components: RewardComponents
    failure_codes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failure_codes


def verify_action(
    action: FounderAction, state: FounderState, sidecar: dict[str, Any]
) -> Verification:
    """Return machine-readable components and codes without mutating state."""

    failures: list[str] = []
    component = {
        "action_validity": 0.0,
        "budget_time_constraints": 0.0,
        "source_locator_validity": 0.0,
        "state_arithmetic": 0.0,
        "future_leakage": 0.0,
    }
    if action.action_type in sidecar["allowed_actions"]:
        component["action_validity"] = 0.2
    else:
        failures.append("ACTION_NOT_ALLOWED")

    cost = sidecar["action_costs"][action.action_type]
    exact_cost = (
        action.spend_cents == cost["budget_cents"] and action.founder_hours == cost["hours"]
    )
    affordable = (
        action.spend_cents <= state.budget_cents and action.founder_hours <= state.founder_hours
    )
    if exact_cost and affordable:
        component["budget_time_constraints"] = 0.2
    else:
        if not exact_cost:
            failures.append("COST_TAMPERING")
        if not affordable:
            failures.append("BUDGET_OR_TIME_OVERSPEND")

    valid_urls = {locator["url"] for locator in sidecar["source_locators"]}
    if action.source_locator in valid_urls:
        component["source_locator_validity"] = 0.2
    else:
        failures.append("FORGED_SOURCE_LOCATOR")

    expected_budget = state.budget_cents - action.spend_cents
    expected_hours = state.founder_hours - action.founder_hours
    if (
        action.claim.budget_after_cents == expected_budget
        and action.claim.founder_hours_after == expected_hours
    ):
        component["state_arithmetic"] = 0.2
    else:
        failures.append("STATE_ARITHMETIC_MISMATCH")

    rationale_lower = action.rationale.casefold()
    forbidden = [token.casefold() for token in sidecar["future_leakage_tokens"]]
    if action.claim.observed_step == state.step_count and not any(
        token in rationale_lower for token in forbidden
    ):
        component["future_leakage"] = 0.2
    else:
        failures.append("FUTURE_LEAKAGE")

    return Verification(RewardComponents(**component), tuple(sorted(set(failures))))


def strategic_quality_audit(action: FounderAction) -> dict[str, float | str]:
    """Optional non-RLVR heuristic placeholder, deliberately outside reward."""

    score = min(len(action.rationale.split()) / 80.0, 1.0)
    return {
        "audit_version": "heuristic-v0.1.0-not-a-reward",
        "strategic_quality": round(score, 3),
        "warning": "Not calibrated to real startup success and never scalarized into hard reward.",
    }
