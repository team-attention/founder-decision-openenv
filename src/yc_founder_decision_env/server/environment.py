"""Frozen four-step synthetic founder-decision transition environment."""

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from openenv.core.env_server.interfaces import Environment

from ..data import load_bundle, record_sha256
from ..models import (
    ActionName,
    FounderAction,
    FounderObservation,
    FounderState,
    PublicActionSpec,
    RewardComponents,
)
from ..verifier import verify_action

ZERO_REWARD = RewardComponents(
    action_validity=0.0,
    budget_time_constraints=0.0,
    source_locator_validity=0.0,
    state_arithmetic=0.0,
    future_leakage=0.0,
)

ACTION_ORDER: tuple[ActionName, ...] = (
    "interview_users",
    "build_feature",
    "sell_pilot",
    "fundraise",
    "change_price",
    "abstain",
)


class FounderDecisionEnvironment(Environment):
    """OpenEnv-compatible deterministic environment with isolated instance state."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._sidecar: dict[str, Any] | None = None
        self._state = FounderState(
            episode_id=None,
            step_count=0,
            case_id="uninitialized",
            seed=0,
            week=0,
            budget_cents=0,
            founder_hours=0,
            active_users=0,
            interviews_completed=0,
            qualified_pipeline=0,
            mrr_cents=0,
            price_cents=0,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> FounderObservation:
        del kwargs
        selected_seed = 0 if seed is None else seed
        records, sidecars = load_bundle()
        record = records[selected_seed % len(records)]
        digest = record_sha256(record.model_dump())
        self._sidecar = sidecars[digest]
        initial = self._sidecar["initial_state"]
        deterministic_id = str(uuid5(NAMESPACE_URL, f"ycfd:{selected_seed}:{digest}"))
        self._state = FounderState(
            episode_id=episode_id or deterministic_id,
            step_count=0,
            case_id=self._sidecar["case_id"],
            seed=selected_seed,
            week=0,
            terminated=False,
            truncated=False,
            **initial,
        )
        return self._observation(ZERO_REWARD, [])

    def step(
        self,
        action: FounderAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> FounderObservation:
        del timeout_s, kwargs
        if self._sidecar is None:
            raise RuntimeError("RESET_REQUIRED")
        if self._state.terminated or self._state.truncated:
            raise RuntimeError("EPISODE_ALREADY_DONE")
        verification = verify_action(action, self._state, self._sidecar)
        if verification.passed:
            self._state.budget_cents -= action.spend_cents
            self._state.founder_hours -= action.founder_hours
            transition = self._sidecar["transitions"][action.action_type]
            for field, delta in transition.items():
                setattr(self._state, field, max(0, getattr(self._state, field) + delta))
        self._state.step_count += 1
        self._state.week = self._state.step_count
        self._state.terminated = self._state.budget_cents == 0 or self._state.founder_hours == 0
        self._state.truncated = self._state.step_count >= 4 and not self._state.terminated
        return self._observation(verification.components, list(verification.failure_codes))

    def _observation(
        self, components: RewardComponents, failure_codes: list[str]
    ) -> FounderObservation:
        assert self._sidecar is not None
        done = self._state.terminated or self._state.truncated
        return FounderObservation(
            case_id=self._state.case_id,
            week=self._state.week,
            budget_cents=self._state.budget_cents,
            founder_hours=self._state.founder_hours,
            active_users=self._state.active_users,
            interviews_completed=self._state.interviews_completed,
            qualified_pipeline=self._state.qualified_pipeline,
            mrr_cents=self._state.mrr_cents,
            price_cents=self._state.price_cents,
            allowed_actions=self._sidecar["allowed_actions"],
            action_specs=self._public_action_specs(),
            source_locators=[item["url"] for item in self._sidecar["source_locators"]],
            reward_components=components,
            failure_codes=failure_codes,
            terminated=self._state.terminated,
            truncated=self._state.truncated,
            done=done,
            reward=components.total,
            metadata={
                "transition_model_version": self._state.transition_model_version,
                "hard_reward_only": True,
            },
        )

    def _public_action_specs(self) -> list[PublicActionSpec]:
        assert self._sidecar is not None
        allowed = set(self._sidecar["allowed_actions"])
        return [
            PublicActionSpec(
                action_type=name,
                allowed=name in allowed,
                spend_cents=self._sidecar["action_costs"][name]["budget_cents"],
                founder_hours=self._sidecar["action_costs"][name]["hours"],
            )
            for name in ACTION_ORDER
        ]

    @property
    def state(self) -> FounderState:
        return self._state.model_copy(deep=True)
