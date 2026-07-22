"""Strict OpenEnv action, observation, and state models."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field

ActionName = Literal[
    "interview_users",
    "build_feature",
    "sell_pilot",
    "fundraise",
    "change_price",
    "abstain",
]


class PublicActionSpec(BaseModel):
    """Agent-visible action contract; contains no outcome or preference data."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action_type: ActionName
    allowed: bool
    spend_cents: int = Field(ge=0)
    founder_hours: int = Field(ge=0)


class StateClaim(BaseModel):
    """Agent claim used to make arithmetic tampering machine-verifiable."""

    model_config = ConfigDict(extra="forbid", strict=True)

    budget_after_cents: int = Field(ge=0)
    founder_hours_after: int = Field(ge=0)
    observed_step: int = Field(ge=0, le=3)


class FounderAction(Action):
    """One constrained founder decision; unknown fields are rejected upstream."""

    action_type: ActionName
    rationale: str = Field(min_length=1, max_length=1200)
    source_locator: str = Field(min_length=8, max_length=500)
    spend_cents: int = Field(ge=0)
    founder_hours: int = Field(ge=0)
    claim: StateClaim


class RewardComponents(BaseModel):
    """Hard deterministic RLVR components; strategic quality is not included."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action_validity: float = Field(ge=0, le=0.2)
    budget_time_constraints: float = Field(ge=0, le=0.2)
    source_locator_validity: float = Field(ge=0, le=0.2)
    state_arithmetic: float = Field(ge=0, le=0.2)
    future_leakage: float = Field(ge=0, le=0.2)

    @property
    def total(self) -> float:
        return round(
            self.action_validity
            + self.budget_time_constraints
            + self.source_locator_validity
            + self.state_arithmetic
            + self.future_leakage,
            6,
        )


class FounderState(State):
    """Complete synthetic simulator state exposed by state()."""

    case_id: str
    seed: int = Field(ge=0)
    week: int = Field(ge=0, le=4)
    budget_cents: int = Field(ge=0)
    founder_hours: int = Field(ge=0)
    active_users: int = Field(ge=0)
    interviews_completed: int = Field(ge=0)
    qualified_pipeline: int = Field(ge=0)
    mrr_cents: int = Field(ge=0)
    price_cents: int = Field(ge=0)
    terminated: bool = False
    truncated: bool = False
    transition_model_version: str = "frozen-v0.1.0"


class FounderObservation(Observation):
    """Observation returned at reset and after every accepted/rejected action."""

    case_id: str
    week: int
    budget_cents: int
    founder_hours: int
    active_users: int
    interviews_completed: int
    qualified_pipeline: int
    mrr_cents: int
    price_cents: int
    allowed_actions: list[ActionName]
    action_specs: list[PublicActionSpec]
    source_locators: list[str]
    reward_components: RewardComponents
    failure_codes: list[str]
    terminated: bool
    truncated: bool
