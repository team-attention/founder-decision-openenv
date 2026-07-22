"""Typed persistent OpenEnv WebSocket client."""

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import FounderAction, FounderObservation, FounderState


class FounderDecisionEnv(EnvClient[FounderAction, FounderObservation, FounderState]):
    def _step_payload(self, action: FounderAction) -> dict[str, Any]:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[FounderObservation]:
        raw_observation = payload.get("observation", {})
        observation = FounderObservation.model_validate(raw_observation)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: dict[str, Any]) -> FounderState:
        return FounderState.model_validate(payload)
