"""Deterministic tabular Q-learning over public founder observations only."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

from .agent_trial import build_trusted_action, seal_artifact
from .models import ActionName, AgentDecision, FounderObservation
from .rewards import SYNTHETIC_UTILITY_VERSION, synthetic_utility
from .server.environment import ACTION_ORDER, FounderDecisionEnvironment

EncodedState: TypeAlias = tuple[int, int, int, int]
QKey: TypeAlias = tuple[EncodedState, ActionName]
Q_RNG_SEEDS: tuple[int, ...] = (20260723, 20260724, 20260725, 20260726, 20260727)
TRAINING_SEEDS: tuple[int, ...] = tuple(range(16))
EPSILON_SCHEDULE_VERSION = "linear-floor-v0.1.0"
ALPHA_RULE_VERSION = "visit-count-power-v0.1.0"


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class QTrainingConfig:
    """Frozen tabular-learning hyperparameters; no model weights are updated."""

    rng_seed: int
    episodes: int = 1000
    gamma: float = 0.95
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    alpha_exponent: float = 0.6

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episodes, int)
            or isinstance(self.episodes, bool)
            or self.episodes <= 0
        ):
            raise ValueError("EPISODES_MUST_BE_POSITIVE")
        if not _is_finite_number(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("GAMMA_MUST_BE_FINITE_IN_UNIT_INTERVAL")
        if not _is_finite_number(self.epsilon_start) or not 0 <= self.epsilon_start <= 1:
            raise ValueError("EPSILON_START_MUST_BE_FINITE_IN_UNIT_INTERVAL")
        if not _is_finite_number(self.epsilon_end) or not 0 <= self.epsilon_end <= 1:
            raise ValueError("EPSILON_END_MUST_BE_FINITE_IN_UNIT_INTERVAL")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("EPSILON_END_MUST_NOT_EXCEED_START")
        if not _is_finite_number(self.alpha_exponent) or self.alpha_exponent <= 0:
            raise ValueError("ALPHA_EXPONENT_MUST_BE_FINITE_AND_POSITIVE")


def encode_state(observation: FounderObservation) -> EncodedState:
    """Encode only public feasibility features needed for the frozen benchmark."""
    allowed_mask = sum(
        1 << index for index, spec in enumerate(observation.action_specs) if spec.allowed
    )
    return (
        4 - observation.week,
        observation.budget_cents // 1000,
        observation.founder_hours,
        allowed_mask,
    )


def available_actions(observation: FounderObservation) -> list[ActionName]:
    """Return public actions which are allowed and currently affordable."""
    return [
        spec.action_type
        for spec in observation.action_specs
        if spec.allowed
        and spec.spend_cents <= observation.budget_cents
        and spec.founder_hours <= observation.founder_hours
    ]


def epsilon_at(config: QTrainingConfig, episode_index: int) -> float:
    """Return the frozen epsilon schedule for the configured episode horizon."""
    if not 0 <= episode_index < config.episodes:
        raise ValueError("EPISODE_INDEX_OUT_OF_RANGE")
    if config.episodes == 1:
        return config.epsilon_end
    progress = episode_index / (config.episodes - 1)
    return max(config.epsilon_end, config.epsilon_start * (1 - progress))


def q_update(
    q: dict[QKey, float],
    visits: dict[QKey, int],
    state: EncodedState,
    action: ActionName,
    reward: float,
    next_state: EncodedState,
    done: bool,
    *,
    gamma: float = 0.95,
    alpha_exponent: float = 0.6,
    next_actions: list[ActionName],
) -> dict[str, Any]:
    """Apply one visit-dependent tabular Q update and return an audit trace."""
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("GAMMA_MUST_BE_FINITE_IN_UNIT_INTERVAL")
    if not math.isfinite(alpha_exponent) or alpha_exponent <= 0:
        raise ValueError("ALPHA_EXPONENT_MUST_BE_FINITE_AND_POSITIVE")
    if not done and not next_actions:
        raise ValueError("NONTERMINAL_STATE_REQUIRES_FEASIBLE_ACTION")
    key = (state, action)
    visits[key] = visits.get(key, 0) + 1
    alpha = 1 / (visits[key] ** alpha_exponent)
    old = q.get(key, 0.0)
    next_values = [q.get((next_state, candidate), 0.0) for candidate in next_actions]
    target = reward if done else reward + gamma * max(next_values)
    new = old + alpha * (target - old)
    q[key] = new
    return {
        "state": list(state),
        "action": action,
        "reward": reward,
        "next_state": list(next_state),
        "done": done,
        "visit": visits[key],
        "alpha": alpha,
        "old_q": old,
        "target": target,
        "new_q": new,
    }


def greedy_action(q: dict[QKey, float], observation: FounderObservation) -> ActionName:
    """Choose the highest-valued feasible action with ACTION_ORDER tie-breaking."""
    state = encode_state(observation)
    feasible = set(available_actions(observation))
    ranked = [name for name in ACTION_ORDER if name in feasible]
    if not ranked:
        return "abstain"
    return max(ranked, key=lambda name: (q.get((state, name), 0.0), -ACTION_ORDER.index(name)))


def _decision(observation: FounderObservation, action: ActionName, label: str) -> AgentDecision:
    return AgentDecision(
        action_type=action,
        rationale=f"{label}; public observation only.",
        source_locator=observation.source_locators[0],
    )


def serialize_q_table(q: dict[QKey, float]) -> list[dict[str, Any]]:
    """Serialize a Q table in a stable, JSON-safe representation."""
    return [
        {"state": list(state), "action": action, "q": round(value, 12)}
        for (state, action), value in sorted(q.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def _deserialize_q_table(serialized_q_table: list[dict[str, Any]]) -> dict[QKey, float]:
    q: dict[QKey, float] = {}
    for row in serialized_q_table:
        raw_state = row.get("state")
        raw_action = row.get("action")
        raw_value = row.get("q")
        if (
            not isinstance(raw_state, list)
            or len(raw_state) != 4
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in raw_state)
            or raw_action not in ACTION_ORDER
            or not isinstance(raw_value, (int, float))
            or isinstance(raw_value, bool)
        ):
            raise ValueError("INVALID_Q_TABLE")
        state = cast(EncodedState, tuple(raw_state))
        action = cast(ActionName, raw_action)
        q[(state, action)] = float(raw_value)
    return q


def greedy_q_decision(
    observation: FounderObservation, serialized_q_table: list[dict[str, Any]]
) -> AgentDecision:
    """Make a frozen-Q decision without mutating or inspecting private state."""
    q = _deserialize_q_table(serialized_q_table)
    return _decision(observation, greedy_action(q, observation), "frozen-q")


def train_q_learning(config: QTrainingConfig) -> dict[str, Any]:
    """Train a local Q table exclusively on the fixed training split (seeds 0..15)."""
    rng = random.Random(config.rng_seed)
    q: dict[QKey, float] = {}
    visits: dict[QKey, int] = {}
    returns: list[float] = []
    trace: list[dict[str, Any]] = []
    for episode_index in range(config.episodes):
        seed = rng.choice(TRAINING_SEEDS)
        env = FounderDecisionEnvironment()
        observation = env.reset(seed=seed)
        episode_return = 0.0
        while not observation.done:
            state = encode_state(observation)
            feasible = available_actions(observation)
            if not feasible:
                raise RuntimeError("NO_FEASIBLE_PUBLIC_ACTION")
            epsilon = epsilon_at(config, episode_index)
            if rng.random() < epsilon:
                action = rng.choice(feasible)
                mode = "explore"
            else:
                action = greedy_action(q, observation)
                mode = "greedy"
            decision = _decision(observation, action, mode)
            after = env.step(build_trusted_action(observation, decision))
            utility = synthetic_utility(observation, after)
            update = q_update(
                q,
                visits,
                state,
                action,
                utility,
                encode_state(after),
                after.done,
                gamma=config.gamma,
                alpha_exponent=config.alpha_exponent,
                next_actions=available_actions(after),
            )
            if episode_index in {0, config.episodes - 1}:
                trace.append({"episode": episode_index, "seed": seed, **update})
            episode_return += utility
            observation = after
        returns.append(round(episode_return, 6))
    return {
        "config": asdict(config),
        "training_seeds": list(TRAINING_SEEDS),
        "held_out_seeds_seen": [],
        "utility_version": SYNTHETIC_UTILITY_VERSION,
        "episode_returns": returns,
        "curve_mean_every_50": [
            round(sum(returns[index : index + 50]) / len(returns[index : index + 50]), 6)
            for index in range(0, len(returns), 50)
        ],
        "q_table": serialize_q_table(q),
        "visit_counts": [
            {"state": list(state), "action": action, "count": count}
            for (state, action), count in sorted(visits.items(), key=lambda item: item[0])
        ],
        "representative_updates": trace,
    }


def _artifact(episodes: int) -> dict[str, Any]:
    return seal_artifact(
        {
            "schema_version": "q-learning-artifact-v0.1.0",
            "claims": {
                "model_weight_updates": False,
                "observation_only": True,
                "hard_reward_is_contract_compliance": True,
            },
            "training_protocol": {
                "rng_seeds": list(Q_RNG_SEEDS),
                "episodes": episodes,
                "gamma": 0.95,
                "epsilon_schedule_version": EPSILON_SCHEDULE_VERSION,
                "epsilon_start": 0.30,
                "epsilon_end": 0.05,
                "alpha_rule_version": ALPHA_RULE_VERSION,
                "alpha_exponent": 0.6,
                "training_seeds": list(TRAINING_SEEDS),
                "held_out_seeds": [],
                "utility_version": SYNTHETIC_UTILITY_VERSION,
            },
            "runs": [
                train_q_learning(QTrainingConfig(rng_seed=seed, episodes=episodes))
                for seed in Q_RNG_SEEDS
            ],
        }
    )


def main() -> None:
    """Train the five frozen Q-learning runs and write a sealed artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    artifact = _artifact(args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
