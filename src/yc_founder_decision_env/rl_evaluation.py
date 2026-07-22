"""Leakage-safe held-out evaluation for public-observation policy families."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from .agent_trial import build_trusted_action, payload_sha256, seal_artifact, verify_artifact
from .models import ActionName, AgentDecision, FounderObservation
from .q_learning import Q_RNG_SEEDS, available_actions, greedy_q_decision
from .rewards import SYNTHETIC_UTILITY_VERSION, synthetic_utility
from .server.environment import ACTION_ORDER, FounderDecisionEnvironment

HELD_OUT_SEEDS = tuple(range(16, 24))
Policy = Callable[[FounderObservation], AgentDecision]


def _decision(observation: FounderObservation, action: ActionName, label: str) -> AgentDecision:
    return AgentDecision(
        action_type=action,
        rationale=f"{label}; public observation only.",
        source_locator=observation.source_locators[0],
    )


def random_decision(observation: FounderObservation, rng: random.Random) -> AgentDecision:
    """Choose uniformly from current publicly feasible actions."""
    feasible = available_actions(observation)
    if not feasible:
        raise RuntimeError("NO_FEASIBLE_PUBLIC_ACTION")
    return _decision(observation, rng.choice(feasible), "seeded-random")


def black_box_rule_decision(observation: FounderObservation) -> AgentDecision:
    """Apply the frozen, deployable rule using public observation fields only."""
    feasible = set(available_actions(observation))
    if observation.interviews_completed < 4 and "interview_users" in feasible:
        chosen: ActionName = "interview_users"
    elif observation.qualified_pipeline < 2 and "sell_pilot" in feasible:
        chosen = "sell_pilot"
    elif observation.mrr_cents < 15_000 and "sell_pilot" in feasible:
        chosen = "sell_pilot"
    elif observation.price_cents < 3_000 and "change_price" in feasible:
        chosen = "change_price"
    else:
        chosen = next(name for name in ACTION_ORDER if name in feasible)
    return _decision(observation, chosen, "frozen-black-box-rule-v0.1.0")


def evaluate_decisions(seed: int, decisions: list[AgentDecision]) -> dict[str, Any]:
    """Replay a finite public-decision sequence without learning or updates."""
    env = FounderDecisionEnvironment()
    observation = env.reset(seed=seed)
    steps: list[dict[str, Any]] = []
    total_utility = 0.0
    for decision in decisions:
        if observation.done:
            break
        before = observation
        observation = env.step(build_trusted_action(before, decision))
        utility = synthetic_utility(before, observation)
        total_utility += utility
        steps.append(
            {
                "decision": decision.model_dump(mode="json"),
                "observation_sha256": payload_sha256(observation.model_dump(mode="json")),
                "hard_reward": observation.reward,
                "hard_reward_components": observation.reward_components.model_dump(mode="json"),
                "synthetic_utility": utility,
                "failure_codes": observation.failure_codes,
            }
        )
    result: dict[str, Any] = {
        "seed": seed,
        "steps": steps,
        "total_hard_reward": round(sum(step["hard_reward"] for step in steps), 6),
        "total_synthetic_utility": round(total_utility, 6),
        "final_state": env.state.model_dump(mode="json"),
    }
    result["trajectory_sha256"] = payload_sha256(result)
    return result


def _decisions_for_policy(seed: int, policy: Policy) -> list[AgentDecision]:
    env = FounderDecisionEnvironment()
    observation = env.reset(seed=seed)
    decisions: list[AgentDecision] = []
    while not observation.done:
        decision = policy(observation)
        decisions.append(decision)
        observation = env.step(build_trusted_action(observation, decision))
    return decisions


def exhaustive_oracle(seed: int) -> dict[str, Any]:
    """Find the best public-action sequence with explicitly non-deployable query access."""
    best: tuple[float, tuple[int, ...], list[AgentDecision], dict[str, Any]] | None = None
    for indices in itertools.product(range(len(ACTION_ORDER)), repeat=4):
        env = FounderDecisionEnvironment()
        observation = env.reset(seed=seed)
        decisions: list[AgentDecision] = []
        valid_prefix = True
        for index in indices:
            action = ACTION_ORDER[index]
            if action not in available_actions(observation):
                valid_prefix = False
                break
            decision = _decision(observation, action, "exhaustive-black-box-oracle")
            decisions.append(decision)
            observation = env.step(build_trusted_action(observation, decision))
            if observation.done:
                break
        if not valid_prefix:
            continue
        result = evaluate_decisions(seed, decisions)
        candidate = (
            cast(float, result["total_synthetic_utility"]),
            tuple(-index for index in indices),
            decisions,
            result,
        )
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("ORACLE_FOUND_NO_TRAJECTORY")
    replay = evaluate_decisions(seed, best[2])
    return {
        "decisions": [decision.model_dump(mode="json") for decision in best[2]],
        "result": best[3],
        "replay_equal": replay["trajectory_sha256"] == best[3]["trajectory_sha256"],
    }


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("EPISODES_REQUIRED")
    steps = [step for episode in episodes for step in cast(list[dict[str, Any]], episode["steps"])]
    components = (
        "action_validity",
        "budget_time_constraints",
        "source_locator_validity",
        "state_arithmetic",
        "future_leakage",
    )
    failures: Counter[str] = Counter(
        code for step in steps for code in cast(list[str], step["failure_codes"])
    )
    return {
        "n_episodes": len(episodes),
        "mean_hard_reward": round(
            sum(cast(float, episode["total_hard_reward"]) for episode in episodes)
            / len(episodes),
            6,
        ),
        "mean_synthetic_utility": round(
            sum(cast(float, episode["total_synthetic_utility"]) for episode in episodes)
            / len(episodes),
            6,
        ),
        "mean_hard_reward_components": {
            component: round(
                sum(cast(float, step["hard_reward_components"][component]) for step in steps)
                / len(steps),
                6,
            )
            for component in components
        },
        "failure_code_counts": dict(sorted(failures.items())),
        "episodes": episodes,
        "trajectory_sha256s": [cast(str, episode["trajectory_sha256"]) for episode in episodes],
    }


def _terra_decisions(terra: dict[str, Any], seed: int) -> list[AgentDecision]:
    raw_episode = terra[str(seed)]
    if not isinstance(raw_episode, list):
        raise ValueError("TERRA_DECISIONS_MUST_BE_LISTS")
    if not raw_episode or len(raw_episode) > 4:
        raise ValueError("TERRA_DECISION_COUNT_INVALID")
    return [AgentDecision.model_validate(decision) for decision in raw_episode]


def _q_runs(q_artifact: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    raw_runs = q_artifact.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError("Q_RUNS_REQUIRED")
    if not raw_runs:
        return [("test-zero-q", []) for _ in Q_RNG_SEEDS]
    if len(raw_runs) != len(Q_RNG_SEEDS):
        raise ValueError("EXACTLY_FIVE_Q_RUNS_REQUIRED")
    parsed: list[tuple[str, list[dict[str, Any]]]] = []
    for run, expected_seed in zip(raw_runs, Q_RNG_SEEDS, strict=True):
        if not isinstance(run, dict):
            raise ValueError("INVALID_Q_RUN")
        config = run.get("config")
        table = run.get("q_table")
        if not isinstance(config, dict) or config.get("rng_seed") != expected_seed:
            raise ValueError("INVALID_Q_RUN_SEED")
        if not isinstance(table, list) or not all(isinstance(row, dict) for row in table):
            raise ValueError("INVALID_Q_TABLE")
        parsed.append((str(expected_seed), cast(list[dict[str, Any]], table)))
    return parsed


def _frozen_q_policy(table: list[dict[str, Any]]) -> Policy:
    def decide(observation: FounderObservation) -> AgentDecision:
        return greedy_q_decision(observation, table)

    return decide


def _random_policy(rng: random.Random) -> Policy:
    def decide(observation: FounderObservation) -> AgentDecision:
        return random_decision(observation, rng)

    return decide


def build_held_out_report(
    terra: dict[str, Any], q_artifact: dict[str, Any], random_seeds: list[int] | None = None
) -> dict[str, Any]:
    """Evaluate immutable Terra/Q inputs and three fixed baselines on seeds 16..23."""
    if set(terra) != {str(seed) for seed in HELD_OUT_SEEDS}:
        raise ValueError("TERRA_HELD_OUT_SPLIT_REQUIRED")
    rng_seeds = list(Q_RNG_SEEDS if random_seeds is None else random_seeds)
    if not rng_seeds:
        raise ValueError("RANDOM_SEEDS_REQUIRED")
    terra_by_seed = {seed: _terra_decisions(terra, seed) for seed in HELD_OUT_SEEDS}

    random_episodes: list[dict[str, Any]] = []
    random_subaggregates: dict[str, dict[str, Any]] = {}
    for rng_seed in rng_seeds:
        episodes = []
        for seed in HELD_OUT_SEEDS:
            rng = random.Random(rng_seed * 1000 + seed)
            decisions = _decisions_for_policy(seed, _random_policy(rng))
            episodes.append(evaluate_decisions(seed, decisions))
        random_episodes.extend(episodes)
        random_subaggregates[str(rng_seed)] = _aggregate(episodes)

    rule_episodes = [
        evaluate_decisions(seed, _decisions_for_policy(seed, black_box_rule_decision))
        for seed in HELD_OUT_SEEDS
    ]
    terra_episodes = [evaluate_decisions(seed, terra_by_seed[seed]) for seed in HELD_OUT_SEEDS]

    learned_q_episodes: list[dict[str, Any]] = []
    learned_q_subaggregates: dict[str, dict[str, Any]] = {}
    for run_seed, table in _q_runs(q_artifact):
        episodes = [
            evaluate_decisions(
                seed,
                _decisions_for_policy(
                    seed,
                    _frozen_q_policy(table),
                ),
            )
            for seed in HELD_OUT_SEEDS
        ]
        learned_q_episodes.extend(episodes)
        learned_q_subaggregates[run_seed] = _aggregate(episodes)

    oracle_episodes: list[dict[str, Any]] = []
    for seed in HELD_OUT_SEEDS:
        oracle = exhaustive_oracle(seed)
        oracle_result = cast(dict[str, Any], oracle["result"])
        oracle_result["oracle_decisions"] = oracle["decisions"]
        oracle_result["oracle_replay_equal"] = oracle["replay_equal"]
        oracle_episodes.append(oracle_result)

    policies = {
        "random": {**_aggregate(random_episodes), "per_rng_seed": random_subaggregates},
        "black_box_rule": _aggregate(rule_episodes),
        "terra": _aggregate(terra_episodes),
        "learned_q": {
            **_aggregate(learned_q_episodes),
            "per_training_seed": learned_q_subaggregates,
        },
        "exhaustive_oracle": {
            **_aggregate(oracle_episodes),
            "access_advantage": "fresh-environment exhaustive query benchmark; not deployable",
        },
    }
    report = seal_artifact(
        {
            "schema_version": "held-out-comparison-v0.1.0",
            "dataset_revision": "0.1.0",
            "transition_model_version": "frozen-v0.1.0",
            "verifier_version": "verifier-v0.1.0",
            "utility_version": SYNTHETIC_UTILITY_VERSION,
            "training_seeds": list(range(16)),
            "held_out_seeds": list(HELD_OUT_SEEDS),
            "combined_reward": None,
            "combined_reward_reason": (
                "hard reward and synthetic utility are intentionally not scalarized"
            ),
            "policies": policies,
        }
    )
    report["replay"] = {"passed": verify_held_out_replay(report)}
    return seal_artifact(report)


def verify_held_out_replay(report: dict[str, Any]) -> bool:
    """Replay frozen trajectories and rerun oracle search without updating any policy."""
    if not verify_artifact(report):
        return False
    policies = report.get("policies")
    if not isinstance(policies, dict):
        return False
    try:
        for policy_name, aggregate in policies.items():
            if not isinstance(aggregate, dict):
                return False
            episodes = aggregate.get("episodes")
            if not isinstance(episodes, list):
                return False
            for episode in episodes:
                if not isinstance(episode, dict) or not isinstance(episode.get("seed"), int):
                    return False
                seed = cast(int, episode["seed"])
                if policy_name == "exhaustive_oracle":
                    replayed = exhaustive_oracle(seed)["result"]
                    if replayed["trajectory_sha256"] != episode.get("trajectory_sha256"):
                        return False
                    continue
                steps = episode.get("steps")
                if not isinstance(steps, list):
                    return False
                decisions = [AgentDecision.model_validate(step["decision"]) for step in steps]
                replayed = evaluate_decisions(seed, decisions)
                if replayed["trajectory_sha256"] != episode.get("trajectory_sha256"):
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _load_terra_decision_ledger(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("TERRA_LEDGER_OBJECT_REQUIRED")
    episodes = raw.get("episodes", raw)
    if not isinstance(episodes, dict):
        raise ValueError("TERRA_EPISODES_REQUIRED")
    normalized: dict[str, Any] = {}
    for seed, episode in episodes.items():
        if isinstance(episode, list):
            normalized[seed] = episode
        elif isinstance(episode, dict) and isinstance(episode.get("turns"), list):
            normalized[seed] = [turn["decision"] for turn in episode["turns"]]
        else:
            raise ValueError("INVALID_TERRA_EPISODE")
    return normalized


def main() -> None:
    """Build a sealed held-out comparison from immutable Terra and Q artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--terra-decisions", required=True, type=Path)
    parser.add_argument("--q-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    terra = _load_terra_decision_ledger(args.terra_decisions)
    q_artifact = json.loads(args.q_artifact.read_text(encoding="utf-8"))
    if not isinstance(q_artifact, dict) or not q_artifact.get("runs"):
        raise ValueError("PRODUCTION_Q_RUNS_REQUIRED")
    report = build_held_out_report(terra, q_artifact)
    if not verify_held_out_replay(report):
        raise RuntimeError("HELD_OUT_REPLAY_FAILURE")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
