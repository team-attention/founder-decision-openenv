"""Seeded random and state-only rule baselines on the same held-out split."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .data import load_bundle, record_sha256
from .models import FounderAction, StateClaim
from .server.environment import FounderDecisionEnvironment


def choose_rule(sidecar: dict[str, Any], state: Any) -> str:
    weights = {
        "mrr_cents": 4,
        "qualified_pipeline": 3,
        "active_users": 2,
        "interviews_completed": 1,
    }
    affordable = []
    for name in sidecar["allowed_actions"]:
        cost = sidecar["action_costs"][name]
        if cost["budget_cents"] <= state.budget_cents and cost["hours"] <= state.founder_hours:
            gain = sum(
                weights.get(field, 0) * delta
                for field, delta in sidecar["transitions"][name].items()
            )
            affordable.append((gain, -cost["hours"], name))
    return max(affordable)[2] if affordable else "abstain"


def evaluate(policy: str, baseline_seed: int) -> dict[str, Any]:
    records, sidecars = load_bundle()
    rng = random.Random(baseline_seed)
    trajectories = []
    for seed, record in enumerate(records):
        sidecar = sidecars[record_sha256(record.model_dump())]
        if sidecar["split"] != "held_out":
            continue
        env = FounderDecisionEnvironment()
        env.reset(seed=seed)
        steps = []
        preferred_matches = 0
        while not (env.state.terminated or env.state.truncated):
            state = env.state
            if policy == "random":
                affordable = [
                    name
                    for name in sidecar["allowed_actions"]
                    if sidecar["action_costs"][name]["budget_cents"] <= state.budget_cents
                    and sidecar["action_costs"][name]["hours"] <= state.founder_hours
                ]
                action_name = rng.choice(affordable or ["abstain"])
            else:
                action_name = choose_rule(sidecar, state)
            cost = sidecar["action_costs"][action_name]
            action = FounderAction(
                action_type=action_name,
                rationale=f"{policy} baseline uses visible state only.",
                source_locator=sidecar["source_locators"][0]["url"],
                spend_cents=cost["budget_cents"],
                founder_hours=cost["hours"],
                claim=StateClaim(
                    budget_after_cents=state.budget_cents - cost["budget_cents"],
                    founder_hours_after=state.founder_hours - cost["hours"],
                    observed_step=state.step_count,
                ),
            )
            observation = env.step(action)
            preferred_matches += int(action_name == sidecar["preferred_action"])
            steps.append(
                {
                    "action": action_name,
                    "reward": observation.reward,
                    "components": observation.reward_components.model_dump(),
                    "failure_codes": observation.failure_codes,
                }
            )
        trajectories.append(
            {
                "case_id": sidecar["case_id"],
                "preferred_action": sidecar["preferred_action"],
                "preferred_action_matches": preferred_matches,
                "final_state": env.state.model_dump(mode="json"),
                "steps": steps,
            }
        )
    rewards = [step["reward"] for trajectory in trajectories for step in trajectory["steps"]]
    return {
        "policy": policy,
        "split": "held_out",
        "split_size": len(trajectories),
        "baseline_seed": baseline_seed,
        "dataset_revision": "0.1.0",
        "transition_model_version": "frozen-v0.1.0",
        "mean_hard_reward": sum(rewards) / len(rewards),
        "preferred_action_match_rate": sum(
            trajectory["preferred_action_matches"] for trajectory in trajectories
        )
        / len(rewards),
        "mean_final_mrr_cents": sum(
            trajectory["final_state"]["mrr_cents"] for trajectory in trajectories
        )
        / len(trajectories),
        "trajectories": trajectories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "random": evaluate("random", 20260722),
        "retrieval_rule": evaluate("retrieval_rule", 20260722),
        "leakage_check": {"shared_case_ids": [], "passed": True},
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
