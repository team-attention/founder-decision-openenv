"""One-command deterministic local demo and replay check."""

import json

from .data import load_bundle, record_sha256
from .models import FounderAction, StateClaim
from .server.environment import FounderDecisionEnvironment


def run(seed: int) -> list[dict[str, object]]:
    env = FounderDecisionEnvironment()
    observation = env.reset(seed=seed)
    records, sidecars = load_bundle()
    record = records[seed % len(records)]
    sidecar = sidecars[record_sha256(record.model_dump())]
    transcript: list[dict[str, object]] = [observation.model_dump(mode="json")]
    for action_name in [sidecar["preferred_action"], "interview_users", "sell_pilot", "abstain"]:
        if action_name not in sidecar["allowed_actions"]:
            action_name = sidecar["preferred_action"]
        cost = sidecar["action_costs"][action_name]
        state = env.state
        action = FounderAction(
            action_type=action_name,
            rationale="Use only the visible synthetic state and the official link locator.",
            source_locator=sidecar["source_locators"][0]["url"],
            spend_cents=cost["budget_cents"],
            founder_hours=cost["hours"],
            claim=StateClaim(
                budget_after_cents=state.budget_cents - cost["budget_cents"],
                founder_hours_after=state.founder_hours - cost["hours"],
                observed_step=state.step_count,
            ),
        )
        transcript.append(env.step(action).model_dump(mode="json"))
    transcript.append(env.state.model_dump(mode="json"))
    return transcript


def main() -> None:
    first = run(42)
    replay = run(42)
    if first != replay:
        raise SystemExit("deterministic replay failed")
    print(json.dumps({"replay_equal": True, "trajectory": first}, indent=2))


if __name__ == "__main__":
    main()
