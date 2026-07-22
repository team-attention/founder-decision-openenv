"""Generate 24 original synthetic cases using the frozen upstream record schema."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "yc_founder_decision_env" / "assets"

ACTIONS = [
    "interview_users",
    "build_feature",
    "sell_pilot",
    "fundraise",
    "change_price",
    "abstain",
]
COSTS = {
    "interview_users": {"budget_cents": 5_000, "hours": 8},
    "build_feature": {"budget_cents": 60_000, "hours": 24},
    "sell_pilot": {"budget_cents": 10_000, "hours": 14},
    "fundraise": {"budget_cents": 20_000, "hours": 20},
    "change_price": {"budget_cents": 2_000, "hours": 4},
    "abstain": {"budget_cents": 0, "hours": 1},
}
TRANSITIONS = {
    "interview_users": {"interviews_completed": 4, "active_users": 1},
    "build_feature": {"active_users": 3},
    "sell_pilot": {"qualified_pipeline": 2, "mrr_cents": 15_000},
    "fundraise": {"qualified_pipeline": 1},
    "change_price": {"price_cents": 500, "mrr_cents": 5_000},
    "abstain": {},
}
LOCATORS = [
    {
        "locator_id": "yc-sus-2022-talk-users",
        "url": "https://www.ycombinator.com/blog/startup-school-videos",
        "heading": "How To Talk To Users with Gustaf Alstromer",
        "timestamp_seconds": None,
    },
    {
        "locator_id": "yc-sus-2022-first-customers",
        "url": "https://www.ycombinator.com/blog/startup-school-videos",
        "heading": "How to Get Your First Customers with Gustaf Alstromer",
        "timestamp_seconds": None,
    },
    {
        "locator_id": "yc-sus-2022-mvp",
        "url": "https://www.ycombinator.com/blog/startup-school-videos",
        "heading": "How to Build a Minimum Viable Product with Michael Seibel",
        "timestamp_seconds": None,
    },
    {
        "locator_id": "yc-sus-2022-pricing",
        "url": "https://www.ycombinator.com/blog/startup-school-videos",
        "heading": "Startup Business Models and Pricing with Aaron Epstein",
        "timestamp_seconds": None,
    },
    {
        "locator_id": "yc-sus-2022-fundraising",
        "url": "https://www.ycombinator.com/blog/startup-school-videos",
        "heading": "How Startup Fundraising Works with Brad Flora",
        "timestamp_seconds": None,
    },
    {
        "locator_id": "yc-sus-2022-launch",
        "url": "https://www.ycombinator.com/blog/startup-school-videos",
        "heading": "The Best Way To Launch Your Startup with Kat Manalac",
        "timestamp_seconds": None,
    },
    {
        "locator_id": "yc-sus-2019-ideas",
        "url": "https://www.ycombinator.com/blog/startup-school-week-1-recap-kevin-hale-and-eric-migicovsky",
        "heading": "How to Evaluate Startup Ideas",
        "timestamp_seconds": 43,
    },
    {
        "locator_id": "yc-sus-2019-hypothesis",
        "url": "https://www.ycombinator.com/blog/startup-school-week-1-recap-kevin-hale-and-eric-migicovsky",
        "heading": "A startup idea is a hypothesis",
        "timestamp_seconds": 170,
    },
    {
        "locator_id": "yc-sus-2019-users",
        "url": "https://www.ycombinator.com/blog/startup-school-week-1-recap-kevin-hale-and-eric-migicovsky",
        "heading": "Eric Migicovsky on How to Talk to Users",
        "timestamp_seconds": 919,
    },
    {
        "locator_id": "yc-sus-2018-products",
        "url": "https://www.ycombinator.com/blog/startup-school-2018-curriculum",
        "heading": "Building Products",
        "timestamp_seconds": None,
    },
    {
        "locator_id": "yc-sus-2018-sales",
        "url": "https://www.ycombinator.com/blog/startup-school-2018-curriculum",
        "heading": "Sales From 0 to 1 Million",
        "timestamp_seconds": None,
    },
    {
        "locator_id": "yc-sus-2018-fundraising",
        "url": "https://www.ycombinator.com/blog/startup-school-2018-curriculum",
        "heading": "Fundraising Fundamentals",
        "timestamp_seconds": None,
    },
]


def stable(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    records = []
    sidecars = []
    for index in range(24):
        case_id = f"ycfd-{index + 1:03d}"
        best_action = ACTIONS[index % len(ACTIONS)]
        initial = {
            "budget_cents": 120_000 + (index % 4) * 20_000,
            "founder_hours": 64 + (index % 3) * 8,
            "active_users": 4 + index,
            "interviews_completed": index % 5,
            "qualified_pipeline": index % 4,
            "mrr_cents": (index % 6) * 10_000,
            "price_cents": 2_000 + (index % 5) * 500,
        }
        disallowed = ACTIONS[(index + 3) % len(ACTIONS)]
        allowed = [name for name in ACTIONS if name != disallowed]
        if best_action not in allowed:
            allowed.append(best_action)
            allowed.sort(key=ACTIONS.index)
        prompt = (
            f"Synthetic founder case {case_id}. This scenario is authored by Team Attention, "
            "not copied from YC. Choose one action for the visible frozen state. "
            f"State={stable(initial)}. Allowed actions={stable(allowed)}. "
            "The claim is evaluation only under transition model frozen-v0.1.0; it does not "
            "predict real startup success."
        )
        truth = stable({"preferred_action_under_frozen_verifier": best_action})
        record = {
            "messages": [{"content": prompt, "role": "user"}],
            "ground_truth": truth,
            "dataset": "yc_founder_decision_frozen_v0.1.0",
        }
        digest = hashlib.sha256(stable(record).encode()).hexdigest()
        selected_locators = [LOCATORS[index % 12], LOCATORS[(index + 5) % 12]]
        sidecars.append(
            {
                "metadata_schema_version": "0.1.0",
                "record_sha256": digest,
                "case_id": case_id,
                "split": "train" if index < 16 else "held_out",
                "initial_state": initial,
                "allowed_actions": allowed,
                "action_costs": COSTS,
                "transitions": TRANSITIONS,
                "source_locators": selected_locators,
                "future_leakage_tokens": ["ground_truth", "future_outcome", "week 5"],
                "preferred_action": best_action,
                "synthetic": True,
                "source_content_included": False,
            }
        )
        records.append(record)
    (ASSETS / "dataset-v0.1.0.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (ASSETS / "environment-metadata-v0.1.0.json").write_text(
        json.dumps(sidecars, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"generated records={len(records)} sidecars={len(sidecars)}")


if __name__ == "__main__":
    main()
