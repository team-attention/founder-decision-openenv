"""Synthetic benchmark utility, strictly separate from hard verifier reward."""

from .models import FounderObservation

SYNTHETIC_UTILITY_VERSION = "synthetic-utility-v0.1.0"


def synthetic_utility(before: FounderObservation, after: FounderObservation) -> float:
    """Score public state change; this is not evidence of real startup quality."""
    if after.reward != 1.0 or after.failure_codes:
        return -1.0
    value = (
        0.45 * ((after.mrr_cents - before.mrr_cents) / 15_000)
        + 0.25 * ((after.qualified_pipeline - before.qualified_pipeline) / 2)
        + 0.15 * ((after.active_users - before.active_users) / 3)
        + 0.10 * ((after.interviews_completed - before.interviews_completed) / 4)
        + 0.05 * ((after.price_cents - before.price_cents) / 500)
        - 0.05 * ((before.budget_cents - after.budget_cents) / 60_000)
        - 0.05 * ((before.founder_hours - after.founder_hours) / 24)
    )
    return round(value, 6)
