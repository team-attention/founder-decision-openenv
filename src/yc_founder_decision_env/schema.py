"""Exact upstream RLVR-GSM record contract and strict JSON helpers."""

import json
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    content: str | None
    role: str | None


class CanonicalRecord(BaseModel):
    """Exact field names/nesting/types from allenai/RLVR-GSM."""

    model_config = ConfigDict(extra="forbid", strict=True)

    messages: list[ChatMessage | None] | None
    ground_truth: str | None
    dataset: str | None


def _reject_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"DUPLICATE_FIELD:{key}")
        value[key] = item
    return value


def strict_json_loads(payload: str) -> Any:
    """Decode JSON while rejecting duplicate object keys."""

    return json.loads(payload, object_pairs_hook=_reject_duplicates)


def canonical_round_trip(record: dict[str, Any]) -> dict[str, Any]:
    """Validate then decode/encode without adding or removing any field."""

    validated = CanonicalRecord.model_validate(record, strict=True)
    encoded = validated.model_dump_json(exclude_none=False)
    decoded = strict_json_loads(encoded)
    assert isinstance(decoded, dict)
    return decoded
