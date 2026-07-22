"""Load versioned canonical records and sidecar environment metadata."""

import hashlib
import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

from .schema import CanonicalRecord


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_sha256(record: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(record).encode()).hexdigest()


@lru_cache(maxsize=1)
def load_bundle() -> tuple[list[CanonicalRecord], dict[str, dict[str, Any]]]:
    data_root = files("yc_founder_decision_env").joinpath("assets")
    records_path = data_root.joinpath("dataset-v0.1.0.jsonl")
    sidecar_path = data_root.joinpath("environment-metadata-v0.1.0.json")
    records: list[CanonicalRecord] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(CanonicalRecord.model_validate_json(line, strict=True))
    with sidecar_path.open("r", encoding="utf-8") as handle:
        sidecar_list = json.load(handle)
    sidecars = {entry["record_sha256"]: entry for entry in sidecar_list}
    if len(records) != len(sidecars):
        raise RuntimeError("SIDE_CAR_CARDINALITY_MISMATCH")
    for record in records:
        digest = record_sha256(record.model_dump())
        if digest not in sidecars:
            raise RuntimeError(f"SIDE_CAR_MISSING:{digest}")
    return records, sidecars
