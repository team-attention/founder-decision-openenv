import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from yc_founder_decision_env.data import load_bundle, record_sha256
from yc_founder_decision_env.schema import (
    CanonicalRecord,
    canonical_round_trip,
    strict_json_loads,
)

ROOT = Path(__file__).resolve().parents[1]


def test_actual_upstream_record_validates_and_round_trips() -> None:
    path = ROOT / "evidence/upstream/records/allenai-RLVR-GSM.record-0.json"
    raw = json.loads(path.read_text())
    assert CanonicalRecord.model_validate(raw, strict=True).model_dump() == raw
    assert canonical_round_trip(raw) == raw


def test_schema_snapshot_matches_model_field_shape() -> None:
    snapshot = json.loads((ROOT / "schemas/rlvr-gsm-record.schema-snapshot.json").read_text())
    assert snapshot["required"] == ["messages", "ground_truth", "dataset"]
    assert set(CanonicalRecord.model_fields) == set(snapshot["properties"])
    assert set(CanonicalRecord.model_fields["messages"].annotation.__args__)  # list member is typed


def test_synthetic_records_have_exact_upstream_fields_and_sidecar_only_metadata() -> None:
    records, sidecars = load_bundle()
    assert len(records) == 24
    assert len(sidecars) == 24
    for record in records:
        dumped = record.model_dump()
        assert set(dumped) == {"messages", "ground_truth", "dataset"}
        assert "case_id" not in dumped
        assert sidecars[record_sha256(dumped)]["metadata_schema_version"] == "0.1.0"


@pytest.mark.parametrize(
    "payload",
    [
        '{"messages":[],"ground_truth":"x","dataset":"y","extra":1}',
        '{"messages":[{"content":"x","role":"user","extra":1}],"ground_truth":"x","dataset":"y"}',
    ],
)
def test_extra_schema_tampering_rejected(payload: str) -> None:
    with pytest.raises(ValidationError):
        CanonicalRecord.model_validate(strict_json_loads(payload), strict=True)


def test_upstream_nullable_semantics_are_preserved_exactly() -> None:
    record = CanonicalRecord.model_validate(
        {
            "messages": [None, {"content": None, "role": None}],
            "ground_truth": None,
            "dataset": None,
        },
        strict=True,
    )
    assert canonical_round_trip(record.model_dump()) == record.model_dump()
    assert CanonicalRecord.model_validate(
        {"messages": None, "ground_truth": None, "dataset": None}, strict=True
    )


def test_duplicate_and_malformed_json_rejected() -> None:
    with pytest.raises(ValueError, match="DUPLICATE_FIELD"):
        strict_json_loads('{"dataset":"a","dataset":"b"}')
    with pytest.raises(json.JSONDecodeError):
        strict_json_loads('{"dataset":')
