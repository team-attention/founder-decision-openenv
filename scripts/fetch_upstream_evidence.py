"""Fetch pinned upstream raw files and snapshot the first decoded records."""

import hashlib
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "upstream"

CANDIDATES = [
    {
        "id": "allenai-RLVR-GSM",
        "official_url": "https://huggingface.co/datasets/allenai/RLVR-GSM",
        "revision": "b14884519de816306cf8ce7fc74547284b9ac548",
        "license": "MIT",
        "license_url": "https://huggingface.co/datasets/allenai/RLVR-GSM/blob/b14884519de816306cf8ce7fc74547284b9ac548/README.md",
        "raw_url": "https://huggingface.co/datasets/allenai/RLVR-GSM/resolve/b14884519de816306cf8ce7fc74547284b9ac548/data/train-00000-of-00001.parquet",
        "kind": "parquet",
        "selected": True,
        "decision": (
            "selected: smallest public RLVR record contract; exact schema can stay frozen "
            "while environment data lives in a keyed sidecar"
        ),
    },
    {
        "id": "allenai-RLVR-MATH",
        "official_url": "https://huggingface.co/datasets/allenai/RLVR-MATH",
        "revision": "bd2a93551b503a395fadd1a740d957559cfe6f3c",
        "license": "MIT",
        "license_url": "https://github.com/hendrycks/math/blob/0480a17a8246c8e1c5503a403620921283ee23a9/LICENSE",
        "raw_url": "https://huggingface.co/datasets/allenai/RLVR-MATH/resolve/bd2a93551b503a395fadd1a740d957559cfe6f3c/data/train-00000-of-00001.parquet",
        "kind": "parquet",
        "selected": False,
        "decision": (
            "rejected: adds nullable constraint fields unrelated to the founder-decision task"
        ),
    },
    {
        "id": "tau2-bench-retail",
        "official_url": "https://github.com/sierra-research/tau2-bench",
        "revision": "f0927a4b9cdfa7b374269efaa29ff7fdac90d8fc",
        "license": "MIT",
        "license_url": "https://github.com/sierra-research/tau2-bench/blob/f0927a4b9cdfa7b374269efaa29ff7fdac90d8fc/LICENSE",
        "raw_url": "https://raw.githubusercontent.com/sierra-research/tau2-bench/f0927a4b9cdfa7b374269efaa29ff7fdac90d8fc/data/tau2/domains/retail/tasks.json",
        "kind": "json-array",
        "selected": False,
        "decision": (
            "rejected as record schema: strong environment fit but domain-specific nested "
            "action gold would create policy-visible leakage pressure"
        ),
    },
    {
        "id": "webarena",
        "official_url": "https://github.com/web-arena-x/webarena",
        "revision": "dce04686a56253aefba7b18a4fa0937cf1dc987b",
        "license": "Apache-2.0",
        "license_url": "https://github.com/web-arena-x/webarena/blob/dce04686a56253aefba7b18a4fa0937cf1dc987b/LICENSE",
        "raw_url": "https://raw.githubusercontent.com/web-arena-x/webarena/dce04686a56253aefba7b18a4fa0937cf1dc987b/config_files/test.raw.json",
        "kind": "json-array",
        "selected": False,
        "decision": (
            "rejected: browser snapshot/auth/evaluator fields are not portable to this "
            "bounded synthetic task"
        ),
    },
]


def stable(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def main() -> None:
    raw_dir = OUT / "downloads"
    record_dir = OUT / "records"
    raw_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for candidate in CANDIDATES:
        suffix = ".parquet" if candidate["kind"] == "parquet" else ".json"
        raw_path = raw_dir / f"{candidate['id']}{suffix}"
        with urllib.request.urlopen(candidate["raw_url"], timeout=120) as response:
            payload = response.read()
        raw_path.write_bytes(payload)
        if candidate["kind"] == "parquet":
            table = pq.read_table(raw_path)
            record = table.slice(0, 1).to_pylist()[0]
            shape = {
                name: str(column.type)
                for name, column in zip(table.column_names, table.columns, strict=True)
            }
            null_counts = {
                name: column.null_count
                for name, column in zip(table.column_names, table.columns, strict=True)
            }
        else:
            rows = json.loads(payload)
            record = rows[0]
            shape = {key: type(value).__name__ for key, value in record.items()}
            null_counts = {key: int(value is None) for key, value in record.items()}
        record_path = record_dir / f"{candidate['id']}.record-0.json"
        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest.append(
            {
                **candidate,
                "raw_path": str(raw_path.relative_to(ROOT)),
                "raw_sha256": hashlib.sha256(payload).hexdigest(),
                "record_path": str(record_path.relative_to(ROOT)),
                "record_sha256": hashlib.sha256(stable(record)).hexdigest(),
                "record_shape": shape,
                "top_level_null_counts": null_counts,
            }
        )
    (OUT / "candidate-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"candidates": len(manifest), "selected": "allenai-RLVR-GSM"}))


if __name__ == "__main__":
    main()
