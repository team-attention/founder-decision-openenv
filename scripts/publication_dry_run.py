"""Offline-only validation of the GitHub, HF Dataset, and HF Space bundle."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "DATASET_CARD.md",
    "ENVIRONMENT_CARD.md",
    "LICENSE",
    "NOTICE",
    "RIGHTS_MANIFEST.json",
    "THREAT_MODEL.md",
    "Dockerfile",
    "openenv.yaml",
    "uv.lock",
    "publication/APPROVAL_REQUIRED.md",
    "src/yc_founder_decision_env/assets/dataset-v0.1.0.jsonl",
    "src/yc_founder_decision_env/assets/environment-metadata-v0.1.0.json",
]


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"missing publication files: {missing}")
    records = [
        json.loads(line) for line in (ROOT / REQUIRED[-2]).read_text(encoding="utf-8").splitlines()
    ]
    sidecars = json.loads((ROOT / REQUIRED[-1]).read_text(encoding="utf-8"))
    if len(records) != 24 or len(sidecars) != 24:
        raise SystemExit("dataset cardinality mismatch")
    if any(set(record) != {"messages", "ground_truth", "dataset"} for record in records):
        raise SystemExit("upstream schema drift")
    if any(not sidecar["synthetic"] or sidecar["source_content_included"] for sidecar in sidecars):
        raise SystemExit("rights boundary failure")
    manifest = json.loads((ROOT / "RIGHTS_MANIFEST.json").read_text())
    if (
        manifest["yc"]["policy"]
        != "link-only/no-scrape/no-original-text-transcript-audio-video-redistribution"
    ):
        raise SystemExit("YC rights policy drift")
    print(
        json.dumps(
            {
                "passed": True,
                "remote_mutation": False,
                "github_layout": True,
                "hf_dataset_layout": True,
                "hf_docker_space_layout": True,
                "records": len(records),
            }
        )
    )


if __name__ == "__main__":
    main()
