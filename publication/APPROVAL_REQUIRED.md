# Publication authorization and commands

Command Center authorized public publication on 2026-07-22. The public names intentionally
avoid implying YC affiliation or endorsement:

- GitHub: `team-attention/founder-decision-openenv`
- Hugging Face Dataset: `team-attention/founder-decision-rlvr-v0`
- Hugging Face Docker Space: `team-attention/founder-decision-openenv`

The publication authorization covers:

1. Public organization/repository names and MIT licensing.
2. Publication of the 24 synthetic cases and source-locator sidecar.
3. Inclusion or exclusion of upstream evidence record snapshots.
4. YC link-only rights interpretation and final human legal review.
5. Hugging Face Dataset and Docker Space cards/limitations.
6. The explicit claim boundary: frozen synthetic transition/verifier only.

Exact commands from this repository root:

```bash
gh repo create team-attention/founder-decision-openenv --public --source=. --remote=origin --push

uv run hf repo create team-attention/founder-decision-rlvr-v0 --repo-type dataset
uv run hf upload team-attention/founder-decision-rlvr-v0 \
  src/yc_founder_decision_env/assets/dataset-v0.1.0.jsonl dataset-v0.1.0.jsonl \
  --repo-type dataset
uv run hf upload team-attention/founder-decision-rlvr-v0 \
  src/yc_founder_decision_env/assets/environment-metadata-v0.1.0.json \
  environment-metadata-v0.1.0.json --repo-type dataset
uv run hf upload team-attention/founder-decision-rlvr-v0 DATASET_CARD.md README.md --repo-type dataset

uv run hf repo create team-attention/founder-decision-openenv --repo-type space --space-sdk docker
uv run hf upload team-attention/founder-decision-openenv . . --repo-type space \
  --exclude '.git/*' '.venv/*' 'evidence/upstream/downloads/*'
```
