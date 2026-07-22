---
license: mit
task_categories:
- text-generation
language:
- en
pretty_name: Founder Decision RLVR v0.1.0
size_categories:
- n<1K
---

# Dataset card

This is a 24-record original synthetic dataset for a four-step founder-decision environment.
It is not a scrape, summary corpus, or redistribution of YC content.
It is an independent synthetic benchmark and is not affiliated with or endorsed by YC.
The executable environment and verifier are published at
https://github.com/team-attention/founder-decision-openenv.

## Schema and provenance

Records preserve, without additions, the three-field `allenai/RLVR-GSM` decoded record schema
at revision `b14884519de816306cf8ce7fc74547284b9ac548` (MIT). Environment-specific information is
in a separate versioned sidecar keyed by record SHA-256. Sixteen cases are train and eight are
held out. Top-level, list-element, and nested-member nullable semantics are preserved exactly.
The upstream schema fixture passes strict validation and semantic JSON round-trip.

## Source use

YC Startup School is used only through manually curated official URL, heading, and optional
timestamp locators. No YC article text, transcript, image, audio, or video is included.

## Intended use and limitations

Use for testing contract-following policies against transition model `frozen-v0.1.0` and
verifier `v0.1.0`. Do not use scores to predict startup success, evaluate founders, allocate
capital, or infer that one prescribed playbook is universally correct. Synthetic dynamics do
not establish real-world causality or strategic quality.
