# Reproducibility and limitations

## What this repository reproduces

- Seeded synthetic data generation and chronological statistical-model forecasts.
- Tests of known strength recovery under an unequal schedule and noisy measurements.
- Invariance to changing same-day/future inputs, regularization behavior, and new-team fallback.
- Probability conservation, known analytical cases, Dixon–Coles constraints, RPS extremes and invalid data rejection.
- League isolation through explicit identifiers; command-line runs with protected output folders.
- A chronological replay on a compatible dataset supplied by the reviewer.

## Historical metric reproduction versus retraining

`research/predictions.csv`, `summary.json` and `protocol.json` come from the original XGBoost study. Run `python scripts/verify_research.py` from the repository root to reproduce pooled RPS, log loss and accuracy from all 1,140 archived forecasts. The script validates the source checksum and checks the metrics against the original summary (absolute tolerance 1e-7). It preserves the original scoring convention and allows float32 serialization rounding. It does not retrain the models, regenerate the predictions, or reproduce the bootstrap interval. The original data and feature snapshots are required for retraining. Previously examined seasons are labeled retrospective. No profit or sustained market advantage is established.

The synthetic demo uses seed 23. The suite also checks a distinct artificial sample for strength recovery. Numerical results may have minor cross-platform floating-point differences. Dependencies are pinned. Each CLI run saves `run_manifest.json` with Python/platform/dependency versions, model settings, the synthetic seed when applicable, and SHA-256 fingerprints of its input and source modules. For CSV evaluation, the input fingerprint covers the original file bytes. For the demo it covers the saved synthetic CSV bytes. Compare manifests before comparing runs.

## Source lineage

`research/source_manifest.json` records the source archive version and SHA-256 hashes of the original core modules. Modules were packaged with relative imports. Synthetic generation was moved out of the tests into a reusable module. CLI input checks and dedicated release tests were added. The underlying rating and score formulas were retained.

Some inherited internal comments and error strings remain in Spanish; public documentation and the CLI are in English. The full research archive contains many more studies than this curated software release. Neither passing CI nor running this core marks Colombia operationally validated.

## Data and code reuse

The repository contains artificial demonstration data generated at runtime and research summaries. It includes saved model probabilities, match identifiers, dates and observed outcome categories for metric auditing, but does not bundle provider training features or shot datasets. No general third-party data redistribution permission or open-source license is implied. The owner should choose a code license explicitly before inviting reuse; absence of a license does not prevent reviewers from inspecting the portfolio.
