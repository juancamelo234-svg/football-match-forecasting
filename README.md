# Football Match Forecasting

**Juan Carlos Camelo Betancourth · Independent quantitative research**

Can opponent-adjusted chance creation produce informative football outcome probabilities? This project implements interpretable team ratings, a score probability model, and chronological evaluation in Python.

The portfolio release contains the runnable statistical core of a larger Premier League and Colombia research project. It includes an offline synthetic demonstration, behavioral tests, a match-data contract, and documented historical research. Synthetic results demonstrate software behavior; they are not evidence of forecasting performance.

## Quick start

Requires Python 3.11 or 3.12. From this repository's directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
football-forecast demo
python scripts/verify_research.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. Installing dependencies requires internet; the demo and tests then run offline. Output goes to `outputs/demo/`: seeded synthetic matches, 20 chronological predictions after an 80-match warmup, evaluation metrics, and a run manifest recording input/code SHA-256 fingerprints, dependency versions, seed and settings. Choose a new `--output` directory to rerun; existing runs are protected from overwriting.

## Method

1. Validate match identities, dates, npxG, scores and competition boundaries.
2. Fit opponent-adjusted attack and defensive vulnerability using time decay and L2 regularization.
3. Estimate home and away npxG, then convert each to expected total goals using past venue-specific goal/npxG ratios.
4. Build a Poisson score matrix with an optional Dixon–Coles low-score correction.
5. Sum score probabilities into home/draw/away, totals and both-teams-to-score probabilities.
6. Replay in date order and score the probability distribution against observed results and a historical-frequency baseline.

See [methodology](docs/methodology.md), [data dictionary](docs/data_dictionary.md), and [reproducibility](docs/reproducibility.md).

## Historical research finding

An archived retrospective comparison evaluated XGBoost against the statistical reference over 1,140 Premier League matches (2023/24–2025/26).

| Metric | Statistical reference | XGBoost |
| --- | ---: | ---: |
| RPS, lower is better | **0.198436** | 0.199462 |
| Log loss, lower is better | **0.975523** | 0.983789 |
| Most likely outcome accuracy | 53.42% | **54.04%** |

The candidate improved pick accuracy slightly but did not improve probability scores; it was not promoted. These seasons had been examined in other research. The weekly bootstrap interval for the RPS difference includes zero. This is not an untouched final test or evidence of market outperformance.

The original [predictions](research/predictions.csv), [summary](research/summary.json), and [protocol](research/protocol.json) are included. Run `python scripts/verify_research.py` to recompute and verify all three pooled metrics against the published summary. The script checks the prediction file checksum and allows a 1e-7 absolute numerical tolerance for serialized probabilities. This reproduces **scoring of saved forecasts**, not XGBoost training. Retraining requires the original research features and datasets. See [scope and limitations](docs/reproducibility.md).

## Evaluate your own match data

Supply a CSV that meets the [data contract](docs/data_dictionary.md):

```bash
football-forecast evaluate --input data/private/matches.csv --output outputs/my-study
```

The command requires explicit league identity. Parameters can be changed with `--half-life`, `--shrink-k`, `--rho`, and `--min-history`. Evaluation defaults are 240 days, 2, −0.04, and 80 matches. Demo settings intentionally use the original synthetic reference (120 days, 8, 0). Choose settings before viewing test results.

## Repository map

| Location | Purpose |
| --- | --- |
| `src/football_forecasting/ratings.py` | Regularized team ratings and expected npxG |
| `src/football_forecasting/probabilities.py` | Score matrix, aggregation, scoring, replay |
| `src/football_forecasting/validation.py` | Competition boundaries |
| `src/football_forecasting/synthetic.py` | Seeded artificial league generator |
| `src/football_forecasting/cli.py` | Demo and CSV evaluation commands |
| `tests/` | Known-strength recovery, temporal isolation, probability and data checks |
| `docs/` | Definitions, method, reproduction and publishing instructions |
| `research/` | Archived forecasts, study evidence and source provenance |
| `scripts/verify_research.py` | Recompute the historical comparison metrics |

## Research scope

The larger archive contains 12 completed Premier League seasons plus subsequent observations and independent Colombian ingestion and exploratory studies. The public core can fit one explicitly labeled league at a time; it does not certify a supplied dataset's historical coverage. Colombia's operational validation remains incomplete. The provisional stability filter, lineup/shot-map explorations and XGBoost training pipeline are outside this compact release.

The project was developed with AI coding assistance. Its value rests on the research design, transparent assumptions, inspectable code and behavioral validation. Reviewers can run the software independently and inspect the documented limitations.
