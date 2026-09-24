# Repository audit and chronological train/test evaluation

**September 24, 2026 — Share with caveats.** The published statistical core was reviewed and refitted on real historical Premier League inputs. This review covers the public portfolio, not every experiment in the private research archive. XGBoost training and Colombian operational validation remain outside this release.

## What was run

Settings were held at the existing CLI defaults: ratings/conversion half-life **240 days**, L2 strength **2**, Dixon–Coles **rho = -0.04**, minimum history **80**. No grid search, parameter tuning or automatic model promotion occurred. Each season below is a separate temporal split; later splits may train on earlier test seasons. These are intentionally sequential evaluations, not mutually untouched datasets.

- **Frozen holdout:** fit ratings, goal conversion and the frequency baseline once using only matches before the first test day. Keep all fixed throughout that season.
- **Walk-forward:** refit before each test day using only earlier UTC days. Earlier test-period matches enter subsequent training, but the target day never does.
- **Poisson control:** reuse the walk-forward goal intensities with rho = 0, isolating the low-score adjustment.
- **Frequency baseline:** historical home/draw/away proportions using the same eligible training history as its model. This is a simple baseline, not the betting market.

Training begins in August 2014. The table gives initial training sizes; walk-forward training grows during the test. Test dates and exact counts are in [splits.csv](../research/audit_20260924/splits.csv).

| Test season | Initial training matches | Test matches | Frozen RPS ↓ | Walk-forward RPS ↓ | Frequency RPS ↓ | Walk-forward log loss ↓ | Walk-forward accuracy ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023/24 | 3,420 | 380 | 0.200314 | 0.190959 | 0.233828 | 0.933479 | 59.21% |
| 2024/25 | 3,800 | 380 | 0.210758 | 0.197084 | 0.235376 | 0.970564 | 52.89% |
| 2025/26 | 4,180 | 380 | 0.211094 | 0.207265 | 0.227694 | 1.022526 | 48.16% |
| 2026/27 | 4,560 | 50 | 0.198896 | 0.202338 | 0.227577 | 1.041870 | 46.00% |

The 2026/27 sample covers August 21–September 20, 2026: 50 available matches, not a completed season. Every one of the 1,190 target matches received a forecast under each of three variants (3,570 output rows).

## Interpretation

Across the three complete test seasons, walk-forward Dixon–Coles achieved **RPS 0.198436**, **log loss 0.975523**, and **53.42% accuracy**. Frozen-season ratings produced **RPS 0.207389** and **50.88% accuracy**. The current 50-match sample reverses that ordering: frozen RPS **0.198896** versus walk-forward **0.202338**. No model selection should be made from that small reversal.

The refitted walk-forward probabilities match all 1,140 archived statistical-reference predictions with maximum absolute discrepancy **3.03e-9** after joining on provider match ID. Thus statistical-reference fitting was reproduced, not just its saved scores. This does not reproduce XGBoost training.

The three-season RPS difference versus historical frequencies was **-0.033863**, with a descriptive paired ISO-week bootstrap 95% interval **[-0.040963, -0.026783]** (2,000 resamples; 109 week blocks). This supports a retrospective advantage over that simple baseline within this sample. It is not a market comparison, and the interval does not account for earlier research selection. The current-season intervals use only five week blocks and are too fragile for a general advantage claim.

Dixon–Coles improves pooled RPS only slightly relative to the rho = 0 control (**0.198436 vs. 0.198497**). No independent superiority claim for that adjustment is justified here. Draw probability averages **24.96%** in the 50-match sample versus **32%** observed draws; monitor calibration rather than treating 50 observations as proof of systematic error. Five-bin calibration tables are included as descriptive diagnostics.

## Verified repairs and remaining limits

1. **Duplicate IDs through the Python API — fixed.** Previously, different fixtures sharing a match ID passed core validation even though the CLI rejected them. The core now rejects null/duplicate IDs; a regression test covers the alternate-fixture case. The reviewed dataset contained no such duplicates, so historical results were unaffected.
2. **Numeric-string scores — fixed.** Numeric strings were checked as numbers but retained as strings for later calculations. Scores are now converted to integers after finite/nonnegative/integer validation. A regression test confirms equivalence with numeric input. The source dataset already had integer scores.
3. **Holdout reproducibility — completed.** Added explicit test boundaries and frozen-training mode, per-forecast training cutoffs/counts/latest dates, log loss and accuracy outputs, and a saved protocol before execution. Tests alter all test targets and same-day observations to verify isolation.
4. **Verification documentation — corrected.** The initial notes still said hosted installation/CI had not run. The original publication passed both Python versions; the audit commit is checked by the same workflow plus the new saved-evidence verifier.
5. **Source availability and research bias — unresolved limitations.** Inputs are the retained archive, not newly independently downloaded provider data. All these seasons have prior research exposure. There is no untouched prospective result, no new market-odds comparison, and no XGBoost retraining in this audit.

## Data checks

The archived source contains **4,610 matches**: 12 complete 380-match seasons and 50 current-season matches. Checks found zero duplicate IDs, duplicate fixture keys, missing core values, negative npxG, noninteger scores, npxG values exceeding xG, or post-audit dates. Opponent npxG-against fields agree exactly. Source CSV/archive hashes and coverage counts are in [data_quality.json](../research/audit_20260924/data_quality.json).

**1,911 kickoff timestamps are marked approximate** in the source. Day-based information cutoffs avoid pretending their within-day order is reliable, but cannot prove historical provider availability. They can also affect fractional-day decay weights slightly. This audit validates internal consistency, not every source match against an independent provider.

## Reproduce

Verify saved evaluation outputs without private data:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/verify_research.py
python scripts/verify_train_test.py
```

Refit using a private canonical CSV satisfying the [data contract](data_dictionary.md), with the additional integer `season` column (season's starting year):

```bash
python scripts/train_test.py --input data/private/premier.csv --output outputs/my-audit
```

The input used here was derived from the archive's `data/processed/understat_matches_2014_2026.csv`: retain match_id, season, date, home, away, home_npxg, away_npxg, home_goals and away_goals; add competition_id = ENG_PL; prefix match_id/home/away with `ENG_PL::`; preserve row order and numeric values. No rows were excluded. Source and canonical input hashes are retained. Private provider training features are not committed.

For a single frozen holdout through the CLI:

```bash
football-forecast evaluate --input data/private/premier.csv --output outputs/holdout \
  --test-start 2025-08-15 --test-end 2026-05-25 --freeze-training
```

Omit `--freeze-training` for daily walk-forward evaluation over the same test window. Start is inclusive, end exclusive; boundaries must be UTC midnight dates. Both commands preserve existing output directories.

## Review coverage

Totals below describe scoped components, not a percentage certification. Zero observed defects does not certify unseen data or unsupported experiments. Code and saved outputs were inspected; GitHub's Markdown rendering was not visually audited.

| Category | Observed defects | Assessment |
| --- | --- | --- |
| Usefulness/completeness | 0 / 4 | Setup, definitions, runnable model and real-data evaluation covered; private inputs required for refitting. |
| Analytical clarity | 0 / 4 | Training protocols, benchmark, historical evidence and limitations distinguished. |
| Visual/interaction consistency | N/A | Repository/CLI, not a dashboard; no rendered layout audit. |

| Category | Observed defects | Assessment |
| --- | --- | --- |
| Source authority/confidence | 0 / 1 | Versioned source archive traced; external provider accuracy not independently re-fetched. |
| Value accuracy | 0 / 12 | All twelve season/model score rows independently checked using an alternative RPS identity. |
| Within-chart agreement | N/A | No charts in this audit. |
| Complete source details | 0 / 4 | Source quality record, fitting protocol, split record and output checksums retained. |
| Cross-artifact consistency | 0 / 2 | README/summary metrics and original/refitted reference forecasts reconciled. |
| Data-quality controls | 0 / 4 | Identity, numeric validity, league boundaries and temporal isolation checked; two defects repaired. |
| Conclusion support | 0 / 4 | Baseline comparison, temporal variation, DC ablation and reproducibility claims bounded to evidence. |
