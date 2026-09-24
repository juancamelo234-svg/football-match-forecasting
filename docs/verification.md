# Local release verification

Verified September 24, 2026 on Python 3.12.14 with NumPy 2.3.5, pandas 2.2.3 and SciPy 1.17.0.

- Editable package build and installation succeeded using preinstalled pinned dependencies and no build isolation.
- `python -m unittest discover -s tests -q`: 27 tests passed.
- `python -m football_forecasting.cli demo --output outputs/release-check`: 100 synthetic input matches, 80 warmup matches, 20 predictions.
- Synthetic RPS: 0.16545931754515572. This is a mechanics check, not a performance claim.
- Seed reproducibility, run-manifest reproducibility and output overwrite protection are checked by the test suite.
- `python scripts/verify_research.py`: all pooled RPS, log loss and accuracy values reproduce from 1,140 archived predictions within 1e-7 absolute tolerance. The saved prediction checksum matches the source archive.
- The current release was rechecked with `python -m football_forecasting.cli demo --output outputs/portfolio-verification`.

The original publication passed fresh dependency installation, all then-current tests, archived scoring and the demo on GitHub Actions for Python 3.11 and 3.12 ([run](https://github.com/juancamelo234-svg/football-match-forecasting/actions/runs/35945143764)). The audit adds seven regression checks and `scripts/verify_train_test.py`; see the commit’s Actions result for its hosted verification.

See [the audit](audit_train_test.md) for refitted real-data results, independent scoring checks, repaired input defects and coverage limits.
