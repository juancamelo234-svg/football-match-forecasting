# Local release verification

Verified September 24, 2026 on Python 3.12.14 with NumPy 2.3.5, pandas 2.2.3 and SciPy 1.17.0.

- Editable package build and installation succeeded using preinstalled pinned dependencies and no build isolation.
- `python -m unittest discover -s tests -q`: 20 tests passed.
- `python -m football_forecasting.cli demo --output outputs/release-check`: 100 synthetic input matches, 80 warmup matches, 20 predictions.
- Synthetic RPS: 0.16545931754515572. This is a mechanics check, not a performance claim.
- Seed reproducibility, run-manifest reproducibility and output overwrite protection are checked by the test suite.
- `python scripts/verify_research.py`: all pooled RPS, log loss and accuracy values reproduce from 1,140 archived predictions within 1e-7 absolute tolerance. The saved prediction checksum matches the source archive.
- The current release was rechecked with `python -m football_forecasting.cli demo --output outputs/portfolio-verification`.

A fresh network dependency installation and hosted GitHub Actions execution have not been verified here. The workflow is configured to run on Python 3.11 and 3.12 after publication.
