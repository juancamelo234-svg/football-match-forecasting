"""Fixed-parameter retrospective holdouts; requires a private canonical match CSV.

Usage: python scripts/train_test.py --input data/private/premier.csv --output outputs/audit
No tuning is performed. Saved forecasts can be audited without redistributing inputs.
"""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from football_forecasting.probabilities import (
    calibration_table, markets, probability_metrics, run, score_matrix,
)
from football_forecasting.ratings import validate_matches
from football_forecasting.validation import validate_competition

SETTINGS = dict(half_life=240.0, shrink_k=2.0, rho=-0.04, min_history=80)
SEASONS = [2023, 2024, 2025, 2026]


def bootstrap(frame, draws=2000, seed=20260924):
    """Paired ISO-week block bootstrap of model minus frequency-baseline RPS."""
    p = frame[['p_home', 'p_draw', 'p_away']].to_numpy()
    b = frame[['base_home', 'base_draw', 'base_away']].to_numpy()
    y = np.eye(3)[frame.outcome.to_numpy(int)]
    losses = lambda v: ((v.cumsum(1)[:, :2] - y.cumsum(1)[:, :2]) ** 2).mean(1)
    delta = losses(p) - losses(b)
    weeks = pd.to_datetime(frame.date, utc=True).dt.strftime('%G-%V')
    groups = [delta[np.flatnonzero(weeks.to_numpy() == key)] for key in sorted(weeks.unique())]
    rng = np.random.default_rng(seed)
    values = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]).mean()
              for _ in range(draws)]
    return {'mean_delta_rps': float(delta.mean()),
            'ci95': np.quantile(values, [.025, .975]).tolist(), 'weeks': len(groups),
            'draws': draws, 'seed': seed,
            'limitation': 'Retrospective descriptive interval; no adjustment for prior research selection.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory; previous runs are preserved.')
    data = pd.read_csv(args.input, dtype={'match_id': str})
    validate_competition(data, expected='ENG_PL', required=True)
    data = validate_matches(data)
    if 'season' not in data or data.season.isna().any():
        parser.error('season (integer starting year) is required.')
    for season in SEASONS:
        if not data.season.eq(season).any():
            parser.error(f'Missing test season {season}')
    args.output.mkdir(parents=True)
    source_dir = Path(__file__).resolve().parents[1] / 'src/football_forecasting'
    protocol = {
        'settings_fixed_before_this_run': SETTINGS, 'selection': 'No hyperparameter search or promotion',
        'test_seasons': SEASONS, 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_dir.glob('*.py'))},
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': platform.python_version(),
        'dependencies': {n: version(n) for n in ['numpy', 'pandas', 'scipy']},
        'protocols': {'frozen_dc': 'Ratings, conversion and baseline frozen before first test day',
                      'walk_forward_dc': 'Refit using only earlier UTC days, including earlier test days',
                      'walk_forward_poisson': 'Same walk-forward intensities, rho=0 control'},
        'interpretation': 'Retrospective diagnostics on previously examined seasons, not fresh blind tests',
    }
    # Persist the complete protocol before fitting or inspecting test results.
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    forecasts, records, splits = [], [], []
    for season in SEASONS:
        targets = data.loc[data.season.eq(season)]
        start, end = targets.date.min().normalize(), targets.date.max().normalize() + pd.Timedelta(days=1)
        selected = data.loc[data.date.lt(end)]
        expected_ids = set(targets.match_id)
        splits.append({'season': season, 'train_n_at_start': int(data.date.lt(start).sum()),
                       'train_start': data.loc[data.date.lt(start), 'date'].min().isoformat(),
                       'train_end': data.loc[data.date.lt(start), 'date'].max().isoformat(),
                       'test_start': start.isoformat(), 'test_end_exclusive': end.isoformat(),
                       'test_n': len(targets)})
        for frozen in [True, False]:
            name = 'frozen_dc' if frozen else 'walk_forward_dc'
            out = run(selected, **SETTINGS, test_start=start, test_end=end, freeze_training=frozen)
            if set(out.match_id) != expected_ids or len(out) != len(expected_ids):
                raise ValueError('Test population mismatch')
            out.attrs = {}
            out['season'], out['model'] = season, name
            forecasts.append(out)
            if not frozen:
                poisson = out.copy()
                probs = pd.DataFrame([markets(score_matrix(h, a, rho=0))
                                      for h, a in zip(out.lambda_home, out.lambda_away)])
                for col in probs:
                    poisson[col] = probs[col].to_numpy()
                poisson['model'] = 'walk_forward_poisson'
                forecasts.append(poisson)
            print(f'{season}: {name}, {len(out)} matches completed', flush=True)
    all_predictions = pd.concat(forecasts, ignore_index=True)
    for (season, model), frame in all_predictions.groupby(['season', 'model']):
        metrics = probability_metrics(frame[['p_home', 'p_draw', 'p_away']], frame.outcome)
        base = probability_metrics(frame[['base_home', 'base_draw', 'base_away']], frame.outcome)
        records.append({'season': int(season), 'model': model, **metrics,
                        **{'baseline_' + k: v for k, v in base.items() if k != 'n'},
                        'new_team_matches': int(frame.new_team.sum())})
    comparisons = []
    calibration = []
    for period, subset in [('2023-2025', all_predictions[all_predictions.season.lt(2026)]),
                           ('2026_available', all_predictions[all_predictions.season.eq(2026)])]:
        for model, frame in subset.groupby('model'):
            comparisons.append({'period': period, 'model': model,
                                **probability_metrics(frame[['p_home', 'p_draw', 'p_away']], frame.outcome),
                                **bootstrap(frame)})
            table = calibration_table(frame[['p_home', 'p_draw', 'p_away']], frame.outcome, bins=5)
            table['period'], table['model'] = period, model
            calibration.append(table)
    all_predictions.to_csv(args.output / 'predictions.csv', index=False)
    pd.DataFrame(records).to_csv(args.output / 'metrics.csv', index=False)
    pd.DataFrame(splits).to_csv(args.output / 'splits.csv', index=False)
    pd.concat(calibration).to_csv(args.output / 'calibration.csv', index=False)
    (args.output / 'comparisons.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    main()
