"""Verify saved holdout evidence independently; no private training data needed."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_test import bootstrap


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'research/audit_20260924'
    hashes = json.loads((folder / 'checksums.json').read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    pred = pd.read_csv(folder / 'predictions.csv')
    metrics = pd.read_csv(folder / 'metrics.csv')
    splits = pd.read_csv(folder / 'splits.csv').set_index('season')
    if pred.duplicated(['season', 'model', 'match_id']).any():
        raise ValueError('Duplicate predictions')
    if len(metrics) != 12 or len(pred) != 3570:
        raise ValueError('Incorrect evaluation population')
    for (season, model), block in pred.groupby(['season', 'model']):
        row = metrics[(metrics.season == season) & (metrics.model == model)].iloc[0]
        split = splits.loc[season]
        dates = pd.to_datetime(block.date, utc=True)
        cutoff = pd.to_datetime(block.training_cutoff, utc=True)
        latest = pd.to_datetime(block.training_max_date, utc=True)
        if not (latest < cutoff).all() or not (cutoff <= dates.dt.normalize()).all():
            raise ValueError('Training timestamp leakage')
        if not ((dates >= pd.Timestamp(split.test_start)) &
                (dates < pd.Timestamp(split.test_end_exclusive))).all():
            raise ValueError('Test dates outside specified split')
        if len(block) != split.test_n:
            raise ValueError('Test count mismatch')
        if model == 'frozen_dc':
            if not cutoff.eq(pd.Timestamp(split.test_start)).all():
                raise ValueError('Frozen cutoff changed')
            if not block.training_matches.eq(split.train_n_at_start).all():
                raise ValueError('Frozen training sample changed')
        y = block.outcome.to_numpy(int)
        for prefix, columns in [('', ['p_home', 'p_draw', 'p_away']),
                                ('baseline_', ['base_home', 'base_draw', 'base_away'])]:
            p = block[columns].to_numpy()
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise ValueError('Invalid probabilities')
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-9, rtol=0)
            # Independent identity for three ordered outcomes, avoiding the model's scorer.
            rps = (((p[:, 0] - (y == 0)) ** 2 + (p[:, 2] - (y == 2)) ** 2) / 2).mean()
            loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()
            accuracy = (p.argmax(1) == y).mean()
            for key, value in [('rps', rps), ('log_loss', loss), ('accuracy', accuracy)]:
                np.testing.assert_allclose(value, row[prefix + key], atol=1e-12, rtol=0)
    # Validate the reported uncertainty using the saved seed and week-block protocol.
    for item in json.loads((folder / 'comparisons.json').read_text()):
        block = pred[pred.model.eq(item['model'])]
        block = block[block.season.lt(2026) if item['period'] == '2023-2025' else block.season.eq(2026)]
        recomputed = bootstrap(block, item['draws'], item['seed'])
        np.testing.assert_allclose(recomputed['ci95'], item['ci95'], atol=1e-12, rtol=0)
    print('Verified 12 evaluations, 3,570 forecast rows, temporal metadata, metrics and bootstrap intervals.')


if __name__ == '__main__':
    main()
