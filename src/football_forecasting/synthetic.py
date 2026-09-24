"""Seeded artificial data for numerical checks; no real football evidence."""
import numpy as np
import pandas as pd

def synthetic(n=180, noise=False, seed=7):
    rng = np.random.default_rng(seed)
    teams = list('ABCDEF')
    a = np.exp(np.array([0.4, 0.2, 0.1, -0.1, -0.2, -0.4]))
    d = np.exp(np.array([-0.3, 0.3, -0.15, 0.15, -0.05, 0.05]))
    rows = []
    for i in range(n):
        h = int(rng.choice(6, p=[0.4, 0.12, 0.12, 0.12, 0.12, 0.12]))
        v = int(rng.choice([j for j in range(6) if j != h]))
        lh, la = (1.5 * a[h] * d[v], 1.1 * a[v] * d[h])
        rows.append({'match_id': i, 'date': pd.Timestamp('2020-01-01', tz='UTC') + pd.Timedelta(days=i), 'home': teams[h], 'away': teams[v], 'home_npxg': rng.gamma(20, lh / 20) if noise else lh, 'away_npxg': rng.gamma(20, la / 20) if noise else la, 'home_goals': int(rng.poisson(lh * 1.12)), 'away_goals': int(rng.poisson(la * 1.12))})
    return (pd.DataFrame(rows), a, d)
