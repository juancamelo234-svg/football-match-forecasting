"""Ratings xG: cuasi-verosimilitud Poisson con regularizacion logaritmica.

El xG es continuo: se estima su media, no se supone que sea un conteo.
Ataque y defensa tienen media geometrica 1; los interceptos se estiman juntos.
shrink_k es la intensidad de una penalizacion L2, NO partidos ficticios.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .validation import validate_competition, validate_fit_teams
COLS = ['date', 'home', 'away', 'home_npxg', 'away_npxg']

def validate_matches(history):
    validate_competition(history)
    if 'result_status' in history and (not history.result_status.eq('recorded_by_provider').all()):
        raise ValueError('Resultados administrativos o en cuarentena no admitidos en el modelo deportivo')
    missing = set(COLS) - set(history.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    h = history.copy()
    h['date'] = pd.to_datetime(h['date'], utc=True, errors='raise')
    if h[COLS].isna().any().any():
        raise ValueError('Fechas, equipos y npxG deben estar completos')
    if (h.home == h.away).any():
        raise ValueError('Un equipo no puede jugar contra si mismo')
    if h.duplicated(['date', 'home', 'away']).any():
        raise ValueError('Partidos duplicados')
    for c in ['home_npxg', 'away_npxg']:
        h[c] = pd.to_numeric(h[c], errors='raise')
        if not np.isfinite(h[c]).all() or (h[c] < 0).any():
            raise ValueError('npxG debe ser finito y no negativo')
    if not all((isinstance(t, str) and t.strip() for t in pd.concat([h.home, h.away]))):
        raise ValueError('Equipos deben tener nombres no vacios')
    return h

def normalize_ratings(attack, defence, mu_home, mu_away):
    """Cambia la escala conservando exactamente todas las intensidades."""
    a, d = (np.asarray(attack, float), np.asarray(defence, float))
    if not (np.isfinite(a).all() and np.isfinite(d).all()) or (a <= 0).any() or (d <= 0).any():
        raise ValueError('Ratings positivos y finitos requeridos')
    ga, gd = (np.exp(np.log(a).mean()), np.exp(np.log(d).mean()))
    return (a / ga, d / gd, mu_home * ga * gd, mu_away * ga * gd)

def fit_ratings(history, as_of, half_life_days=120.0, n_iter=2000, shrink_k=8.0, tol=1e-07, penalty_mode='legacy', reference_mass=400.0):
    """Usa fechas UTC anteriores al DIA de as_of: excluye todo ese dia.

    Convencion conservadora para fuentes sin hora fiable de fin del partido.
    No comprueba disponibilidad historica/versiones del proveedor de xG.
    """
    if not np.isfinite([half_life_days, shrink_k, tol]).all() or half_life_days <= 0 or shrink_k < 0 or (tol <= 0):
        raise ValueError('Parametros invalidos')
    if not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError('n_iter debe ser entero positivo')
    if penalty_mode not in {'legacy', 'normalized'} or not np.isfinite(reference_mass) or reference_mass <= 0:
        raise ValueError('Escala de penalizacion invalida')
    h = validate_matches(history)
    cutoff = pd.Timestamp(as_of)
    cutoff = (cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff.tz_convert('UTC')).normalize()
    if pd.isna(cutoff):
        raise ValueError('Fecha de corte invalida')
    past = h.loc[h.date < cutoff].copy()
    if past.empty:
        raise ValueError('No hay partidos anteriores al dia de prediccion')
    w = np.exp2(-(cutoff - past.date).dt.total_seconds().to_numpy() / 86400 / half_life_days)
    if w.sum() <= 0:
        raise ValueError('Todo el historial tiene peso numericamente cero')
    teams = np.union1d(past.home, past.away)
    index = {t: i for i, t in enumerate(teams)}
    hi, ai = (past.home.map(index).to_numpy(), past.away.map(index).to_numpy())
    n, m = (len(teams), len(past))
    y = np.r_[past.home_npxg.to_numpy(), past.away_npxg.to_numpy()]
    weights = np.r_[w, w]
    raw_weight_mass = float(weights.sum())
    if penalty_mode == 'normalized':
        weights = weights * (reference_mass / raw_weight_mass)
    if np.dot(w, past.home_npxg) <= 0 or np.dot(w, past.away_npxg) <= 0:
        raise ValueError('Se necesita xG positivo agregado para ambos tipos de localia')
    attackers, defenders = (np.r_[hi, ai], np.r_[ai, hi])
    side = np.r_[np.zeros(m, int), np.ones(m, int)]
    initial = np.zeros(2 + 2 * n)
    initial[:2] = np.log([np.dot(w, past.home_npxg) / w.sum(), np.dot(w, past.away_npxg) / w.sum()])

    def objective(theta):
        raw_a, raw_d = (theta[2:2 + n], theta[2 + n:])
        a, d = (raw_a - raw_a.mean(), raw_d - raw_d.mean())
        eta = theta[side] + a[attackers] + d[defenders]
        mean = np.exp(eta)
        residual = weights * (mean - y)
        loss = np.dot(weights, mean - y * eta) + 0.5 * shrink_k * np.dot(theta[2:], theta[2:])
        ga = np.bincount(attackers, weights=residual, minlength=n)
        gd = np.bincount(defenders, weights=residual, minlength=n)
        grad = np.r_[np.bincount(side, weights=residual, minlength=2), ga - ga.mean() + shrink_k * raw_a, gd - gd.mean() + shrink_k * raw_d]
        return (loss, grad)
    result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=[(-15, 8)] * 2 + [(-8, 8)] * (2 * n), options={'maxiter': n_iter, 'ftol': tol * 0.01, 'gtol': tol})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f'El ajuste no convergio: {result.message}')
    if (np.abs(result.x[2:]) > 7.999).any() or (result.x[:2] < -14.999).any() or (result.x[:2] > 7.999).any():
        raise RuntimeError('Ajuste en limite numerico; revisar datos')
    a, d = (result.x[2:2 + n], result.x[2 + n:])
    exposure = np.bincount(hi, weights=w, minlength=n) + np.bincount(ai, weights=w, minlength=n)
    w2 = np.bincount(hi, weights=w * w, minlength=n) + np.bincount(ai, weights=w * w, minlength=n)
    return {'competition_id': validate_competition(h), 'attack': dict(zip(teams, np.exp(a - a.mean()))), 'defence': dict(zip(teams, np.exp(d - d.mean()))), 'mu_home': float(np.exp(result.x[0])), 'mu_away': float(np.exp(result.x[1])), 'n_eff': dict(zip(teams, exposure)), 'effective_sample_size': dict(zip(teams, exposure ** 2 / w2)), 'converged': True, 'iterations': int(result.nit), 'objective': float(result.fun), 'as_of': cutoff.isoformat(), 'training_matches': m, 'half_life_days': float(half_life_days), 'shrink_k': float(shrink_k), 'penalty_mode': penalty_mode, 'reference_mass': float(reference_mass), 'raw_weight_mass': raw_weight_mass}

def predict_npxg(fit, home, away):
    """Equipo nuevo: prior de liga (rating 1), marcado explicitamente."""
    validate_fit_teams(fit, home, away)
    if home == away:
        raise ValueError('Equipos deben ser distintos')
    a, d = (fit['attack'], fit['defence'])
    return {'expected_npxg_home': fit['mu_home'] * a.get(home, 1.0) * d.get(away, 1.0), 'expected_npxg_away': fit['mu_away'] * a.get(away, 1.0) * d.get(home, 1.0), 'new_team': home not in a or away not in a}

def build_features(matches, half_lives=(45.0, 150.0), shrink_k=8.0, min_history=20):
    """Predictores previos al partido; no devuelve el xG observado del objetivo."""
    df = validate_matches(matches).sort_values('date')
    if not half_lives or len(set(half_lives)) != len(half_lives) or any((x <= 0 for x in half_lives)):
        raise ValueError('Vidas medias positivas y distintas requeridas')
    rows = []
    for day, block in df.groupby(df.date.dt.normalize()):
        if (df.date < day).sum() < min_history:
            continue
        fits = {v: fit_ratings(df, day, half_life_days=v, shrink_k=shrink_k) for v in half_lives}
        for _, match in block.iterrows():
            row = {'date': match.date, 'home': match.home, 'away': match.away}
            if 'competition_id' in match:
                row['competition_id'] = match.competition_id
            for hl, fit in fits.items():
                for side, team in [('home', match.home), ('away', match.away)]:
                    for label, key in [('att', 'attack'), ('def', 'defence'), ('neff', 'n_eff')]:
                        row[f'{side}_{label}_{hl:g}'] = fit[key].get(team, 0.0 if key == 'n_eff' else 1.0)
            row.update(predict_npxg(fits[max(half_lives)], match.home, match.away))
            for side in ['home', 'away']:
                for label in ['att', 'def']:
                    row[f'{side}_{label}_form'] = row[f'{side}_{label}_{min(half_lives):g}'] / row[f'{side}_{label}_{max(half_lives):g}']
            rows.append(row)
    return pd.DataFrame(rows)

def rating_table(fit):
    t = pd.DataFrame({'ataque': fit['attack'], 'defensa': fit['defence'], 'partidos_equivalentes': fit['n_eff'], 'tamano_efectivo': fit['effective_sample_size']})
    t['fuerza_neta'] = t.ataque / t.defensa
    return t.sort_values('fuerza_neta', ascending=False)
