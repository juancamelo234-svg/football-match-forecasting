"""Tiros -> npxG completo -> ratings -> goles totales -> probabilidades.

Version previa al partido. No incorpora minutos de rojas, xGOT ni pesos
por marcador sin validar. Los resultados oficiales vienen del calendario.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import poisson
from .ratings import fit_ratings, predict_npxg, validate_matches
from .validation import validate_competition
SHOT_COLS = ['shot_id', 'match_id', 'team', 'xg', 'is_penalty', 'is_set_piece']

def shots_to_match(shots, fixtures, drop_red_card_minutes=False):
    """Requiere calendario COMPLETO de partidos con cobertura verificada.

    fixtures: match_id,date,home,away,home_goals,away_goals,shots_complete.
    shots_complete=True significa descarga completa verificada, no imputada.
    Asi distinguimos cero tiros de una descarga ausente. Autogoles quedan en
    el resultado oficial; no se reconstruye el resultado desde result=Goal.
    """
    if drop_red_card_minutes:
        raise ValueError('Filtro de rojas desactivado: falta exposicion temporal')
    league = validate_competition(fixtures)
    if league:
        if 'competition_id' not in shots or shots.competition_id.isna().any() or (not shots.competition_id.eq(league).all()):
            raise ValueError('Tiros de otra liga o sin identificador de competicion')
    elif 'competition_id' in shots:
        raise ValueError('Calendario sin liga para tiros identificados')
    required = ['match_id', 'date', 'home', 'away', 'home_goals', 'away_goals', 'shots_complete']
    if set(required) - set(fixtures) or set(SHOT_COLS) - set(shots):
        raise ValueError('Esquema incompleto de tiros/calendario')
    f, s = (fixtures[required].copy(), shots[SHOT_COLS].copy())
    if f.isna().any().any() or s.isna().any().any():
        raise ValueError('Campos obligatorios ausentes')
    if not pd.api.types.is_bool_dtype(f.shots_complete) or not f.shots_complete.all():
        raise ValueError('Hay partidos sin cobertura de tiros confirmada')
    if f.match_id.duplicated().any() or s.duplicated(['match_id', 'shot_id']).any():
        raise ValueError('Identificadores duplicados')
    f['date'] = pd.to_datetime(f.date, utc=True)
    if f.date.isna().any() or (f.home == f.away).any():
        raise ValueError('Partido invalido')
    for col in ['home_goals', 'away_goals']:
        v = pd.to_numeric(f[col], errors='raise')
        if not np.isfinite(v).all() or (v < 0).any() or (v % 1 != 0).any():
            raise ValueError('Marcador oficial invalido')
        f[col] = v.astype(int)
    for col in ['is_penalty', 'is_set_piece']:
        if not pd.api.types.is_bool_dtype(s[col]):
            raise ValueError('Indicadores de tiros deben ser booleanos')
    if not np.isfinite(s.xg).all() or not s.xg.between(0, 1).all():
        raise ValueError('xG de cada tiro debe estar entre 0 y 1')
    base = pd.concat([f.assign(team=f.home, is_home=True, goals=f.home_goals), f.assign(team=f.away, is_home=False, goals=f.away_goals)], ignore_index=True)
    base = base[['match_id', 'date', 'team', 'is_home', 'goals']]
    check = s.merge(base[['match_id', 'team']], on=['match_id', 'team'], how='left', indicator=True, validate='many_to_one')
    if (check['_merge'] != 'both').any():
        raise ValueError('Tiro sin equipo/partido valido en calendario')
    nonpen = ~s.is_penalty
    op = nonpen & ~s.is_set_piece
    s['open_play_xg'] = s.xg.where(op, 0.0)
    s['set_piece_xg'] = s.xg.where(nonpen & s.is_set_piece, 0.0)
    s['npxg'] = s.xg.where(nonpen, 0.0)
    s['penalty_xg'] = s.xg.where(s.is_penalty, 0.0)
    s['shots'] = nonpen.astype(int)
    s['shots_open_play'] = op.astype(int)
    values = ['open_play_xg', 'set_piece_xg', 'npxg', 'penalty_xg', 'shots', 'shots_open_play']
    agg = s.groupby(['match_id', 'team'], as_index=False)[values].sum()
    out = base.merge(agg, on=['match_id', 'team'], how='left', validate='one_to_one')
    out[values] = out[values].fillna(0.0)
    out['npxg_per_shot'] = out.npxg.div(out.shots.replace(0, np.nan))
    out['xgot'] = np.nan
    if league:
        out['competition_id'] = league
    return out

def match_wide(team_match):
    t = team_match.copy()
    league = None
    if 'competition_id' in t:
        if t.competition_id.isna().any() or t.competition_id.nunique() != 1:
            raise ValueError('No mezclar ligas en tiros agregados')
        league = t.competition_id.iloc[0]
        t = t.drop(columns='competition_id')
    if not pd.api.types.is_bool_dtype(t.is_home) or t.is_home.isna().any():
        raise ValueError('is_home debe ser booleano completo')
    if t.duplicated(['match_id', 'is_home']).any() or not t.groupby('match_id').size().eq(2).all():
        raise ValueError('Se requiere exactamente un local y un visitante')
    sides = []
    for home, prefix in [(True, 'home'), (False, 'away')]:
        part = t.loc[t.is_home == home].drop(columns='is_home')
        part = part.rename(columns={c: f'{prefix}_{c}' for c in part if c not in ['match_id', 'date']})
        sides.append(part)
    m = sides[0].merge(sides[1], on=['match_id', 'date'], how='outer', validate='one_to_one', indicator=True)
    if (m['_merge'] != 'both').any():
        raise ValueError('Fechas/localias inconsistentes')
    out = m.drop(columns='_merge').rename(columns={'home_team': 'home', 'away_team': 'away'})
    if league:
        out['competition_id'] = league
    validate_competition(out)
    return out

def _tau(x, y, lh, la, rho):
    vals = np.array([1 - lh * la * rho, 1 + lh * rho, 1 + la * rho, 1 - rho])
    if not np.isfinite(vals).all() or (vals < 0).any():
        raise ValueError('rho incompatible con estas intensidades; no se recorta')
    t = np.ones_like(x, dtype=float)
    for mask, value in zip([(x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)], vals):
        t[mask] = value
    return t

def score_matrix(lambda_home, lambda_away, rho=0.0, max_goals=10, tail_tol=1e-10):
    """max_goals es minimo: amplia soporte para controlar masa omitida.

    rho=0 es baseline independiente, hasta validar una correccion Dixon-Coles.
    """
    if not np.isfinite([lambda_home, lambda_away, rho, tail_tol]).all() or min(lambda_home, lambda_away) < 0 or (not 0 < tail_tol < 0.01):
        raise ValueError('Intensidades/rho/tolerancia invalidos')
    if not isinstance(max_goals, int) or not 1 <= max_goals <= 10000:
        raise ValueError('max_goals debe ser entero entre 1 y 10000')
    q = max(poisson.ppf(1 - tail_tol / 2, lambda_home), poisson.ppf(1 - tail_tol / 2, lambda_away))
    if not np.isfinite(q) or max(q, max_goals) > 1000:
        raise ValueError('Soporte excesivo para este modelo de futbol')
    k = np.arange(max(max_goals, int(q)) + 1)
    gx, gy = np.meshgrid(k, k, indexing='ij')
    m = np.outer(poisson.pmf(k, lambda_home), poisson.pmf(k, lambda_away)) * _tau(gx, gy, lambda_home, lambda_away, rho)
    return m / m.sum()

def markets(matrix, line=2.5):
    """Solo lineas de medio gol; lineas asiaticas requieren otro liquidador."""
    m = np.asarray(matrix, float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or (not np.isfinite(m).all()) or (m < 0).any() or (not np.isclose(m.sum(), 1)):
        raise ValueError('Matriz de probabilidades invalida')
    if not np.isfinite(line) or line < 0 or (not np.isclose(line % 1, 0.5)):
        raise ValueError('Usar lineas de medio gol, por ejemplo 2.5')
    x, y = np.indices(m.shape)
    return {'p_home': float(m[x > y].sum()), 'p_draw': float(np.trace(m)), 'p_away': float(m[x < y].sum()), 'p_over': float(m[x + y > line].sum()), 'p_under': float(m[x + y < line].sum()), 'p_btts': float(m[(x > 0) & (y > 0)].sum())}

def _validate_probabilities(probs, outcomes):
    p, o = (np.asarray(probs, float), np.asarray(outcomes))
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0 or (o.shape != (len(p),)):
        raise ValueError('Se requieren probabilidades (n,3) y resultados (n,) no vacios')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or (not np.allclose(p.sum(1), 1, atol=1e-09, rtol=0)):
        raise ValueError('Probabilidades invalidas')
    if not np.isin(o, [0, 1, 2]).all():
        raise ValueError('Resultado debe ser 0, 1 o 2')
    return (p, o.astype(int))

def rps(probs, outcomes):
    p, o = _validate_probabilities(probs, outcomes)
    observed = np.eye(3)[o]
    return float(np.mean(np.sum((p.cumsum(1)[:, :2] - observed.cumsum(1)[:, :2]) ** 2, axis=1) / 2))

def probability_metrics(probs, outcomes):
    """Score a three-outcome distribution; log loss uses a 1e-15 floor."""
    p, y = _validate_probabilities(probs, outcomes)
    return {
        'n': len(y),
        'rps': rps(p, y),
        'log_loss': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
        'accuracy': float((p.argmax(axis=1) == y).mean()),
        'mean_draw_probability': float(p[:, 1].mean()),
        'observed_draw_rate': float((y == 1).mean()),
    }

def devig(odds):
    o = np.asarray(odds, float)
    if o.ndim != 2 or o.shape[1] != 3 or (not np.isfinite(o).all()) or (o <= 1).any():
        raise ValueError('Cuotas decimales finitas >1, forma (n,3)')
    inv = 1 / o
    return inv / inv.sum(1, keepdims=True)

def calibration_table(probs, outcomes, bins=10):
    p, o = _validate_probabilities(probs, outcomes)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError('bins debe ser entero positivo')
    observed = np.eye(3)[o]
    rows = []
    for col, name in enumerate(['local', 'empate', 'visitante']):
        b = np.minimum((p[:, col] * bins).astype(int), bins - 1)
        for i in range(bins):
            selected = b == i
            rows.append({'resultado': name, 'desde': i / bins, 'hasta': (i + 1) / bins, 'n': int(selected.sum()), 'predicha': float(p[selected, col].mean()) if selected.any() else np.nan, 'observada': float(observed[selected, col].mean()) if selected.any() else np.nan})
    return pd.DataFrame(rows)

def run(matches, half_life=120.0, shrink_k=8.0, rho=0.0, min_history=40,
        test_start=None, test_end=None, freeze_training=False):
    """Walk-forward diario. Conversion provisional npxG -> goles totales:
    factor por localia = suma ponderada goles / suma ponderada npxG del pasado.
    Incluye a nivel liga penaltis/autogoles y desviaciones de finalizacion.
    Es un baseline que debe validarse; no estima habilidades de penaltis.
    """
    if not isinstance(min_history, int) or min_history < 1:
        raise ValueError('min_history debe ser entero positivo')
    df = validate_matches(matches).sort_values('date')
    def boundary(value):
        if value is None:
            return None
        result = pd.to_datetime(value, utc=True)
        if pd.isna(result) or result != result.normalize():
            raise ValueError('Test boundaries must be valid UTC midnight dates')
        return result
    start, end = boundary(test_start), boundary(test_end)
    if start is not None and end is not None and start >= end:
        raise ValueError('test_start must precede exclusive test_end')
    if freeze_training and start is None:
        raise ValueError('Frozen training requires test_start')
    for c in ['home_goals', 'away_goals']:
        if c not in df:
            raise ValueError('Faltan goles oficiales')
        v = df[c].to_numpy(float)
        if not np.isfinite(v).all() or (v < 0).any() or (v % 1 != 0).any():
            raise ValueError('Goles oficiales invalidos')
        df[c] = v.astype(int)
    rows = []
    skipped = 0
    cached = None
    for day, block in df.groupby(df.date.dt.normalize()):
        if (start is not None and day < start) or (end is not None and day >= end):
            continue
        cutoff = start if freeze_training else day
        past = df.loc[df.date < cutoff]
        if len(past) < min_history:
            skipped += len(block)
            continue
        if cached is None or not freeze_training:
            fit = fit_ratings(past, cutoff, half_life_days=half_life, shrink_k=shrink_k)
            w = np.exp2(-(cutoff - past.date).dt.total_seconds().to_numpy() / 86400 / half_life)
            factors = {s: float(np.dot(w, past[f'{s}_goals']) / np.dot(w, past[f'{s}_npxg'])) for s in ['home', 'away']}
            outcomes = np.where(past.home_goals > past.away_goals, 0, np.where(past.home_goals == past.away_goals, 1, 2))
            base = np.bincount(outcomes, minlength=3) / len(outcomes)
            cached = fit, factors, base
        else:
            fit, factors, base = cached
        for _, m in block.iterrows():
            pred = predict_npxg(fit, m.home, m.away)
            lh = pred['expected_npxg_home'] * factors['home']
            la = pred['expected_npxg_away'] * factors['away']
            row = {'date': m.date, 'home': m.home, 'away': m.away, **pred, 'lambda_home': lh, 'lambda_away': la, 'goal_factor_home': factors['home'], 'goal_factor_away': factors['away'], **markets(score_matrix(lh, la, rho)), 'outcome': 0 if m.home_goals > m.away_goals else 1 if m.home_goals == m.away_goals else 2, **dict(zip(['base_home', 'base_draw', 'base_away'], base))}
            if 'match_id' in m:
                row['match_id'] = m.match_id
            if 'competition_id' in m:
                row['competition_id'] = m.competition_id
            row.update(training_cutoff=cutoff.isoformat(), training_matches=len(past),
                       training_max_date=past.date.max().isoformat())
            rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs.update({'skipped_warmup': skipped, 'input_matches': len(df), 'note': 'Evaluacion descriptiva; separar validacion/test ANTES de optimizar'})
    if len(out):
        metrics = probability_metrics(out[['p_home', 'p_draw', 'p_away']], out.outcome)
        baseline = probability_metrics(out[['base_home', 'base_draw', 'base_away']], out.outcome)
        out.attrs.update({key: value for key, value in metrics.items() if key != 'n'})
        out.attrs.update({'baseline_' + key: value for key, value in baseline.items() if key != 'n'})
    return out
