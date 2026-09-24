"""Pruebas de propiedades conocidas; no prueban ventaja predictiva real."""
import unittest
from football_forecasting.synthetic import synthetic
import numpy as np
import pandas as pd
from football_forecasting.ratings import fit_ratings, predict_npxg, normalize_ratings, build_features
from football_forecasting.probabilities import shots_to_match, match_wide, score_matrix, markets, rps, devig, calibration_table, run

class ModelTests(unittest.TestCase):

    def test_known_strengths_unbalanced_schedule(self):
        df, a, d = synthetic()
        f = fit_ratings(df, df.date.max() + pd.Timedelta(days=1), half_life_days=100000000.0, shrink_k=0, tol=1e-10)
        np.testing.assert_allclose(list(f['attack'].values()), a, rtol=0.001)
        np.testing.assert_allclose(list(f['defence'].values()), d, rtol=0.001)
        np.testing.assert_allclose([f['mu_home'], f['mu_away']], [1.5, 1.1], rtol=0.001)

    def test_equal_teams(self):
        df, _, _ = synthetic()
        df['home_npxg'], df['away_npxg'] = (1.5, 1.1)
        f = fit_ratings(df, df.date.max() + pd.Timedelta(days=1))
        np.testing.assert_allclose(list(f['attack'].values()), 1, atol=1e-06)
        np.testing.assert_allclose(list(f['defence'].values()), 1, atol=1e-06)
        np.testing.assert_allclose([f['mu_home'], f['mu_away']], [1.5, 1.1], atol=1e-06)

    def test_noisy_recovery(self):
        df, a, d = synthetic(2400, True)
        f = fit_ratings(df, df.date.max() + pd.Timedelta(days=1), half_life_days=100000000.0, shrink_k=0)
        np.testing.assert_allclose(list(f['attack'].values()), a, rtol=0.08)
        np.testing.assert_allclose(list(f['defence'].values()), d, rtol=0.08)

    def test_scale_invariance(self):
        a = np.array([0.8, 1.7, 2.1])
        d = np.array([0.5, 0.9, 1.4])
        aa, dd, mh, ma = normalize_ratings(a, d, 1.5, 1.1)
        np.testing.assert_allclose(1.5 * np.outer(a, d), mh * np.outer(aa, dd), atol=1e-12)
        np.testing.assert_allclose(1.1 * np.outer(a, d), ma * np.outer(aa, dd), atol=1e-12)
        self.assertAlmostEqual(np.exp(np.log(aa).mean()), 1)

    def test_future_and_same_day_excluded(self):
        df, _, _ = synthetic(100)
        cutoff = df.date.iloc[70] + pd.Timedelta(hours=20)
        f = fit_ratings(df, cutoff)
        changed = df.copy()
        changed.loc[changed.date >= cutoff.normalize(), ['home_npxg', 'away_npxg']] = 1000.0
        ff = fit_ratings(changed, cutoff)
        self.assertEqual(f, ff)
        self.assertEqual(f['training_matches'], 70)

    def test_zero_xg_team_and_new_team(self):
        df, _, _ = synthetic()
        df.loc[df.home == 'A', 'home_npxg'] = 0.0
        df.loc[df.away == 'A', 'away_npxg'] = 0.0
        f = fit_ratings(df, df.date.max() + pd.Timedelta(days=1), shrink_k=8)
        self.assertTrue(np.isfinite(list(f['attack'].values())).all())
        p = predict_npxg(f, 'NEW', 'B')
        self.assertTrue(p['new_team'])
        self.assertGreater(p['expected_npxg_home'], 0)

    def test_bad_input_and_convergence(self):
        df, _, _ = synthetic()
        for h in [pd.concat([df, df.iloc[:1]]), df.assign(home_npxg=-1), df.assign(home_npxg=np.nan)]:
            with self.assertRaises(ValueError):
                fit_ratings(h, '2021-01-01')
        with self.assertRaises(RuntimeError):
            fit_ratings(df, '2021-01-01', n_iter=1)
        with self.assertRaises(ValueError):
            fit_ratings(df, '2019-01-01')
        with self.assertRaises(ValueError):
            fit_ratings(df, '2021-01-01', half_life_days=0)

    def test_regularization_shrinks_log_strengths(self):
        df, _, _ = synthetic()
        f = fit_ratings(df, '2021-01-01', shrink_k=0)
        g = fit_ratings(df, '2021-01-01', shrink_k=100)
        strength = lambda x: sum((np.square(np.log(list(x[k].values()))).sum() for k in ['attack', 'defence']))
        self.assertLess(strength(g), strength(f))

    def fixture_shots(self):
        f = pd.DataFrame([{'match_id': 1, 'date': '2023-01-01', 'home': 'A', 'away': 'B', 'home_goals': 2, 'away_goals': 1, 'shots_complete': True}])
        s = pd.DataFrame({'shot_id': [1, 2, 3], 'match_id': [1, 1, 1], 'team': ['A'] * 3, 'xg': [0.2, 0.3, 0.76], 'is_penalty': [False, False, True], 'is_set_piece': [False, True, True]})
        return (f, s)

    def test_aggregation_zero_shots_official_goals_and_set_pieces(self):
        f, s = self.fixture_shots()
        t = shots_to_match(s, f)
        self.assertEqual(len(t), 2)
        a = t.loc[t.team == 'A'].iloc[0]
        b = t.loc[t.team == 'B'].iloc[0]
        self.assertAlmostEqual(a.npxg, 0.5)
        self.assertAlmostEqual(a.open_play_xg, 0.2)
        self.assertAlmostEqual(a.set_piece_xg, 0.3)
        self.assertAlmostEqual(a.penalty_xg, 0.76)
        self.assertAlmostEqual(a.npxg_per_shot, 0.25)
        self.assertEqual(b.shots, 0)
        self.assertTrue(pd.isna(b.npxg_per_shot))
        self.assertTrue(t.xgot.isna().all())
        m = match_wide(t)
        self.assertEqual(m.home_goals.iloc[0], 2)
        self.assertEqual(m.away_goals.iloc[0], 1)

    def test_coverage_duplicate_and_red_filter_guards(self):
        f, s = self.fixture_shots()
        with self.assertRaises(ValueError):
            shots_to_match(s, f.assign(shots_complete=False))
        with self.assertRaises(ValueError):
            shots_to_match(pd.concat([s, s.iloc[:1]]), f)
        with self.assertRaises(ValueError):
            shots_to_match(s, f, True)
        with self.assertRaises(ValueError):
            shots_to_match(s.assign(team='OTHER'), f)
        t = shots_to_match(s, f)
        with self.assertRaises(ValueError):
            match_wide(pd.concat([t, t.iloc[:1]]))

    def test_probability_known_cases(self):
        m = score_matrix(0, 0)
        self.assertEqual(m[0, 0], 1)
        p = markets(m)
        self.assertEqual(p['p_draw'], 1)
        self.assertEqual(p['p_btts'], 0)
        m = score_matrix(1.2, 1.7)
        p = markets(m)
        self.assertAlmostEqual(p['p_btts'], (1 - np.exp(-1.2)) * (1 - np.exp(-1.7)), places=8)
        self.assertAlmostEqual(p['p_home'] + p['p_draw'] + p['p_away'], 1)
        self.assertAlmostEqual(p['p_over'] + p['p_under'], 1)
        self.assertGreater(score_matrix(8, 8).shape[0], 11)
        np.testing.assert_allclose(score_matrix(1.2, 1.7, -0.05).T, score_matrix(1.7, 1.2, -0.05))

    def test_probability_guards(self):
        for lh, la, rho in [(1, 1, 2), (-1, 2, 0), (np.nan, 1, 0), (30, 30, -0.05)]:
            with self.assertRaises(ValueError):
                score_matrix(lh, la, rho)
        with self.assertRaises(ValueError):
            markets(score_matrix(1, 1), 2.0)

    def test_scoring_calibration_and_odds(self):
        self.assertEqual(rps(np.eye(3), [0, 1, 2]), 0)
        self.assertEqual(rps([[1, 0, 0]], [2]), 1)
        np.testing.assert_allclose(devig([[2, 3, 4]]).sum(1), 1)
        with self.assertRaises(ValueError):
            devig([[0, 3, 4]])
        with self.assertRaises(ValueError):
            rps([[0.2, 0.2, 0.2]], [0])
        c = calibration_table(np.eye(3), [0, 1, 2])
        self.assertTrue(c.groupby('resultado').n.sum().eq(3).all())

    def test_features_exclude_targets_and_keep_form(self):
        df, _, _ = synthetic(45)
        features = build_features(df, min_history=40)
        self.assertEqual(len(features), 5)
        self.assertNotIn('home_npxg', features)
        self.assertIn('home_att_form', features)

    def test_walk_forward_goal_conversion_and_no_future_leakage(self):
        df, _, _ = synthetic(48)
        out = run(df, min_history=40)
        self.assertEqual(len(out), 8)
        self.assertEqual(out.attrs['skipped_warmup'], 40)
        self.assertTrue(np.isfinite(out.lambda_home).all())
        self.assertTrue(np.isfinite(out.attrs['baseline_rps']))
        first = out.iloc[0]
        past = df.iloc[:40]
        w = np.exp2(-(df.date.iloc[40] - past.date).dt.total_seconds().to_numpy() / 86400 / 120)
        self.assertAlmostEqual(first.goal_factor_home, np.dot(w, past.home_goals) / np.dot(w, past.home_npxg))
        changed = df.copy()
        changed.loc[changed.index >= 41, ['home_npxg', 'away_npxg']] = 100.0
        changed.loc[changed.index >= 41, ['home_goals', 'away_goals']] = 20
        again = run(changed, min_history=40)
        pd.testing.assert_series_equal(out.iloc[0], again.iloc[0])
if __name__ == '__main__':
    unittest.main(verbosity=2)
