"""Regression checks for input contracts and explicit temporal holdouts."""
import unittest

import numpy as np
import pandas as pd

from football_forecasting.probabilities import probability_metrics, run
from football_forecasting.ratings import validate_matches
from football_forecasting.synthetic import synthetic


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame, _, _ = synthetic(60, noise=True)
        self.start = self.frame.date.iloc[45].normalize()
        self.end = self.frame.date.iloc[55].normalize()

    def test_duplicate_id_rejected_even_for_different_fixture(self):
        self.frame.loc[1, 'match_id'] = self.frame.loc[0, 'match_id']
        with self.assertRaises(ValueError):
            validate_matches(self.frame)

    def test_numeric_string_scores_equal_integer_scores(self):
        strings = self.frame.copy()
        for side in ['home', 'away']:
            strings[side + '_goals'] = strings[side + '_goals'].astype(str)
        expected = run(self.frame, min_history=55)
        pd.testing.assert_frame_equal(run(strings, min_history=55), expected)

    def test_holdout_boundaries_and_training_metadata(self):
        out = run(self.frame, test_start=self.start, test_end=self.end)
        self.assertEqual(len(out), 10)
        self.assertTrue((out.date >= self.start).all())
        self.assertTrue((out.date < self.end).all())
        self.assertTrue((pd.to_datetime(out.training_max_date, utc=True)
                         < pd.to_datetime(out.training_cutoff, utc=True)).all())
        self.assertEqual(out.training_matches.tolist(), list(range(45, 55)))

    def test_frozen_predictions_ignore_all_test_targets(self):
        options = dict(test_start=self.start, test_end=self.end, freeze_training=True)
        out = run(self.frame, **options)
        changed = self.frame.copy()
        mask = changed.date >= self.start
        changed.loc[mask, ['home_npxg', 'away_npxg']] = 100.0
        changed.loc[mask, ['home_goals', 'away_goals']] = [20, 0]
        again = run(changed, **options)
        columns = ['p_home', 'p_draw', 'p_away', 'lambda_home', 'lambda_away',
                   'base_home', 'base_draw', 'base_away']
        pd.testing.assert_frame_equal(out[columns], again[columns])
        self.assertTrue(out.training_matches.eq(45).all())

    def test_same_day_scores_and_npxg_do_not_enter_replay(self):
        frame = self.frame.copy()
        frame.loc[46, 'date'] = frame.loc[45, 'date'] + pd.Timedelta(hours=2)
        options = dict(test_start=self.start, test_end=self.start + pd.Timedelta(days=1))
        before = run(frame, **options)
        frame.loc[45:, ['home_npxg', 'away_npxg']] = 25.0
        frame.loc[45:, ['home_goals', 'away_goals']] = [15, 0]
        after = run(frame, **options)
        self.assertEqual(len(before), 2)
        pd.testing.assert_frame_equal(before[['p_home', 'p_draw', 'p_away']],
                                      after[['p_home', 'p_draw', 'p_away']])

    def test_invalid_holdout_dates(self):
        for options in [dict(freeze_training=True),
                        dict(test_start=self.end, test_end=self.start),
                        dict(test_start='2020-01-01T12:00:00Z'),
                        dict(test_start='NaT')]:
            with self.assertRaises(ValueError):
                run(self.frame, **options)

    def test_metrics_against_hand_calculation(self):
        result = probability_metrics([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]], [0, 2])
        self.assertAlmostEqual(result['rps'], (0.085 + 0.265) / 2)
        self.assertAlmostEqual(result['log_loss'], -(np.log(0.6) + np.log(0.3)) / 2)
        self.assertEqual(result['accuracy'], 0.5)
