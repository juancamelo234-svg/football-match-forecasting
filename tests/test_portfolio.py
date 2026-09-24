"""Release boundary checks: league isolation and reproducible CLI output."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from football_forecasting.cli import main
from football_forecasting.ratings import fit_ratings, predict_npxg
from football_forecasting.synthetic import synthetic
from football_forecasting.validation import validate_competition


class PortfolioTests(unittest.TestCase):
    def tagged_data(self):
        frame, _, _ = synthetic(100)
        frame["competition_id"] = "ENG_PL"
        for column in ["home", "away", "match_id"]:
            frame[column] = "ENG_PL::" + frame[column].astype(str)
        return frame

    def test_mixed_leagues_rejected(self):
        frame = self.tagged_data()
        frame.loc[0, "competition_id"] = "COL_A"
        with self.assertRaises(ValueError):
            fit_ratings(frame, "2021-01-01")

    def test_ratings_cannot_predict_other_league(self):
        fit = fit_ratings(self.tagged_data(), "2021-01-01")
        with self.assertRaises(ValueError):
            predict_npxg(fit, "COL_A::A", "COL_A::B")

    def test_public_input_requires_league(self):
        frame, _, _ = synthetic(10)
        with self.assertRaises(ValueError):
            validate_competition(frame, required=True)

    def test_demo_reproducible_and_existing_output_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a", Path(directory) / "b"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["demo", "--output", str(first)])
                main(["demo", "--output", str(second)])
            for name in ["predictions.csv", "metrics.json", "synthetic_matches.csv", "run_manifest.json"]:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["demo", "--output", str(first)])

    def test_evaluate_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "matches.csv", Path(directory) / "evaluation"
            self.tagged_data().to_csv(source, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["evaluate", "--input", str(source), "--output", str(output)])
            self.assertTrue((output / "metrics.json").is_file())
