import unittest

import numpy as np
import pandas as pd

from src.rental_model import (
    LEAKAGE_COLUMNS,
    build_features,
    time_split,
    train_and_evaluate,
)


def make_synthetic_days(n_days: int = 730) -> pd.DataFrame:
    """Two years of fake days where rentals peak at temp 0.6 and drop in bad weather."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2011-01-01", periods=n_days, freq="D")
    temp = rng.uniform(0.1, 0.9, n_days)
    weathersit = rng.choice([1, 2, 3], n_days, p=[0.6, 0.3, 0.1])
    cnt = (
        6000
        - 15000 * (temp - 0.6) ** 2
        - 1500 * (weathersit == 3)
        + rng.normal(0, 200, n_days)
    )
    return pd.DataFrame(
        {
            "dteday": dates,
            "season": (dates.month % 12) // 3 + 1,
            "yr": (dates.year - 2011),
            "holiday": 0,
            "workingday": (dates.dayofweek < 5).astype(int),
            "weathersit": weathersit,
            "temp": temp,
            "hum": rng.uniform(0.3, 0.9, n_days),
            "windspeed": rng.uniform(0.05, 0.4, n_days),
            "casual": cnt * 0.2,
            "registered": cnt * 0.8,
            "cnt": cnt,
        }
    )


class TestRentalModel(unittest.TestCase):
    def setUp(self) -> None:
        self.df = make_synthetic_days()

    def test_features_exclude_leakage_columns(self) -> None:
        features = build_features(self.df)
        for col in LEAKAGE_COLUMNS + ["cnt"]:
            self.assertNotIn(col, features.columns)

    def test_features_have_fixed_columns_when_levels_missing(self) -> None:
        only_clear_days = self.df[self.df["weathersit"] == 1]
        self.assertListEqual(
            list(build_features(only_clear_days).columns),
            list(build_features(self.df).columns),
        )

    def test_time_split_is_chronological(self) -> None:
        train, test = time_split(self.df, "2012-07-01")
        self.assertEqual(len(train) + len(test), len(self.df))
        self.assertLess(train["dteday"].max(), test["dteday"].min())

    def test_linear_model_beats_baseline_and_finds_peak(self) -> None:
        results = train_and_evaluate(self.df, "2012-07-01")
        r2 = results.metrics["R2"]
        self.assertGreater(r2["Linear regression"], 0.8)
        self.assertGreater(r2["Linear regression"], r2["Baseline (train mean)"])
        self.assertAlmostEqual(results.peak_temp, 0.6, delta=0.05)
        self.assertEqual(results.permutation_importance.index[0], "temp")


if __name__ == "__main__":
    unittest.main()
