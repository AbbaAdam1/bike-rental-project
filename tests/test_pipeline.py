import unittest

import pandas as pd

from src.bike_rental_pipeline import clean_dataset, engineer_features


class TestBikeRentalPipeline(unittest.TestCase):
    def setUp(self) -> None:
        # Includes duplicate row, string pollution, and missing values.
        self.sample = pd.DataFrame(
            {
                "instant": [3, 1, 2, 2],
                "date_of_record": ["2011-01-03", "2011-01-01", "2011-01-02", "2011-01-02"],
                "season": [1, 1, 1, 1],
                "yr": [0, 0, 0, 0],
                "mnth": [1, 1, 1, 1],
                "holiday": [0, 0, 0, 0],
                "week_day": [1, 6, 0, 0],
                "working_day": [1, 0, 0, 0],
                "weather_situation": [1, 1, 2, 2],
                "temperature": ["0.3", "32°C", "0.5", "0.5"],
                "feels_like_temp": [0.31, 0.29, 0.52, 0.52],
                "humidity": ["low", "0.45", None, None],
                "wind_speed": ["10km/h", None, 0.2, 0.2],
                "casual_users": [100, 120, 130, 130],
                "registered_users": [400, 450, 460, 460],
                "total_count": [500, "high", 590, 590],
            }
        )

    def test_clean_dataset_removes_duplicates_and_fixes_fields(self) -> None:
        cleaned, report = clean_dataset(self.sample)

        self.assertEqual(report["duplicates_removed"], 1)
        self.assertEqual(cleaned["instant"].tolist(), [1, 2, 3])
        self.assertFalse(cleaned.duplicated().any())

        for col in ["temp", "atemp", "hum", "windspeed", "cnt"]:
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned[col]))
            self.assertFalse(cleaned[col].isna().any())

    def test_unit_suffixed_values_are_converted_to_normalized_scale(self) -> None:
        cleaned, _ = clean_dataset(self.sample)
        jan_1 = cleaned.loc[cleaned["dteday"] == "2011-01-01"].iloc[0]
        jan_3 = cleaned.loc[cleaned["dteday"] == "2011-01-03"].iloc[0]

        self.assertAlmostEqual(jan_1["temp"], 32 / 41)
        self.assertAlmostEqual(jan_3["windspeed"], 10 / 67)

    def test_normalized_columns_stay_in_range(self) -> None:
        cleaned, _ = clean_dataset(self.sample)

        for col in ["temp", "atemp", "hum", "windspeed"]:
            self.assertTrue(cleaned[col].between(0, 1).all(), col)

    def test_missing_cnt_rebuilt_from_casual_and_registered(self) -> None:
        cleaned, report = clean_dataset(self.sample)
        jan_1 = cleaned.loc[cleaned["dteday"] == "2011-01-01"].iloc[0]

        self.assertEqual(jan_1["cnt"], 120 + 450)
        self.assertEqual(report["cnt_rebuilt_from_components"], 1)

    def test_out_of_range_values_do_not_skew_fill_median(self) -> None:
        sample = self.sample.copy()
        sample["humidity"] = ["0.4", "0.6", None, None]
        sample.loc[len(sample)] = sample.iloc[0]
        sample.loc[len(sample) - 1, ["date_of_record", "humidity"]] = ["2011-01-04", "9.0"]

        cleaned, report = clean_dataset(sample)
        jan_2 = cleaned.loc[cleaned["dteday"] == "2011-01-02"].iloc[0]

        self.assertEqual(report["rows_dropped_normalized_out_of_range"]["hum"], 1)
        self.assertAlmostEqual(jan_2["hum"], 0.5)

    def test_engineer_features_adds_expected_columns(self) -> None:
        cleaned, _ = clean_dataset(self.sample)
        featured = engineer_features(cleaned)

        for col in ["day", "month", "year", "is_weekend", "temp_hum_interaction"]:
            self.assertIn(col, featured.columns)


if __name__ == "__main__":
    unittest.main()
