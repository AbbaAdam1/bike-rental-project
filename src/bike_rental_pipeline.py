from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

# Charts are only saved to files, so use the non-interactive backend (works without a display).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.rental_model import plot_feature_importance, report_results, train_and_evaluate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_INPUT_PATH = DEFAULT_OUTPUT_DIR / "day_corrupted.csv"

logger = logging.getLogger(__name__)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and map column names to the UCI bike-sharing schema."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace("[^a-z0-9]", "_", regex=True)
    )
    column_map = {
        "date_of_record": "dteday",
        "holiday_": "holiday",
        "week_day": "weekday",
        "working_day": "workingday",
        "weather_situation": "weathersit",
        "temperature": "temp",
        "feels_like_temp": "atemp",
        "humidity": "hum",
        "wind_speed": "windspeed",
        "casual_users": "casual",
        "registered_users": "registered",
        "total_count": "cnt",
    }
    df.columns = [column_map.get(col, col) for col in df.columns]
    return df


# Divisors that map raw units onto the UCI dataset's normalized 0-1 scale.
# temp = C / 41, hum = % / 100, windspeed = km/h / 67 (per the UCI readme).
UNIT_DIVISORS = {
    "temp": {"°C": 41.0, "C": 41.0},
    "hum": {"%": 100.0},
    "windspeed": {"km/h": 67.0},
}

NORMALIZED_COLUMNS = ["temp", "atemp", "hum", "windspeed"]

# UCI season codes; 1 starts at the winter solstice, so January is season 1.
SEASON_NAMES = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
# Distinct hues checked for colorblind separation; one color per season.
SEASON_COLORS = {"Winter": "#2a78d6", "Spring": "#1baf7a", "Summer": "#eb6834", "Fall": "#4a3aa7"}


def _clean_numeric_series(
    series: pd.Series, unit_divisors: dict[str, float] | None = None
) -> tuple[pd.Series, int]:
    """Coerce values to numeric, converting unit-suffixed values to the normalized scale.

    Values that cannot be parsed (e.g. "high", "low") become NaN.
    """
    unit_divisors = unit_divisors or {}
    series_as_str = series.astype("string").str.strip()
    numeric = pd.to_numeric(series_as_str, errors="coerce")

    # Anything present but not plain-numeric counts as a corrupted token.
    corrupted_mask = series_as_str.notna() & numeric.isna()

    # Longest suffix first so "°C" is matched before "C".
    for unit in sorted(unit_divisors, key=len, reverse=True):
        has_unit = corrupted_mask & numeric.isna() & series_as_str.str.endswith(unit)
        values = pd.to_numeric(
            series_as_str[has_unit].str[: -len(unit)].str.strip(), errors="coerce"
        )
        numeric.loc[has_unit] = values / unit_divisors[unit]

    return numeric.astype(float), int(corrupted_mask.sum())


def clean_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean a corrupted bike-rental dataset and return quality metrics."""
    df = standardize_column_names(raw_df)
    report: dict[str, Any] = {
        "initial_rows": len(df),
        "duplicates_removed": 0,
        "rows_dropped_invalid_dates": 0,
        "rows_dropped_normalized_out_of_range": {},
        "corrupted_tokens_cleaned": {},
        "cnt_rebuilt_from_components": 0,
        "missing_filled": {},
        "cnt_outliers_clipped": 0,
        "final_rows": 0,
    }

    before_dedup = len(df)
    df = df.drop_duplicates().copy()
    report["duplicates_removed"] = before_dedup - len(df)

    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")
        invalid_dates = int(df["dteday"].isna().sum())
        if invalid_dates:
            df = df.dropna(subset=["dteday"]).copy()
        report["rows_dropped_invalid_dates"] = invalid_dates

    numeric_to_clean = ["temp", "hum", "windspeed", "cnt"]
    for col in numeric_to_clean:
        if col in df.columns:
            df[col], cleaned_count = _clean_numeric_series(
                df[col], UNIT_DIVISORS.get(col)
            )
            report["corrupted_tokens_cleaned"][col] = cleaned_count

    # Drop out-of-range rows before filling, so bad values don't skew the medians.
    for col in NORMALIZED_COLUMNS:
        if col in df.columns:
            invalid_mask = (df[col] < 0) | (df[col] > 1)
            dropped = int(invalid_mask.sum())
            if dropped:
                df = df.loc[~invalid_mask].copy()
            report["rows_dropped_normalized_out_of_range"][col] = dropped

    # cnt is defined as casual + registered, so rebuild it exactly where possible.
    if {"cnt", "casual", "registered"}.issubset(df.columns):
        rebuildable = df["cnt"].isna() & df["casual"].notna() & df["registered"].notna()
        df.loc[rebuildable, "cnt"] = df.loc[rebuildable, "casual"] + df.loc[rebuildable, "registered"]
        report["cnt_rebuilt_from_components"] = int(rebuildable.sum())

    numeric_fill = ["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
    for col in numeric_fill:
        if col in df.columns:
            missing = int(df[col].isna().sum())
            if missing > 0:
                df[col] = df[col].fillna(df[col].median())
            report["missing_filled"][col] = missing

    categorical_fill = [
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]
    for col in categorical_fill:
        if col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    if "cnt" in df.columns:
        outliers = int(((df["cnt"] < 0) | (df["cnt"] > 10000)).sum())
        df["cnt"] = df["cnt"].clip(0, 10000)
        report["cnt_outliers_clipped"] = outliers

    int_cols = [
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mode().iloc[0]).astype(int)

    if "dteday" in df.columns:
        df = df.sort_values("dteday").reset_index(drop=True)

    if "instant" in df.columns:
        df["instant"] = range(1, len(df) + 1)

    report["final_rows"] = len(df)
    return df, report


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used for analysis and plotting."""
    df = df.copy()
    if "dteday" in df.columns:
        df["day"] = df["dteday"].dt.day
        df["month"] = df["dteday"].dt.month
        df["year"] = df["dteday"].dt.year

    if "weekday" in df.columns:
        # UCI weekday values use 0 for Sunday and 6 for Saturday.
        df["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)

    if {"temp", "hum"}.issubset(df.columns):
        df["temp_hum_interaction"] = df["temp"] * df["hum"]

    return df


def create_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate project visualizations and save them in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    key_features = ["temp", "hum", "windspeed", "weathersit", "season", "is_weekend", "cnt"]
    available_features = [c for c in key_features if c in df.columns]

    if "cnt" in available_features:
        corr_with_target = (
            df[available_features].corr(numeric_only=True)["cnt"].drop("cnt").sort_values(
                key=abs, ascending=False
            )
        )

        plt.figure(figsize=(8, 6))
        colors = ["darkred" if x > 0 else "darkblue" for x in corr_with_target.values]
        bars = plt.barh(range(len(corr_with_target)), corr_with_target.values, color=colors, alpha=0.7)

        for i, (bar, val) in enumerate(zip(bars, corr_with_target.values)):
            x_pos = val + 0.01 if val > 0 else val - 0.01
            align = "left" if val > 0 else "right"
            plt.text(x_pos, i, f"{val:.2f}", va="center", ha=align, fontweight="bold")

        plt.yticks(range(len(corr_with_target)), corr_with_target.index)
        plt.xlabel("Correlation with Bike Rentals")
        plt.title("Key Feature Correlations with Daily Bike Rentals")
        plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_focused.png", dpi=300, bbox_inches="tight")
        plt.close()

    if {"temp", "cnt", "season"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        # Neutral band and lines so only the seasons carry color.
        plt.axvspan(0.5, 0.7, color="#e6e6e6", zorder=0, label="Goldilocks Zone")
        plt.axvline(x=0.5, color="#555555", linestyle="--", linewidth=1.5, zorder=1)
        plt.axvline(x=0.7, color="#555555", linestyle="--", linewidth=1.5, zorder=1)

        sns.scatterplot(
            x=df["temp"],
            y=df["cnt"],
            hue=df["season"].map(SEASON_NAMES),
            hue_order=list(SEASON_COLORS),
            palette=SEASON_COLORS,
            alpha=0.75,
            s=60,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

        plt.title("Bike Rentals vs. Temperature by Season\n(Optimal Zone: 0.5-0.7 Normalized Temp)")
        plt.xlabel("Normalized Temperature (0-1 scale)")
        plt.ylabel("Daily Rental Count")
        # List the seasons first and the shaded zone last.
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [labels.index(name) for name in SEASON_COLORS] + [labels.index("Goldilocks Zone")]
        plt.legend(
            [handles[i] for i in order], [labels[i] for i in order],
            title="Season", bbox_to_anchor=(1.05, 1), loc="upper left",
        )
        plt.tight_layout()
        plt.savefig(output_dir / "rentals_vs_temp.png", dpi=300, bbox_inches="tight")
        plt.close()

    if {"season", "cnt"}.issubset(df.columns):
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="season", y="cnt", data=df)
        plt.title("Bike Rentals by Season")
        plt.xlabel("Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)")
        plt.ylabel("Rental Count")
        plt.tight_layout()
        plt.savefig(output_dir / "rentals_by_season.png", dpi=300, bbox_inches="tight")
        plt.close()

    if {"weathersit", "cnt"}.issubset(df.columns):
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="weathersit", y="cnt", data=df)
        plt.title("Bike Rentals by Weather Situation")
        plt.xlabel("Weather (1=Clear, 2=Mist, 3=Light Rain, 4=Heavy Rain)")
        plt.ylabel("Rental Count")
        plt.tight_layout()
        plt.savefig(output_dir / "rentals_by_weather.png", dpi=300, bbox_inches="tight")
        plt.close()


def run_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean, engineer features, plot, and write the cleaned data and quality report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(input_path)
    cleaned_df, report = clean_dataset(raw_df)
    modeled_df = engineer_features(cleaned_df)

    create_plots(modeled_df, output_dir / "images")
    modeled_df.to_csv(output_dir / "day_cleaned.csv", index=False)
    (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2))
    return modeled_df, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the corrupted bike-rental dataset, plot it, and model daily rentals."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_PATH,
        help="Corrupted CSV to clean (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Where to write the cleaned CSV, quality report, and charts (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # matplotlib logs routine plotting details at INFO; keep the output to pipeline messages.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if not args.input.exists():
        raise SystemExit(
            f"Input file not found: {args.input}\n"
            "Generate it first with: python script/produce-file.py"
        )

    logger.info("Cleaning %s", args.input)
    modeled_df, report = run_pipeline(args.input, args.output_dir)
    logger.info("Rows: %d before cleaning, %d after", report["initial_rows"], report["final_rows"])
    logger.info("Duplicates removed: %d", report["duplicates_removed"])
    logger.info("Missing values filled: %s", report["missing_filled"])
    logger.info("Corrupted tokens cleaned: %s", report["corrupted_tokens_cleaned"])
    logger.info("Out-of-range rows dropped: %s", report["rows_dropped_normalized_out_of_range"])
    logger.info("cnt rebuilt from casual + registered: %d", report["cnt_rebuilt_from_components"])
    logger.info("Wrote cleaned data, quality report, and charts to %s", args.output_dir)

    results = train_and_evaluate(modeled_df)
    plot_feature_importance(results.permutation_importance, args.output_dir / "images")
    report_results(results)


if __name__ == "__main__":
    main()
