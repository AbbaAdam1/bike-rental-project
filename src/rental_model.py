from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


logger = logging.getLogger(__name__)

RANDOM_SEED = 42
DEFAULT_SPLIT_DATE = "2012-07-01"

# casual + registered == cnt, so they are never used as features (target leakage).
LEAKAGE_COLUMNS = ["casual", "registered"]

NUMERIC_FEATURES = ["temp", "hum", "windspeed", "yr", "holiday", "workingday"]
# Fixed levels keep train and test columns aligned even if a level is absent from one.
CATEGORY_LEVELS = {"season": [1, 2, 3, 4], "weathersit": [1, 2, 3, 4]}

# Max temperature the UCI dataset uses to normalize temp (temp = C / 41).
TEMP_MAX_C = 41.0


RAW_FEATURES = NUMERIC_FEATURES + list(CATEGORY_LEVELS)


@dataclass
class ModelResults:
    metrics: pd.DataFrame
    linear_coefficients: pd.Series
    permutation_importance: pd.Series
    peak_temp: float | None
    train_rows: int
    test_rows: int


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model feature matrix from a cleaned dataset."""
    X = df[NUMERIC_FEATURES].astype(float).copy()
    # Squared term lets the linear model capture the rise-then-fall with temperature.
    X["temp_sq"] = X["temp"] ** 2
    for col, levels in CATEGORY_LEVELS.items():
        # First level is the reference category.
        for level in levels[1:]:
            X[f"{col}_{level}"] = (df[col] == level).astype(float)
    return X


def time_split(
    df: pd.DataFrame, split_date: str = DEFAULT_SPLIT_DATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically so the model is tested on days after everything it trained on."""
    cutoff = pd.Timestamp(split_date)
    dates = pd.to_datetime(df["dteday"])
    return df.loc[dates < cutoff].copy(), df.loc[dates >= cutoff].copy()


def _score(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def train_and_evaluate(
    df: pd.DataFrame, split_date: str = DEFAULT_SPLIT_DATE
) -> ModelResults:
    """Fit a baseline, a linear model and a random forest, and score them on held-out days."""
    train_df, test_df = time_split(df, split_date)
    X_train, X_test = build_features(train_df), build_features(test_df)
    y_train, y_test = train_df["cnt"], test_df["cnt"]

    baseline_pred = np.full(len(y_test), y_train.mean())

    linear = LinearRegression().fit(X_train, y_train)
    # Kept for comparison: trees cannot predict above the training range, so they
    # underestimate the later, higher-demand test period.
    forest = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1
    ).fit(X_train, y_train)

    metrics = pd.DataFrame(
        {
            "Baseline (train mean)": _score(y_test, baseline_pred),
            "Linear regression": _score(y_test, linear.predict(X_test)),
            "Random forest": _score(y_test, forest.predict(X_test)),
        }
    ).T

    coefficients = pd.Series(linear.coef_, index=X_train.columns)

    # Vertex of the fitted parabola in temp: where predicted rentals peak.
    b1, b2 = coefficients["temp"], coefficients["temp_sq"]
    peak_temp = float(-b1 / (2 * b2)) if b2 < 0 else None

    importance_series = raw_feature_importance(linear, test_df)

    return ModelResults(
        metrics=metrics,
        linear_coefficients=coefficients,
        permutation_importance=importance_series,
        peak_temp=peak_temp,
        train_rows=len(train_df),
        test_rows=len(test_df),
    )


def raw_feature_importance(
    model: LinearRegression, test_df: pd.DataFrame, n_repeats: int = 10
) -> pd.Series:
    """Drop in held-out R2 when each raw input is shuffled.

    Shuffling raw columns (not encoded ones) keeps temp and temp_sq together and
    scores each categorical as a whole.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    y = test_df["cnt"]
    base_r2 = r2_score(y, model.predict(build_features(test_df)))

    drops = {}
    for col in RAW_FEATURES:
        # Shuffling a constant column changes nothing, so its score would be meaningless.
        if test_df[col].nunique() < 2:
            continue
        scores = []
        for _ in range(n_repeats):
            shuffled = test_df.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            scores.append(r2_score(y, model.predict(build_features(shuffled))))
        drops[col] = base_r2 - float(np.mean(scores))
    return pd.Series(drops).sort_values(ascending=False)


FEATURE_LABELS = {
    "temp": "Temperature",
    "season": "Season",
    "weathersit": "Weather (clear / mist / rain)",
    "windspeed": "Wind speed",
    "hum": "Humidity",
    "holiday": "Holiday",
    "workingday": "Working day",
    "yr": "Year",
}


def plot_feature_importance(importance: pd.Series, output_dir: Path) -> None:
    """Save a bar chart of how much held-out R2 drops when each input is shuffled."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Smallest first so the biggest driver ends up on top; R2 shown in points out of 100.
    points = (importance * 100).iloc[::-1]
    labels = [FEATURE_LABELS.get(name, name) for name in points.index]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(labels, points.values, color="#2a78d6", height=0.6)
    ax.bar_label(bars, labels=[f"{v:.0f}" if v >= 1 else "<1" for v in points.values], padding=4)

    fig.text(0.01, 0.98, "Temperature drives daily bike rentals the most", fontsize=14, va="top")
    fig.text(
        0.01, 0.92,
        "Points of prediction accuracy (R², out of 100) the model loses "
        "when each factor is scrambled",
        fontsize=10, color="#555555", va="top",
    )
    ax.set_xlabel("Accuracy lost (points)")
    ax.set_xlim(0, points.max() * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Leave room at the top for the title and subtitle.
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def report_results(results: ModelResults) -> None:
    logger.info(
        "Model trained on %d days, tested on %d later days:\n%s",
        results.train_rows, results.test_rows, results.metrics.round(3).to_string(),
    )
    logger.info(
        "Top drivers (drop in R² when shuffled):\n%s",
        results.permutation_importance.head(5).round(3).to_string(),
    )
    if results.peak_temp is not None:
        logger.info(
            "Linear model predicts rentals peak at normalized temp %.2f (~%.0f°C)",
            results.peak_temp, results.peak_temp * TEMP_MAX_C,
        )
