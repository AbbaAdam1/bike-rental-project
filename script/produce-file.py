import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "day.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "day_corrupted.csv"

logger = logging.getLogger(__name__)


def corrupt_data(input_path=None, output_path=None, seed=42):
    input_path = Path(input_path) if input_path is not None else DEFAULT_INPUT_PATH
    output_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_PATH

    rng = np.random.default_rng(seed)
    df = pd.read_csv(input_path)

    num_rows = len(df)

    # Allow string values to be injected into numeric columns.
    corrupted_cols = ['temp', 'hum', 'windspeed', 'cnt']
    df[corrupted_cols] = df[corrupted_cols].astype(object)

    # 1. Random missing values
    for col in ['temp', 'hum', 'windspeed']:
        missing_indices = rng.choice(df.index, size=5, replace=False)
        df.loc[missing_indices, col] = np.nan

    # 2. String injection in numeric fields
    string_injection_indices = rng.choice(df.index, size=5, replace=False)
    df.loc[string_injection_indices[:2], 'cnt'] = 'high'
    df.loc[string_injection_indices[2:], 'hum'] = 'low'

    # 3. Outliers
    #outlier_indices = np.random.choice(df.index, size=5, replace=False)
    #df.loc[outlier_indices, 'temp'] = [1000, -100, 200, 300, 999]

    # 4. Bad formatting
    bad_format_indices = rng.choice(df.index, size=5, replace=False)
    df.loc[bad_format_indices[:2], 'temp'] = ['32°C', '27C']
    df.loc[bad_format_indices[2:4], 'hum'] = ['50%', '75%']
    df.loc[bad_format_indices[4:], 'windspeed'] = ['10km/h']

    # 5. Drop windspeed for some rows
    drop_indices = rng.choice(df.index, size=5, replace=False)
    df.loc[drop_indices, 'windspeed'] = None

    # 6. Duplicated rows
    duplicated_rows = df.sample(3, random_state=seed)
    df = pd.concat([df, duplicated_rows], ignore_index=True)

    # Shuffle the rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save the corrupted dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Corrupted dataset saved to %s", output_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create a deliberately corrupted copy of the bike-rental dataset."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_PATH,
        help="Clean source CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help="Where to write the corrupted CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed; the same seed always produces the same file (default: %(default)s)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    corrupt_data(args.input, args.output, args.seed)
