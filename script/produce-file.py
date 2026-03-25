import pandas as pd
import numpy as np
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def corrupt_data(input_path=None, output_path=None, seed=42):
    if input_path is None:
        input_path = PROJECT_ROOT / "day.csv"
    else:
        input_path = Path(input_path)

    if output_path is None:
        output_path = PROJECT_ROOT / "day_corrupted.csv"
    else:
        output_path = Path(output_path)

    rng = np.random.default_rng(seed)
    random.seed(seed)
    df = pd.read_csv(input_path)

    num_rows = len(df)

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
    duplicated_rows = df.sample(3)
    df = pd.concat([df, duplicated_rows], ignore_index=True)

    # Shuffle the rows
    df = df.sample(frac=1).reset_index(drop=True)

    # Save the corrupted dataset
    df.to_csv(output_path, index=False)
    print(f"[OK] Corrupted dataset saved to '{output_path}'")


if __name__ == "__main__":
    corrupt_data()
