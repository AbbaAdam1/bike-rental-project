import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('day.csv')

# Diagnostic Check: Inspect instant column and dataset state
print("Initial Dataset State:")
print(f"Number of rows: {len(df)}")
print(f"Unique instant values: {df['instant'].nunique()}")
print(f"Duplicate instants: {df['instant'].duplicated().sum()}")
print(f"Is instant sequential? {df['instant'].is_monotonic_increasing}")

# 1. Fix Column Names
# Check if problematic column names exist
expected_columns = [
    'instant', 'Date of Record', 'Season', 'Year', 'Mnth', 'Holiday?', 'Week Day',
    'Working Day', 'Weather Situation', 'Temperature', 'Feels Like Temp',
    'Humidity', 'Wind Speed', 'Casual Users', 'Registered Users', 'Total Count'
]
if set(df.columns).intersection(expected_columns):
    df.columns = [
        'instant', 'dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday',
        'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
        'casual', 'registered', 'cnt'
    ]
else:
    # Assume original column names if not refactored
    df.columns = [
        'instant', 'dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday',
        'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
        'casual', 'registered', 'cnt'
    ]

# 2. Remove Duplicate Rows
df.drop_duplicates(inplace=True)

# 3. Fix instant Column: Reset to sequential values (1 to n)
df.reset_index(drop=True, inplace=True)
df['instant'] = df.index + 1

# 4. Fix Missing Values
for col in ['hum', 'windspeed', 'casual', 'registered']:
    if df[col].isna().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# 5. Fix Inconsistent Data Types
for col in ['temp', 'atemp', 'cnt']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 6. Fix Inconsistent Formatting
# Date
df['dteday'] = pd.to_datetime(df['dteday'], errors='coerce', format='mixed')

# Season
df['season'] = df['season'].replace({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4})
df['season'] = df['season'].astype(int)

# Weather Situation
df['weathersit'] = df['weathersit'].replace({'Clear': 1, 'Mist': 2, 'Light Rain': 3, 'Heavy Rain': 4})
df['weathersit'] = df['weathersit'].astype(int)

# 7. Fix Outliers
df['temp'] = df['temp'].clip(0, 1)
df['hum'] = df['hum'].clip(0, 1)
df['cnt'] = df['cnt'].clip(0, 10000)

# 8. Fix Inconsistent Categorical Data
df['holiday'] = df['holiday'].replace({'No': 0, 'Yes': 1}).astype(int)
df['workingday'] = df['workingday'].replace({'No': 0, 'Yes': 1}).astype(int)

# Diagnostic Check: Verify final state
print("\nAfter Cleaning:")
print(f"Number of rows: {len(df)}")
print(f"Unique instant values: {df['instant'].nunique()}")
print(f"Duplicate instants: {df['instant'].duplicated().sum()}")
print(f"Is instant sequential? {df['instant'].is_monotonic_increasing}")
print(f"First few instant values: {df['instant'].head().tolist()}")
print(f"Last few instant values: {df['instant'].tail().tolist()}")

# Save cleaned dataset
df.to_csv('day_cleaned.csv', index=False)
print("Cleaned dataset saved as 'day_cleaned.csv'")