import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('day.csv')

# Set random seed for reproducibility
np.random.seed(42)

# 1. Missing Values: Set 5% of values in specified columns to NaN
for col in ['hum', 'windspeed', 'casual', 'registered']:
    mask = np.random.rand(len(df)) < 0.05
    df.loc[mask, col] = np.nan

# 2. Inconsistent Data Types: Convert 5% of temp and atemp to strings
for col in ['temp', 'atemp']:
    mask = np.random.rand(len(df)) < 0.05
    df.loc[mask, col] = df.loc[mask, col].astype(str)

# 3. Duplicate Rows: Duplicate 10 random rows
duplicates = df.sample(10, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)

# 4. Inconsistent Formatting
# Date: Convert 10% of dteday to MM/DD/YYYY
mask_date = np.random.rand(len(df)) < 0.1
df.loc[mask_date, 'dteday'] = pd.to_datetime(df.loc[mask_date, 'dteday']).dt.strftime('%m/%d/%Y')

# Season: Replace 10% of season with text labels
season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
mask_season = np.random.rand(len(df)) < 0.1
df.loc[mask_season, 'season'] = df.loc[mask_season, 'season'].map(season_map)

# Weather: Replace 10% of weathersit with text labels
weather_map = {1: 'Clear', 2: 'Mist', 3: 'Light Rain', 4: 'Heavy Rain'}
mask_weather = np.random.rand(len(df)) < 0.1
df.loc[mask_weather, 'weathersit'] = df.loc[mask_weather, 'weathersit'].map(weather_map)

# 5. Outliers: Introduce outliers in 5% of temp, hum, and cnt
mask_outlier = np.random.rand(len(df)) < 0.05
df.loc[mask_outlier, 'temp'] = np.random.uniform(2, 10, sum(mask_outlier))
df.loc[mask_outlier, 'hum'] = np.random.uniform(2, 5, sum(mask_outlier))
df.loc[mask_outlier, 'cnt'] = np.random.randint(10000, 20000, sum(mask_outlier))

# 6. Problematic Column Names
df.columns = [
    'instant', 'Date of Record', 'Season', 'Year', 'Mnth', 'Holiday?', 'Week Day',
    'Working Day', 'Weather Situation', 'Temperature', 'Feels Like Temp',
    'Humidity', 'Wind Speed', 'Casual Users', 'Registered Users', 'Total Count'
]

# 7. Inconsistent Categorical Data: Replace 10% of holiday and workingday with text
mask_cat = np.random.rand(len(df)) < 0.1
df.loc[mask_cat, 'Holiday?'] = df.loc[mask_cat, 'Holiday?'].map({0: 'No', 1: 'Yes'})
df.loc[mask_cat, 'Working Day'] = df.loc[mask_cat, 'Working Day'].map({0: 'No', 1: 'Yes'})

# 8. Invalid Strings in Numeric Column
mask_invalid = np.random.rand(len(df)) < 0.02
df.loc[mask_invalid, 'Total Count'] = '10000–'

# Save the refactored dataset
df.to_csv('day_refactored.csv', index=False)
print("Refactored dataset saved as 'day_refactored.csv' with 741 rows (731 original + 10 duplicates)")