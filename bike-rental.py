import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the corrupted dataset (734 rows: 731 original + 3 duplicates) from 'day_corrupted.csv'
df = pd.read_csv('day_corrupted.csv')
# Dataset loaded with corruptions: 5 NaNs in temp/hum, 5 NaNs + 5 None in windspeed, strings ("high", "32°C", etc.), outliers (1000, -100), duplicates, shuffled rows

# Print initial state to inspect rows, columns, duplicates, and missing values
print("Initial State:")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Missing values:\n{df.isna().sum()}")
# Shows 734 rows, 3 duplicates, 5 NaNs in temp/hum, 10 in windspeed (5 NaNs + 5 None)

# Clean column names by removing spaces/special characters and mapping to UCI standard
original_columns = df.columns
df.columns = df.columns.str.strip().str.lower().str.replace('[^a-z0-9]', '_', regex=True)
column_map = {
    'date_of_record': 'dteday', 'season': 'season', 'year': 'yr', 'mnth': 'mnth',
    'holiday_': 'holiday', 'week_day': 'weekday', 'working_day': 'workingday',
    'weather_situation': 'weathersit', 'temperature': 'temp', 'feels_like_temp': 'atemp',
    'humidity': 'hum', 'wind_speed': 'windspeed', 'casual_users': 'casual',
    'registered_users': 'registered', 'total_count': 'cnt', 'instant': 'instant'
}
df.columns = [column_map.get(col, col) for col in df.columns]
if list(original_columns) != list(df.columns):
    print(f"Fixed column names: {list(original_columns)} → {list(df.columns)}")
else:
    print("No column name changes needed.")
# Ensures columns match UCI dataset (e.g., 'total_count' → 'cnt')

# Remove 3 duplicate rows to restore 731 unique records
initial_rows = len(df)
df = df.drop_duplicates()
duplicates_removed = initial_rows - len(df)
if duplicates_removed > 0:
    print(f"Removed {duplicates_removed} duplicate rows.")
# Reduces rows from 734 to 731

# Fix non-sequential instant by sorting by dteday and resetting to 1–731
if not df['instant'].is_monotonic_increasing:
    df['dteday'] = pd.to_datetime(df['dteday'], errors='coerce', format='mixed')
    df = df.sort_values('dteday').reset_index(drop=True)
    df['instant'] = df.index + 1
    print("Sorted by dteday and reset instant to sequential (1 to n).")
else:
    print("Instant column already sequential.")
# Restores chronological order; instant set to 1, 2, ..., 731

# Clean numeric columns by explicitly handling invalid strings and None
def clean_numeric(col):
    # Convert to string, replace None with 'None' string for handling
    col_str = col.fillna('None').astype(str)
    # Count specific corrupted values: 'high', 'low', units, or 'None'
    corrupted_patterns = [r'high', r'low', r'°C', r'%', r'km/h', r'None']
    initial_corrupted = col_str.str.contains('|'.join(corrupted_patterns), case=False, na=False).sum()
    # Remove invalid patterns, keeping valid numeric strings (e.g., '0.5')
    col_clean = col_str.str.replace(r'°C|%|km/h|high|low|None', '', regex=True).str.strip()
    # Convert to numeric, invalid values become NaN
    cleaned_col = pd.to_numeric(col_clean, errors='coerce')
    return cleaned_col, initial_corrupted
numeric_cols = ['temp', 'hum', 'windspeed', 'cnt']
for col in numeric_cols:
    if col in df:
        df[col], cleaned = clean_numeric(df[col])
        if cleaned > 0:
            print(f"Cleaned {cleaned} non-numeric values in {col} (e.g., 'high', '32°C').")
# Converts 'high', '32°C', 'None', etc. to NaN; expects 2 in temp ('32°C', '27C'), 5 in hum (3 'low', 2 '%'), 6 in windspeed (5 'None', 1 '10km/h'), 2 in cnt ('high')

# Fill missing values in numeric columns with median to maintain distribution
numeric_cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
for col in numeric_cols:
    if col in df:
        missing_before = df[col].isna().sum()
        if missing_before > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"Filled {missing_before} missing values in {col} with median.")
# Fills 7 in temp (5 NaNs + 2 strings), 8 in hum (5 + 3), 11 in windspeed (5 NaNs + 5 None + 1 string), 2 in cnt (2 'high')

# Fill missing categorical values with mode (if any)
categorical_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
for col in categorical_cols:
    if col in df:
        missing_before = df[col].isna().sum()
        if missing_before > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"Filled {missing_before} missing values in {col} with mode.")
# Ensures no missing values in categorical columns (likely none in your dataset)

# Clip outliers in normalized columns (temp, atemp, hum) to [0, 1] and cnt to [0, 10000]
for col in ['temp', 'atemp', 'hum']:
    if col in df:
        outliers_before = ((df[col] < 0) | (df[col] > 1)).sum()
        df[col] = df[col].clip(0, 1)
        if outliers_before > 0:
            print(f"Clipped {outliers_before} outliers in {col} to range [0, 1].")
if 'cnt' in df:
    outliers_before = ((df['cnt'] < 0) | (df['cnt'] > 10000)).sum()
    df['cnt'] = df['cnt'].clip(0, 10000)
    if outliers_before > 0:
        print(f"Clipped {outliers_before} outliers in cnt to range [0, 10000].")
# Clips 5 outliers in temp (1000, -100, 200, 300, 999); likely 0 in hum/cnt

# Ensure categorical columns are integers
for col in ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']:
    if col in df:
        non_int_before = df[col].apply(lambda x: not isinstance(x, int) and pd.notna(x)).sum()
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mode()[0]).astype(int)
        if non_int_before > 0:
            print(f"Converted {non_int_before} non-integer values in {col} to integers.")

#Ensures season, yr, etc., are integers (likely none needed)
df['dteday'] = pd.to_datetime(df['dteday'])
df['day'] = df['dteday'].dt.day
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year

# Create is_weekend (1 if Saturday/Sunday, 0 otherwise)
df['is_weekend'] = df['weekday'].isin([0, 6]).astype(int)

# Create interaction feature for temperature and humidity
df['temp_hum_interaction'] = df['temp'] * df['hum']

# 1. Correlation Heatmap (all numeric columns, but focus on predictors)
predictors = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
              'temp', 'atemp', 'hum', 'windspeed', 'is_weekend']
corr_matrix = df[predictors + ['cnt']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
            annot_kws={"size": 10})
plt.title('Correlation of Key Features with Bike Rentals')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
# Note: Includes all listed predictors and cnt; excludes instant, casual, registered

# 2. Scatter Plot: cnt vs. temp by season
plt.figure(figsize=(8, 6))
sns.scatterplot(x='temp', y='cnt', hue='season', size='hum', data=df)
plt.title('Bike Rentals vs. Temperature by Season')
plt.xlabel('Normalized Temperature')
plt.ylabel('Rental Count')
plt.tight_layout()
plt.savefig('rentals_vs_temp.png')
plt.show()
# Plots only temp, cnt, season, and hum (size); no other columns

# 3. Boxplot: cnt by season
plt.figure(figsize=(8, 6))
sns.boxplot(x='season', y='cnt', data=df)
plt.title('Bike Rentals by Season')
plt.xlabel('Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)')
plt.ylabel('Rental Count')
plt.tight_layout()
plt.savefig('rentals_by_season.png')
plt.show()
# Plots only season and cnt

# 4. Boxplot: cnt by weathersit
plt.figure(figsize=(8, 6))
sns.boxplot(x='weathersit', y='cnt', data=df)
plt.title('Bike Rentals by Weather Situation')
plt.xlabel('Weather (1=Clear, 2=Mist, 3=Light Rain)')
plt.ylabel('Rental Count')
plt.tight_layout()
plt.savefig('rentals_by_weather.png')
plt.show()
# Plots only weathersit and cnt

# 5. Line Plot: cnt vs. weekday (using is_weekend for clarity)
plt.figure(figsize=(8, 6))
sns.lineplot(x='weekday', y='cnt', hue='is_weekend', data=df)
plt.title('Bike Rentals by Weekday (Hue: Weekend vs Weekday)')
plt.xlabel('Weekday (0=Sun, 6=Sat)')
plt.ylabel('Rental Count')
plt.tight_layout()
plt.savefig('rentals_by_weekday.png')
plt.show()
# Plots only weekday, cnt, and is_weekend
# Print final state to confirm cleaning
print("\nAfter Cleaning:")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Missing values:\n{df.isna().sum()}")
print(f"Instant sequential: {df['instant'].is_monotonic_increasing}")
print(f"First 5 instant: {df['instant'].head().tolist()}")
print(f"Last 5 instant: {df['instant'].tail().tolist()}")
# Confirms 731 rows, no duplicates, no missing values, sequential instant

# Save cleaned dataset for EDA and modeling
df.to_csv('day_cleaned.csv', index=False)
print("✅ Data cleaned and saved as 'day_cleaned.csv'")
# Dataset ready for Bike Rental Prediction Project