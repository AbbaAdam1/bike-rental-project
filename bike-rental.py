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

# Remove duplicate rows
initial_rows = len(df)
df = df.drop_duplicates()
duplicates_removed = initial_rows - len(df)
if duplicates_removed > 0:
    print(f"Removed {duplicates_removed} duplicate rows.")

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

# Drop invalid rows in normalized columns (temp, atemp, hum)
for col in ['temp', 'atemp', 'hum']:
    if col in df:
        invalid_rows = ((df[col] < 0) | (df[col] > 1)).sum()
        if invalid_rows > 0:
            df = df[(df[col] >= 0) & (df[col] <= 1)]
            print(f"Dropped {invalid_rows} invalid rows in {col} outside [0, 1].")

# Clip outliers in cnt only (valid range 0–10000)
if 'cnt' in df:
    outliers_before = ((df['cnt'] < 0) | (df['cnt'] > 10000)).sum()
    df['cnt'] = df['cnt'].clip(0, 10000)
    if outliers_before > 0:
        print(f"Clipped {outliers_before} outliers in cnt to range [0, 10000].")

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

# 1. Focused Correlation with Target Variable
# Select only key variables that support your weather story
key_features = ['temp', 'hum', 'windspeed', 'weathersit', 'season', 'is_weekend', 'cnt']
corr_with_target = df[key_features].corr()['cnt'].drop('cnt').sort_values(key=abs, ascending=False)

plt.figure(figsize=(8, 6))
colors = ['darkred' if x > 0 else 'darkblue' for x in corr_with_target.values]
bars = plt.barh(range(len(corr_with_target)), corr_with_target.values, color=colors, alpha=0.7)

# Add correlation values on bars
for i, (bar, val) in enumerate(zip(bars, corr_with_target.values)):
    plt.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.2f}',
             va='center', ha='left' if val > 0 else 'right', fontweight='bold')

plt.yticks(range(len(corr_with_target)), corr_with_target.index)
plt.xlabel('Correlation with Bike Rentals')
plt.title('Key Feature Correlations with Daily Bike Rentals')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('correlation_focused.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Scatter Plot: cnt vs. temp by season (focused on Goldilocks Zone)
plt.figure(figsize=(10, 6))

# Create the scatter plot with just season coloring
sns.scatterplot(x='temp', y='cnt', hue='season', data=df, alpha=0.7, s=60)

# Add vertical lines to highlight the Goldilocks Zone (0.5-0.7)
plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Optimal Zone')
plt.axvline(x=0.7, color='red', linestyle='--', alpha=0.8, linewidth=2)

# Add shaded region for Goldilocks Zone
plt.axvspan(0.5, 0.7, alpha=0.2, color='green', label='Goldilocks Zone')

# Customize the plot
plt.title('Bike Rentals vs. Temperature by Season\n(Optimal Zone: 0.5-0.7 Normalized Temp)', fontsize=14)
plt.xlabel('Normalized Temperature (0-1 scale)', fontsize=12)
plt.ylabel('Daily Rental Count', fontsize=12)

# Improve legend
season_labels = ['Winter', 'Spring', 'Summer', 'Fall']
handles, labels = plt.gca().get_legend_handles_labels()
# Update season labels in legend
for i, label in enumerate(labels[:4]):  # First 4 are seasons
    labels[i] = season_labels[int(label)-1]
plt.legend(handles, labels, title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('rentals_vs_temp.png', dpi=300, bbox_inches='tight')
plt.show()

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
