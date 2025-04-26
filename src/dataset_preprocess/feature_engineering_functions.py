import pandas as pd
import numpy as np

def add_min_max_quantile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add min, max, and quantile-based features for each 14-day climate variable."""
    categories = ['DayAirTmpAvg', 'DayPrecip', 'DayRelHumAvg', 'DaySoilTmpAvg', 'DayWindSpdAvg']
    
    for cat in categories:
        cols = [f'{cat}{str(i).zfill(2)}' for i in range(1, 15)]
        df[f'{cat}_min14'] = df[cols].min(axis=1)
        df[f'{cat}_max14'] = df[cols].max(axis=1)
        df[f'{cat}_q25'] = df[cols].quantile(0.25, axis=1)
        df[f'{cat}_q75'] = df[cols].quantile(0.75, axis=1)
        df[f'{cat}_iqr'] = df[f'{cat}_q75'] - df[f'{cat}_q25']  # Interquartile range
    return df

def add_trend_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend and differential stats (first-order differences and slope) to capture volatility."""
    categories = ['DayAirTmpAvg', 'DayPrecip', 'DayRelHumAvg', 'DaySoilTmpAvg', 'DayWindSpdAvg']

    for cat in categories:
        cols = [f'{cat}{str(i).zfill(2)}' for i in range(1, 15)]
        # First-order difference (last - first)
        df[f'{cat}_diff14'] = df[cols[-1]] - df[cols[0]]
        # Linear slope: regression of values across the 14 days
        X = np.arange(14).reshape(-1, 1)  # [0,1,2,...,13]
        slopes = []
        for _, row in df[cols].iterrows():
            y = row.values
            # Handle missing values by skipping regression
            if np.isnan(y).any():
                slopes.append(np.nan)
            else:
                A = np.hstack([X, np.ones_like(X)])
                m, _ = np.linalg.lstsq(A, y.reshape(-1, 1), rcond=None)[0]
                slopes.append(m)
        df[f'{cat}_slope14'] = slopes
    return df

def add_extreme_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary flags for extreme heat/wind/drought events."""
    df['extreme_heat'] = (df['DayAirTmpAvg14'] > 95).astype(int)
    df['high_wind_event'] = (df['DayWindSpdAvg14'] > 20).astype(int)
    df['no_precip_10d'] = df[[f'DayPrecip{str(i).zfill(2)}' for i in range(5, 15)]].sum(axis=1) == 0
    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    # Dynamically compute temp_max14, humidity_min14, wind_max14
    temp_cols = [f'DayAirTmpAvg{str(i).zfill(2)}' for i in range(1, 15)]
    humidity_cols = [f'DayRelHumAvg{str(i).zfill(2)}' for i in range(1, 15)]
    wind_cols = [f'DayWindSpdAvg{str(i).zfill(2)}' for i in range(1, 15)]
    
    temp_max14 = df[temp_cols].max(axis=1)
    humidity_min14 = df[humidity_cols].min(axis=1)
    wind_max14 = df[wind_cols].max(axis=1)
    
    # Now you can safely create the interaction features
    df['temp_humidity_ratio'] = temp_max14 / (humidity_min14 + 1e-3)
    df['wind_temp_product'] = wind_max14 * temp_max14
    
    return df


def add_rolling_std_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling standard deviation over 14-day period for each climate feature."""
    categories = ['DayAirTmpAvg', 'DayPrecip', 'DayRelHumAvg', 'DaySoilTmpAvg', 'DayWindSpdAvg']
    for cat in categories:
        cols = [f'{cat}{str(i).zfill(2)}' for i in range(1, 15)]
        df[f'{cat}_std14'] = df[cols].std(axis=1)
    return df

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratios between key interacting features."""
    df['temp_humidity_ratio'] = df['DayAirTmpAvg14'] / (df['DayRelHumAvg14'] + 1e-3)
    df['wind_temp_ratio'] = df['DayWindSpdAvg14'] / (df['DayAirTmpAvg14'] + 1e-3)
    return df

def add_cumulative_climate_load(df: pd.DataFrame) -> pd.DataFrame:
    """Add total cumulative sum for key climate signals over 14 days."""
    for cat in ['DayAirTmpAvg', 'DayWindSpdAvg']:
        cols = [f'{cat}{str(i).zfill(2)}' for i in range(1, 15)]
        df[f'{cat}_sum14'] = df[cols].sum(axis=1)
    return df

def add_soil_dryness_index(df: pd.DataFrame) -> pd.DataFrame:
    """Create a dryness index based on soil heat vs rainfall."""
    soil_cols = [f'DaySoilTmpAvg{str(i).zfill(2)}' for i in range(1, 15)]
    precip_cols = [f'DayPrecip{str(i).zfill(2)}' for i in range(1, 15)]

    soil_mean = df[soil_cols].mean(axis=1)
    precip_total = df[precip_cols].sum(axis=1) + 1e-3
    df['soil_dryness_index'] = soil_mean / precip_total
    return df

# ==================== TEST ==================== #
if __name__ == "__main__":
    import os
    import sys

    NUM_ROWS = 4000
    # Path to your dataset
    # Add project path to os.path
    current_path = os.path.abspath('.')
    project_name = 'wild_fire'
    project_path = os.path.join(current_path.split(project_name)[0], project_name)
    print(project_path)

    sys.path.append(project_path)
    data_path = os.path.join(project_path, 'data', 'final_data', 'total_data.csv')
    output_dir = os.path.join(project_path, 'data', 'featured_data')
    os.makedirs(output_dir, exist_ok=True)  # Create directory if not exists

    df_original = pd.read_csv(data_path)
    df_original.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    df_original.dropna(inplace=True)
    df_original.drop_duplicates(subset=['date', 'latitude', 'longitude'], keep='first', inplace=True)

    # Drop the FBFM_VALUE column if it's all missing or a constant (to avoid unusable feature)
    if 'FBFM_VALUE' in df_original.columns and df_original['FBFM_VALUE'].nunique() <= 1:
        df_original.drop(columns=['FBFM_VALUE'], inplace=True)

    # Fix latitude/longitude duplication issues: use 'lat'/'lng' as the correct coordinates
    if 'lat' in df_original.columns and 'latitude' in df_original.columns:
        df_original.drop(columns=['latitude', 'longitude'], inplace=True)
        df_original.rename(columns={'lat': 'latitude', 'lng': 'longitude'}, inplace=True)

    def process_and_fill_date_column(df, date_col='date'):
        """
        Process a mixed-format date column, extract date features,
        and fill missing values in those features.
        """
        df = df.copy()
        df[date_col] = [x[:10] for x in df[date_col]]
        df[date_col] = pd.to_datetime(df[date_col])

        # Extract features
        df[f'{date_col}_year'] = df[date_col].dt.year.astype(int)
        df[f'{date_col}_month'] = df[date_col].dt.month.astype(int)
        df[f'{date_col}_day'] = df[date_col].dt.day.astype(int)
        df[f'{date_col}_weekday'] = df[date_col].dt.weekday.astype(int)

        # Fill missing values with mode or a safe fallback
        for col in [f'{date_col}_year', f'{date_col}_month', f'{date_col}_day', f'{date_col}_weekday']:
            if df[col].isna().any():
                mode_val = df[col].mode(dropna=True)
                fallback = mode_val[0] if not mode_val.empty else 0
                df[col] = df[col].fillna(fallback)

        return df

    # Drop rows with any missing values (small percentage of data)
    df_original.dropna(inplace=True)
    df_original = process_and_fill_date_column(df_original)

# ============================================================= #
# ============== FEATURE ENGINEERING TESTING=================== #
# ============================================================= #
    print("Running feature engineering functions individually...\n")

    print("Testing: add_min_max_quantile_features")
    df = df_original.copy()
    df = add_min_max_quantile_features(df)
    df.head(NUM_ROWS).to_csv(os.path.join(output_dir, 'test_output_min_max_quantile.csv'), index=False)

    print("Testing: add_trend_diff_features")
    df = df_original.copy()
    df = add_trend_diff_features(df)
    df.head(NUM_ROWS).to_csv(os.path.join(output_dir, 'test_output_trend_diff.csv'), index=False)

    print("Testing: add_extreme_event_flags")
    df = df_original.copy()
    df = add_extreme_event_flags(df)
    df.head(NUM_ROWS).to_csv(os.path.join(output_dir, 'test_output_extreme_flags.csv'), index=False)

    print("Testing: add_interaction_features")
    df = df_original.copy()
    df = add_interaction_features(df)
    df.head(NUM_ROWS).to_csv(os.path.join(output_dir, 'test_output_interaction_features.csv'), index=False)

    print("Testing: add_rolling_std_features")
    df = df_original.copy()
    df = add_rolling_std_features(df)
    df.head(NUM_ROWS).to_csv(os.path.join(output_dir, 'test_output_std_features.csv'), index=False)

    print("Testing: add_ratio_features")
    df = df_original.copy()
    df = add_ratio_features(df)
    df.head(NUM_ROWS).to_csv(os.path.join(output_dir, 'test_output_ratio_features.csv'), index=False)

    print("Testing: add_cumulative_climate_load")
    df = df_original.copy()
    df = add_cumulative_climate_load(df)
    df.head(NUM_ROWS).to_csv(os.path.join(output_dir, 'test_output_cumulative_climate_load.csv'), index=False)

    print("Testing: add_soil_dryness_index")
    df = df_original.copy()
    df = add_soil_dryness_index(df)
    df.head(NUM_ROWS).to_csv(os.path.join(output_dir, 'test_output_soil_dryness_index.csv'), index=False)

    print("All feature tests completed successfully.")
