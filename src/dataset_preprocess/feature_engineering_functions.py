import pandas as pd
import numpy as np

''' add_min_max_quantile_features
Reason: Identifies volatility and risk windows. High max temps or low Q25 humidity = danger zone
        These help the model grasp extreme conditions and variability, not just central tendencies
Details:
        'max': peakly daily temperatures or wind speeds may directly contribute to ignition or fire spread
        'min': lowest relative humidity or coldest day can reveal
        'quantiles (Q25, Q75): show the distribution shape and help the model understand variability
                e.g.:: whether a region had a consisten high temperature rather than just one-off peaks.
        'Interquatile Range (IQR): Measures climate volatibility as fire is more likely in unstable, swinging 
                enviroinments, where there is continuous fluctuating wind and temperature.
'''
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


'''add_trend_diff_features
Reason: Detect sudden climate changes (e.g., drying out quickly), which can precede wildfires. Particularly useful 
        for models to detect "build-up" patterns — not just what's happening now, but what's been changing
Details:
        'First-order difference': Measures net change over 14 days. For example, a sharp drop in humidity or spike
                in temperature could mean vegetation is drying fast.
        'slope': fitting a simple regression line across 14 days gives the rate of change. A positive slope in 
                temperature and negative in humidity may signal escalating fire risk.
'''
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



''' add_extreme_event_flags
Reason: Let the model learn binary conditions associated with wildfires — based on thresholds observed
        in domain knowledge (e.g., no rain = dry vegetation, high wind spreads fires, extreme heat triggers ignition).
Details:
        'extreme_heat': Fires ignite more easily and vegetation dries rapidly under high temperatures. 
                A threshold of >95°F is considered high-risk in many fire-prone areas.
        'high_wind_event': Wind speeds above 20 mph can spread fires rapidly and carry embers across distances.
        'no_precip_10d': A dry spell of 10 consecutive days (no measurable rainfall) leads to dry vegetation 
                and low soil moisture, significantly increasing flammability.
'''
def add_extreme_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary flags for extreme heat/wind/drought events."""
    df['extreme_heat'] = (df['DayAirTmpAvg14'] > 95).astype(int)
    df['high_wind_event'] = (df['DayWindSpdAvg14'] > 20).astype(int)
    df['no_precip_10d'] = df[[f'DayPrecip{str(i).zfill(2)}' for i in range(5, 15)]].sum(axis=1) == 0
    return df


''' add_interaction_features
Reason: Combined climate effects are often stronger predictors of fire than any one variable alone.
        For example, hot + dry is much riskier than just hot or just dry.
Details:
        'temp_humidity_ratio': When high temperatures are paired with low humidity, vegetation dries rapidly.
        'wind_temp_product': Heat and wind together can escalate both ignition and spread, especially in dry zones.
'''
def add_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    data['temp_humidity_ratio'] = data['temp_max14'] / (data['humidity_min14'] + 1e-3)
    data['wind_temp_product'] = data['wind_max14'] * data['temp_max14']
    return data


''' add_seasonal_indicators
Reason: California's wildfire season typically starts in summer and peaks during fall.
        Seasonality is a strong temporal prior for wildfire activity.
Details:
        'is_summer': Binary flag for June-August.
        'is_fall': Binary flag for September-November, which often has dry conditions and strong winds (e.g. Diablo winds).
'''
def add_seasonal_indicators(data: pd.DataFrame) -> pd.DataFrame:
    # Assumes `date_month` already added
    data['is_summer'] = data['date_month'].isin([6, 7, 8]).astype(int)
    data['is_fall'] = data['date_month'].isin([9, 10, 11]).astype(int)
    return data


''' add_normalized_climate_features
Reason: Normalized values help quantify relative stress — not just how hot/windy it is, but how extreme that is compared
        to environmental factors like soil moisture or average wind.
Details:
        'temp_norm': Max temp normalized by soil temperature. High values may indicate rapid surface heating, increasing risk.
        'humidity_wind_index': Low humidity with strong winds can be dangerous. This feature captures that balance.
'''
def add_normalized_climate_features(data: pd.DataFrame) -> pd.DataFrame:
    data['temp_norm'] = data['temp_max14'] / (data['soil_temp_avg14'] + 1e-3)
    data['humidity_wind_index'] = data['humidity_min14'] / (data['wind_avg14'] + 1e-3)
    return data

''' add_rolling_std_features
Reason: Measures short-term climate instability — frequent fluctuation in temperature, wind, etc. is often linked
        to volatile fire conditions. Models may pick up on this “noisy” behavior as a risk signal.
Details:
        'rolling std': Captures how much variation happens over the 14-day period. A high std in temperature or 
        wind can mean unstable atmospheric behavior which can both start and worsen wildfires.
'''
def add_rolling_std_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling standard deviation over 14-day period for each climate feature."""
    categories = ['DayAirTmpAvg', 'DayPrecip', 'DayRelHumAvg', 'DaySoilTmpAvg', 'DayWindSpdAvg']
    for cat in categories:
        cols = [f'{cat}{str(i).zfill(2)}' for i in range(1, 15)]
        df[f'{cat}_std14'] = df[cols].std(axis=1)
    return df


''' add_ratio_features
Reason: Interaction between temperature, humidity, and wind often signals compounded wildfire risk.
        Ratios help the model understand conditions that are only dangerous when combined.
Details:
        'temp/humidity': High heat and low humidity are much worse together than separately.
        'wind/temp': Fast wind during heat waves makes it easier for wildfires to spread.
'''
def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratios between key interacting features."""
    df['temp_humidity_ratio'] = df['DayAirTmpAvg14'] / (df['DayRelHumAvg14'] + 1e-3)
    df['wind_temp_ratio'] = df['DayWindSpdAvg14'] / (df['DayAirTmpAvg14'] + 1e-3)
    return df


''' add_temperature_anomaly_flag
Reason: Abnormally high temperatures in a short time span can indicate abnormal drying, heat waves, or regional stress.
Details:
        'temperature anomaly flag': Compares max temperature of the last 14 days to the average monthly expectation.
        If the recent max temp is much higher than typical for that month, it could signal elevated fire risk.
'''
def add_temperature_anomaly_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary feature indicating a temperature anomaly vs typical monthly average (hardcoded baseline).
    Can use this website as reference
    https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/4/tavg/1/8/2012-2021?base_prd=true&begbaseyear=1901&endbaseyear=2000
    """
    monthly_avg_temp = {
        1: 55, 
        2: 58, 
        3: 60, 
        4: 65, 
        5: 70, 
        6: 78,
        7: 85, 
        8: 87, 
        9: 82, 
        10: 75, 
        11: 65, 
        12: 58
    }
    df['expected_monthly_temp'] = df['date_month'].map(monthly_avg_temp)
    df['temp_anomaly_flag'] = (df['DayAirTmpAvg14'] > df['expected_monthly_temp'] + 10).astype(int)
    df.drop(columns=['expected_monthly_temp'], inplace=True)
    return df


''' add_precipitation_drought_index
Reason: Dryness is one of the most important precursors to wildfire. A prolonged lack of precipitation is a strong signal.
Details:
        'drought_index_14': Total days with less than 0.1 inches of rain.
        This acts like a simple drought index — more dry days = more fire-prone.
'''
def add_precipitation_drought_index(df: pd.DataFrame) -> pd.DataFrame:
    """Count number of dry days (precip < 0.1 in) over past 14 days."""
    cols = [f'DayPrecip{str(i).zfill(2)}' for i in range(1, 15)]
    df['drought_index_14'] = (df[cols] < 0.1).sum(axis=1)
    return df


''' add_cumulative_climate_load
Reason: The total buildup of heat, wind, or dryness over time contributes to drying out vegetation.
Details:
        'climate load': Total sum over 14 days. High cumulative wind or heat means more stress on the environment.
        Useful for detecting long-term stress even if there aren't spikes (i.e., constant exposure).
'''
def add_cumulative_climate_load(df: pd.DataFrame) -> pd.DataFrame:
    """Add total cumulative sum for key climate signals over 14 days."""
    for cat in ['DayAirTmpAvg', 'DayWindSpdAvg']:
        cols = [f'{cat}{str(i).zfill(2)}' for i in range(1, 15)]
        df[f'{cat}_sum14'] = df[cols].sum(axis=1)
    return df


''' add_zero_humidity_days
Reason: Extreme fire risk increases when there are multiple ultra-low humidity days (< 20%).
Details:
        'zero_humidity_days': Number of days in the past 14 with critical dryness.
        Many fire agencies consider <20% RH to be "Red Flag" conditions.
'''
def add_zero_humidity_days(df: pd.DataFrame) -> pd.DataFrame:
    """Count number of days with dangerously low humidity (<20%)."""
    cols = [f'DayRelHumAvg{str(i).zfill(2)}' for i in range(1, 15)]
    df['zero_humidity_days'] = (df[cols] < 20).sum(axis=1)
    return df


''' add_soil_dryness_index
Reason: Soil temperature and precipitation together influence how dry the ground is — a proxy for vegetation moisture.
Details:
        'soil_dryness_index': Soil temperature divided by 14-day precipitation total.
        High soil temp + low rainfall = dry environment more likely to burn.
'''
def add_soil_dryness_index(df: pd.DataFrame) -> pd.DataFrame:
    """Create a dryness index based on soil heat vs rainfall."""
    soil_cols = [f'DaySoilTmpAvg{str(i).zfill(2)}' for i in range(1, 15)]
    precip_cols = [f'DayPrecip{str(i).zfill(2)}' for i in range(1, 15)]

    soil_mean = df[soil_cols].mean(axis=1)
    precip_total = df[precip_cols].sum(axis=1) + 1e-3
    df['soil_dryness_index'] = soil_mean / precip_total
    return df


''' add_fire_season_flag
Reason: Fire season in California (usually June-November) significantly raises wildfire probability regardless of weather.
Details:
        'fire_season_flag': A binary flag marking June to November as fire season.
        Gives the model a temporal prior to work with.
'''
def add_fire_season_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary feature indicating if the date falls within peak wildfire season."""
    df['fire_season_flag'] = df['date_month'].isin([6, 7, 8, 9, 10, 11]).astype(int)
    return df

# ================================================== #
# ====================== TEST ====================== #
# ================================================== #
if __name__ == "__main__":
    import os
    import sys
    import argparse

    # ----------- Command-line Arguments -------------
    # e.g.: python3 feature_engineering_functions --print
    parser = argparse.ArgumentParser(description="Test feature engineering functions.")
    parser.add_argument("--print", action="store_true", help="Print output to console")
    args = parser.parse_args()
    SHOULD_PRINT = args.print

    NUM_ROWS = 100
    # Path to your dataset
    # Add project path to os.path
    current_path = os.path.abspath('.')
    project_name = 'wild_fire'
    project_path = os.path.join(current_path.split(project_name)[0], project_name)
    print(project_path)

    sys.path.append(project_path)
    data_path = project_path + '/data/final_data/total_data.csv'
    output_dir = project_path + '/src/dataset_preprocess/featured_dataset_check'
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

    # -----------------------------
    print("Testing: add_min_max_quantile_features")
    df = df_original.copy()
    df = add_min_max_quantile_features(df)
    df.head(NUM_ROWS).to_csv(os.path.join(output_dir, 'test_output_min_max_quantile.csv'), index=False)
    if SHOULD_PRINT:
        print(df.filter(regex='_(min14|max14|q25|q75|iqr)').head())
    print("Saved: featured_dataset_check/test_output_min_max_quantile.csv\n")

    # -----------------------------
    print("Testing: add_trend_diff_features")
    df = df_original.copy()
    df = add_trend_diff_features(df)
    df.head(NUM_ROWS).to_csv(os.path.join(output_dir, 'test_output_trend_diff.csv'), index=False)
    if SHOULD_PRINT:
        print(df.filter(regex='_(diff14|slope14)').head())
    print("Saved: featured_dataset_check/test_output_trend_diff.csv\n")

    # -----------------------------
    print("Testing: add_extreme_event_flags")
    df = df_original.copy()
    df = add_extreme_event_flags(df)
    df.head(NUM_ROWS).to_csv(os.path.join(output_dir, 'test_output_extreme_flags.csv'), index=False)
    if SHOULD_PRINT:
        print(df.filter(regex='(extreme_heat|high_wind_event|no_precip_10d)').head())
    print("Saved: featured_dataset_check/test_output_extreme_flags.csv\n")

    print("Done testing all functions individually.")
