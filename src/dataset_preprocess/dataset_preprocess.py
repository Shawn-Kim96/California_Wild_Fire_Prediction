import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from dataset_preprocess.autoencoder import train_autoencoder_with_complete_data, fill_missing_values_only_with_autoencoder



def fill_climate_nan_value(data: pd.DataFrame, method: str):
    
    if method == 'autoencoder':
        model, scaler = train_autoencoder_with_complete_data




def process_and_fill_date_column(df, date_col='date'):
    """
    Process a mixed-format date column, extract date features,
    and fill missing values in those features.
    """
    df = df.copy()
    df[date_col] = [x[:10] for x in df[date_col]]
    df[date_col] = pd.to_datetime(df[date_col])

    # Extract features
    df[f'{date_col}_year'] = df[date_col].dt.year
    df[f'{date_col}_month'] = df[date_col].dt.month
    df[f'{date_col}_day'] = df[date_col].dt.day
    df[f'{date_col}_weekday'] = df[date_col].dt.weekday

    # Fill missing values with mode or a safe fallback
    for col in [f'{date_col}_year', f'{date_col}_month', f'{date_col}_day', f'{date_col}_weekday']:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            fallback = mode_val[0] if not mode_val.empty else 0
            df[col] = df[col].fillna(fallback)

    return df

