import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataPreprocess:
    def __init__(self, data):
        self.data = data
        self.climate_columns = [x for x in data.columns if 'Day' in x]
        self.autoencoder = None
        self.scaler = None
        self.preprocessed_data = None
        self.process_and_fill_date_column()
        self.drop_critical_columns()
        
    def process_and_fill_date_column(self, date_col='date'):
        """
        Process a mixed-format date column, extract date features,
        and fill missing values in those features.
        """
        df = self.data.copy()
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

        self.preprocessed_data = df

    def fill_climate_nan_value(self, method: str):
        climate_data = self.preprocessed_data[self.climate_columns]

        if method == 'autoencoder':
            complete_climate_data = climate_data.dropna()
            self.autoencoder, self.scaler = train_autoencoder_with_complete_data(complete_climate_data)
            nan_filled_data = fill_missing_values_only_with_autoencoder(climate_data, self.autoencoder, self.scaler)
            self.preprocessed_data[self.climate_columns] = nan_filled_data
            
        elif method == 'fill_zero':
            nan_filled_data = self.preprocessed_data.fillna(0)

        elif method == 'mean':
            nan_filled_data = self.preprocessed_data.copy()
            for col in self.climate_columns:
                if col[-2:].isdigit():
                    prefix = ''.join([c for c in col if not c.isdigit()])
                    num = int(''.join([c for c in col if c.isdigit()]))
                    prev_col = f"{prefix}{num-1:02d}"
                    next_col = f"{prefix}{num+1:02d}"

                    mask = nan_filled_data[col].isna()
                    if prev_col in self.preprocessed_data.columns and next_col in self.preprocessed_data.columns:
                        nan_filled_data.loc[mask, col] = (
                            nan_filled_data.loc[mask, [prev_col, next_col]].mean(axis=1)
                        )
                    elif prev_col in self.preprocessed_data.columns:
                        nan_filled_data.loc[mask, col] = nan_filled_data.loc[mask, prev_col]
                    elif next_col in self.preprocessed_data.columns:
                        nan_filled_data.loc[mask, col] = nan_filled_data.loc[mask, next_col]
            self.preprocessed_data[self.climate_columns] = nan_filled_data[self.climate_columns]

        else:
            raise ValueError(f"Unknown method '{method}'")

        self.preprocessed_data = nan_filled_data
        original_length = len(self.preprocessed_data)

        # Only drop rows with NaN in climate columns
        self.preprocessed_data = self.preprocessed_data.dropna(subset=self.climate_columns)
        # print(f"Dropping Nan values after processing :: {original_length} -> {len(self.preprocessed_data)}")
        return self.preprocessed_data


    def drop_critical_columns(self):
        drop_cols = ['burn_probability', 'conditional_flame_length', 
                 'distance_km', 'exposure', 'flame_length_exceedance_4ft', 'flame_length_exceedance_8ft',
                 'wildfire_hazard_potential', 'risk_to_structures', 'acres_burned', 'latitude', 'longitude', 'date']
        self.preprocessed_data.drop(columns=[col for col in drop_cols if col in self.preprocessed_data.columns], inplace=True, errors='ignore')
        if 'Unnamed: 0' in self.preprocessed_data.columns:
            self.preprocessed_data.drop(columns=['Unnamed: 0'], inplace=True)
        

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Train autoencoder with full data (fillna(0) as a simple mask strategy)
def train_autoencoder_with_complete_data(complete_data: pd.DataFrame, num_epochs=30, batch_size=64):
    df_clean = complete_data.dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    model = Autoencoder(input_dim=complete_data.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dataset = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
    
    return model, scaler


def fill_missing_values_only_with_autoencoder(data: pd.DataFrame, model, scaler):
    model.eval()
    data_filled = data.copy()
    X_input = data_filled.fillna(0).values
    
    X_scaled = scaler.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed = model(X_tensor).cpu().numpy()
        reconstructed = scaler.inverse_transform(reconstructed)

    # Fill only the NaNs in original df
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if pd.isna(data.iat[i, j]):
                data_filled.iat[i, j] = reconstructed[i, j]

    return data_filled



if __name__=="__main__":
    """
    This it some sample code on how to use data preprocess module.
    """
    
    total_df = pd.read_csv("/Users/shawn/Documents/sjsu/2025-1/ML_CMPE257/California_Wild_Fire_Prediction/data/final_data/total_data.csv")
    data_preprocess = DataPreprocess(data=total_df)
    
    i = 0
    for _, row in total_df.iterrows():
        i += 1 if any(row.isna()) else 0
    print(f"Original data Nan value row count = {i}")
    
    df_preprocessed = data_preprocess.fill_climate_nan_value(method='autoencoder')
    
    i = 0
    for _, row in df_preprocessed.iterrows():
        i += 1 if any(row.isna()) else 0
    print(f"Preprocessed data Nan value row count = {i}")
