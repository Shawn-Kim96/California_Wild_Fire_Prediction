import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Autoencoder Model
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
def train_autoencoder_with_complete_data(df, num_epochs=30, batch_size=64):
    df_clean = df.dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    model = Autoencoder(input_dim=df.shape[1]).to(device)
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


def fill_missing_values_only(df, model, scaler):
    model.eval()
    df_filled = df.copy()
    X_input = df.fillna(0).values
    
    X_scaled = scaler.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed = model(X_tensor).cpu().numpy()
        reconstructed = scaler.inverse_transform(reconstructed)

    # Fill only the NaNs in original df
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if pd.isna(df.iat[i, j]):
                df_filled.iat[i, j] = reconstructed[i, j]

    return df_filled
