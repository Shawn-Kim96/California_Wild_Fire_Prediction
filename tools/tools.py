import os
import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, f1_score,precision_score,recall_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import torch.nn as nn
import torch.nn.functional as F
import joblib  # for sklearn & xgboost models


def process_and_fill_date_column(df, date_col='date'):
    """
    Process a mixed-format date column, extract date features,
    and fill missing values in those features.
    """
    df = df.copy()


    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

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

class GEN_MAKE_UP_DATE(nn.Module):
   def __init__(self, input_shape):
      super(GEN_MAKE_UP_DATE, self).__init__()
      self.fc1= nn.Linear(input_shape, 55)
      self.fc2=nn.Linear(55, 55)
      self.batchNorm= nn.BatchNorm1d(55)
      self.fc3= nn.Linear(55, 1)

   def forward(self,x:torch.Tensor)-> torch.Tensor:
       x = F.relu(self.fc1(x))
       x = F.relu(self.batchNorm(self.fc2(x)))
       x = self.fc3(x)
       return x


columns_to_fix = [
    'DayPrecip11', 'DayPrecip14',
    'DayRelHumAvg01', 'DayRelHumAvg02', 'DayRelHumAvg03', 'DayRelHumAvg04',
    'DayRelHumAvg05', 'DayRelHumAvg06', 'DayRelHumAvg07', 'DayRelHumAvg08',
    'DayRelHumAvg09', 'DayRelHumAvg10', 'DayRelHumAvg11', 'DayRelHumAvg12',
    'DayRelHumAvg13', 'DayRelHumAvg14',
    'DaySoilTmpAvg01', 'DaySoilTmpAvg02', 'DaySoilTmpAvg03', 'DaySoilTmpAvg04'
]

def get_model_for_column(column_name, models_directory="filling_nan_models")->torch.nn.Module | None:
    """
    Retrieves the trained PyTorch model for a specific column.
   
    input:
        column_name (str): The name of the column for which to retrieve the model.
        models_directory: The directory where the model files are saved.

    output:
        model (torch.nn.Module | None): loaded PyTorch model, or None if the model for the given column is not found
    """
    model_filename = f"model_fill_null_for_{column_name}.pth"
    model_filepath = os.path.join(models_directory, model_filename)

    if os.path.exists(model_filepath):
       
        model = GEN_MAKE_UP_DATE(input_shape=93) #FIX_ME: This is not dynamic as we might change the feature vector size
        model.load_state_dict(torch.load(model_filepath))
        model.eval() 
        return model
    else:
        print(f"Model not found for column '{column_name}'")
        return None


def fill_missing_with_models(dataframe: pd.DataFrame, models: dict, columns_to_fix: list = columns_to_fix) -> pd.DataFrame:
    """
    Fills missing values in specified columns using pretrained deep learning models.

    Input:
        dataframe (pd.DataFrame): The input DataFrame containing missing values to be filled.

        models (dict) : A dictionary of trained PyTorch models, where each model corresponds to a specific column
            and is keyed as 'model_fill_null_for_<column_name>'.

    columns_to_fix (list) optional : List of column names to target for missing value imputation. Defaults to a global `columns_to_fix` list.

    Output:
       df (pd.DataFrame): A copy of the input DataFrame with missing values filled in the specified columns.

    """
    df = dataframe.copy()

    for col in columns_to_fix:
        model = models.get('model_fill_null_for_'+col)
        if model is None:
            print(f"Model for '{col}' not found, skipping.")
            continue

        missing_mask = df[col].isna()
        if missing_mask.sum() == 0:
            continue 

        # Prepare input data for prediction
        input_data = df.loc[missing_mask].drop(columns=[col])
        input_data = input_data.fillna(input_data.mean())  # Handle any leftover NaNs in features

        X_missing = torch.tensor(input_data.values, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            predictions = model(X_missing).squeeze().numpy()

        # Fill predictions back into the DataFrame
        df.loc[missing_mask, col] = predictions

        print(f"Filled {missing_mask.sum()} missing values in '{col}'")

    return df



columns_to_fix = [
    'DayPrecip11', 'DayPrecip14',
    'DayRelHumAvg01', 'DayRelHumAvg02', 'DayRelHumAvg03', 'DayRelHumAvg04',
    'DayRelHumAvg05', 'DayRelHumAvg06', 'DayRelHumAvg07', 'DayRelHumAvg08',
    'DayRelHumAvg09', 'DayRelHumAvg10', 'DayRelHumAvg11', 'DayRelHumAvg12',
    'DayRelHumAvg13', 'DayRelHumAvg14',
    'DaySoilTmpAvg01', 'DaySoilTmpAvg02', 'DaySoilTmpAvg03', 'DaySoilTmpAvg04'
]


def generate_models_via_DL(dataframe:pd.DataFrame, columns_to_fix:list=columns_to_fix)->dict:
    """
    Trains a separate deep learning regression model for each specified column with missing values,
    in order to later impute those missing values based on the learned relationships in the data.

    Input:
        dataframe (pd.DataFrame) :  The input dataset containing missing values.

        columns_to_fix (list): List of column names for which deep learning models will be trained to predict missing values.

    Returns:
          models (dict):  A dictionary containing trained PyTorch models keyed by the column name they were trained to predict,
              using the format 'model_fill_null_for_<column_name>'.

    """
    models = {}

    for item in columns_to_fix:
      temp_dataframe = dataframe.dropna(subset=[item])
      train, test = train_test_split(temp_dataframe, test_size=0.3, random_state=42)

      # Fill other missing values
      train = train.fillna(train.mean())
      test = test.fillna(train.mean())

      y_train = torch.tensor(train[item].values, dtype=torch.float32).view(-1, 1)
      y_test = torch.tensor(test[item].values, dtype=torch.float32).view(-1, 1)
      
      X_train = torch.tensor(train.drop(columns=[item]).values, dtype=torch.float32)
      X_test = torch.tensor(test.drop(columns=[item]).values, dtype=torch.float32)

      model = GEN_MAKE_UP_DATE(X_train.shape[1])
      optimizer = optim.Adam(model.parameters(), lr=0.001)

      def rmse_loss(pred, target):
          return torch.sqrt(nn.functional.mse_loss(pred, target))

      with torch.no_grad():
            preds = model(X_test)
            test_rmse = rmse_loss(preds, y_test)
            print(f"BEFORE TRAINING Test RMSE: {test_rmse.item():.4f}")


      model.train()
      for epoch in range(400):
        optimizer.zero_grad()
        output = model(X_train)

        loss = rmse_loss(output, y_train)
        loss.backward()
        optimizer.step()
        if(epoch==99):
           print('col', item)
           print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
      model.eval()

      with torch.no_grad():
            preds = model(X_test)
            test_rmse = rmse_loss(preds, y_test)
            print(f"Test RMSE: {test_rmse.item():.4f}")
      models['model_fill_null_for_'+item]=model

    return models



def save_models(models: dict, save_dir: str = "main_models"):
    os.makedirs(save_dir, exist_ok=True)

    for name, model in models.items():
        model_path = os.path.join(save_dir, f"{name}.pkl" if not isinstance(model, torch.nn.Module) else f"{name}.pth")

        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), model_path)
        else:
            joblib.dump(model, model_path)

    print(f"Models saved to {save_dir}")

class Fire_prediction_model(nn.Module):
   """
   Adhereing to the professor's demands, this shallow and not deep nueral network acts as a skeleton model for how our DL implementation
   will be predicting.
   input:
       input_shape (int): The number of featurees to be inputed to the network.
   output:
       x (torch.Tensor): our prediction from mode.forward(x)

   """
   def __init__(self, input_shape:int, output_shape:int=1):
      super(Fire_prediction_model, self).__init__()
      self.fc1 = nn.Linear(input_shape, 55)
      self.fc2 = nn.Linear(55, 55)
      self.batchNorm = nn.BatchNorm1d(55)
      self.fc3 = nn.Linear(55, output_shape)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
       x = F.relu(self.fc1(x))
       x = F.relu(self.batchNorm(self.fc2(x)))
       x = self.fc3(x)
       return x

def load_models(save_dir: str = "main_models", input_shape: int = 92) -> dict:
    loaded_models = {}
    for filename in os.listdir(save_dir):
        model_path = os.path.join(save_dir, filename)
        name, ext = os.path.splitext(filename)

        if ext == '.pth':
            model = Fire_prediction_model(input_shape=input_shape)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            loaded_models[name] = model
        elif ext == '.pkl':
            loaded_models[name] = joblib.load(model_path)
        else:
            print(f"Unrecognized file type: {filename}")
    print("Loaded all models")
    return loaded_models