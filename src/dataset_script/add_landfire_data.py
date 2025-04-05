import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import sklearn
import logging
import argparse


# Add project path to os.path
current_path = os.path.abspath('.')
project_name = 'California_Wild_Fire_Prediction'
project_path = os.path.join(current_path.split(project_name)[0], project_name)
print(project_path)
sys.path.append(project_path)
load_dotenv('.env')

landfire_data_dir = os.path.join(project_path, 'data', 'landfire_data', 'processed')
data_name_column_dict = {
    'cbd': 'CBD_VALUE',
    'evc': 'EVC_VALUE',
    'fbfm': 'FBFM_VALUE',
    'fdist': 'FDIST_VALUE',
    'fvc': 'FVC_VALUE',    
}

def add_landfire_data(fire_df):
    """
    Add new landfire columns (features) to existing fire data. 
    Landfire data is used located in data/landfire_data/processed/
    
    params:
        - fire_df: existing fire data path
    """
    
    for data_name, column_name in data_name_column_dict.items():
        t = time.time()
        logging.info(f"{data_name} landfire data concating...")
        
        data_df = pd.read_csv(os.path.join(landfire_data_dir, f"{data_name}_data.csv"))

        # Build KDTree on cbd_df
        coords = np.radians(data_df[["latitude", "longitude"]].values)
        tree = sklearn.neighbors.BallTree(coords, metric='haversine')

        # Query nearest neighbors for fire_df coords
        fire_coords = np.radians(fire_df[["incident_latitude", "incident_longitude"]].values)
        dist, idx = tree.query(fire_coords, k=1)

        # Add matched CBD info to fire_df
        matched = data_df.iloc[idx.flatten()].reset_index(drop=True)
        fire_df[column_name] = matched[column_name]
        fire_df[f"{data_name}_DISTANCE_KM"] = dist.flatten() * 6371  # Convert from radians to km
        logging.info(f"{data_name} landfire data concate complete :: {time.time() - t}")
    
    return fire_df


if __name__=="__main__":
    print("Input have changed")
    # parser = argparse.ArgumentParser(description='Process fire and add local_fire_data final data paths.')
    # parser.add_argument('--fire_data_path', type=str, required=True, help='Path to the fire data file')
    # parser.add_argument('--final_data_path', type=str, required=True, help='Path to the final data file')
    
    # args = parser.parse_args()

    # add_landfire_data(fire_data_path=args.fire_data_path, final_data_path=args.final_data_path)
