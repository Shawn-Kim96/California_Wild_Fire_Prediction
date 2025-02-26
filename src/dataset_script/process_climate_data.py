import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add project path to os.path
current_path = os.path.abspath('.')
project_name = 'california_wild_fire_prediction'
project_path = os.path.join(current_path.split(project_name)[0], project_name)
sys.path.append(project_path)
load_dotenv('.env')

data_path = 'data/climate_data'
total_data = pd.DataFrame()
for data_name in os.listdir(data_path)[:2]:
    total_data = pd.concat([total_data, pd.read_csv(os.path.join(data_path, data_name))])

total_data.sort_values(by=['DATE', 'STATION'])
print(total_data.head(4))
print(total_data.tail(4))
print(1)