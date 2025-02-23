import requests
import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta

# Add project path to os.path
current_path = os.path.abspath('.')
project_name = 'california_wild_fire_prediction'
project_path = os.path.join(current_path.split(project_name)[0], project_name)
sys.path.append(project_path)
load_dotenv('.env')

# NOAA API Endpoint
BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
API_TOKEN = os.environ.get("NOAA_TOKEN")

# Request headers
HEADERS = {"token": API_TOKEN}

# Parameters
DATASET_ID = "GHCND"
LOCATION_ID = "FIPS:06"  # California
# LIMIT = 1000  # Max API return size per request
LIMIT = 100


def fetch_data(start_date, end_date):
    """Fetch weather data from NOAA CDO API within the given date range, handling pagination."""
    offset = 0
    all_data = []

    while True:
        params = {
            "datasetid": DATASET_ID,
            "locationid": LOCATION_ID,
            "startdate": start_date,
            "enddate": end_date,
            "limit": LIMIT,
            "offset": offset
        }

        response = requests.get(BASE_URL, headers=HEADERS, params=params)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break
        
        data = response.json().get("results", [])
        if not data:
            break  # Stop if no more data

        all_data.extend(data)
        offset += LIMIT  # Move to next batch

        time.sleep(1)  # Prevent hitting API rate limit

    return all_data


def get_month_ranges(start_date, end_date):
    """Generate precise month-long date ranges between start_date and end_date."""
    ranges = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    final_date = datetime.strptime(end_date, "%Y-%m-%d")

    while current_date <= final_date:
        # Set the end of the current month OR the final date (whichever is smaller)
        last_day_of_month = (current_date.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)
        range_end = min(last_day_of_month, final_date)

        ranges.append((current_date.strftime("%Y-%m-%d"), range_end.strftime("%Y-%m-%d")))

        # Move to the first day of the next month
        current_date = range_end + timedelta(days=1)

    return ranges



if __name__ == "__main__":
    start_date = '2025-01-01'
    end_date = '2025-01-31'

    date_ranges = get_month_ranges(start_date, end_date)
    all_climate_data = []

    data_name = f"ca_{''.join(start_date.split('-'))}_{''.join(end_date.split('-'))}.csv"
    filename = os.path.join('data', 'climate_data', data_name)

    for start, end in date_ranges:
        print(f"Fetching data from {start} to {end}...")
        weather_data = fetch_data(start, end)
        
        if weather_data:
            all_climate_data.extend(weather_data)

    if all_climate_data:
        df = pd.DataFrame(all_climate_data)
        df.to_csv(filename, index=False)
        print(f"Weather data saved to {filename}.")
    else:
        print("No data retrieved.")
