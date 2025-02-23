import requests
import os
import sys

# Add project path to os.path
current_path = os.path.abspath('.')
project_name = 'california_wild_fire_prediction'
project_path = os.path.join(current_path.split(project_name)[0], project_name)
sys.path.append(project_path)

# NOAA API Endpoint
BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"

# Extract California stations from GHCN station list
STATION_LIST_URL = "http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"

def get_california_stations():
    """Fetches California station IDs from the GHCN station list."""
    stations = []
    response = requests.get(STATION_LIST_URL)
    if response.status_code == 200:
        for line in response.text.split("\n"):
            if " CA " in line:  # California stations have ' CA ' in the metadata
                station_id = line[:11].strip()
                stations.append(station_id)
    return stations

def fetch_weather_data(station_ids, start_date, end_date):
    """Fetches weather data for the given stations and date range."""
    params = {
        "dataset": "daily-summaries",
        "startDate": start_date,
        "endDate": end_date,
        "stations": ",".join(station_ids),
        "dataTypes": "ALL",  # Fetch all available observations
        "format": "csv"
    }
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

def save_to_csv(data, filename):
    """Saves fetched data to a CSV file."""
    with open(filename, "w") as file:
        file.write(data)
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    print("Fetching California station IDs...")
    california_stations = get_california_stations()
    print(','.join(california_stations[:10]))
    start_date = '2025-01-01'
    end_date = '2025-01-01'

    data_name = f"ca_{''.join(start_date.split('-'))}_{''.join(end_date.split('-'))}.csv"
    climate_data_path = os.path.join('data', 'climate_data', data_name)

    if not california_stations:
        print("No stations found for California.")
    else:
        print(f"Found {len(california_stations)} stations in California.")
        print("Fetching weather data...")

        weather_data = fetch_weather_data(california_stations, "2025-01-01", "2025-01-01")
        
        if weather_data:
            save_to_csv(weather_data, climate_data_path)
