import os
import json
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta
from geopy.distance import geodesic

load_dotenv()

BASE_PATH = os.path.abspath('.')
LONGLAT_ZIPCODES = BASE_PATH + "/zipcodes_by_longlat.json"
COUNTY_ZIPCODES = BASE_PATH + "/zipcodes_by_county.json"

# Input & Output files
NON_WILDFIRE_DATES_FILE = BASE_PATH + "/data/processed/non_wildfire/non_wildfire_dates.csv"
OUTPUT_FILE = BASE_PATH + "/data/processed/non_wildfire/processed_non_wildfire_dates.csv"
MIN_START_ROW = 0
MAX_END_ROW = 3100
CHUNK_SIZE = 100  # Number of rows per chunk

class CIMIS:

    def __init__(self, zipcodes=None):
        try:
            self.data_items = {
                "DayAirTmpAvg": "day-air-tmp-avg",      # Average Air Temperature
                "DayPrecip": "day-precip",              # Precipitation
                "DayRelHumAvg": "day-rel-hum-avg",      # Average Relative Humidity
                "DaySoilTmpAvg": "day-soil-tmp-avg",    # Average Soil Temperature
                "DayWindSpdAvg": "day-wind-spd-avg",    # Average Wind Speed
            }
            self.zipcodes = zipcodes if zipcodes is not None else self.get_county_zipcodes()
        except Exception as e:
            print(f"Error occurred when initializing CIMIS: {e}")

    def make_request(self, endpoint, params=None):
        try:
            url = os.getenv("CIMIS_BASE_URL") + endpoint
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Request to CIMIS endpoint {endpoint} failed: {e}")

    def get_data_zipcodes(self, zipcodes, start, end):
        try:
            params = {
                "appKey": os.getenv('CIMIS_API_KEY'),
                "targets": zipcodes,
                "startDate": start,
                "endDate": end,
                "dataItems": ",".join(self.data_items.values()),
                "unitOfMeasure": "M",
            }
            return self.make_request("/data", params)
        except Exception as e:
            print(f"Error: {e}")
    
    def get_county_zipcodes(self):
        try:
            zipcodes = {}
            stations = self.make_request("/station").get("Stations", [])
            for station in stations:
                zipcodes[station.get("City")] = ",".join(station["ZipCodes"])
            return zipcodes
        except Exception as e:
            print(f"Error: {e}")

def process_row_by_county(row, cimis):
    non_incident_date = row.get("date")
    non_incident_county = row.get("county")
    zipcodes = cimis.zipcodes.get(non_incident_county)

    if zipcodes is None:
        return {item: None for item in cimis.data_items.keys()}
    
    station_records = cimis.get_data_zipcodes(zipcodes, non_incident_date, non_incident_date)

    if station_records is None:
        return {item: None for item in cimis.data_items.keys()}
    
    records = station_records.get("Data", {}).get("Providers", [{}])[0].get("Records", None)

    if records is None:
        return {item: None for item in cimis.data_items.keys()}
    
    record = records[0]
    results = {"date": non_incident_date, "latitude": row.get("latitude"), "longitude": row.get("longitude")}
    for item in cimis.data_items.keys():
        value = record.get(item, {}).get("Value", None)
        results[item] = float(value) if value is not None else None
    return results

def process_row_by_longlat(row, cimis):
    non_incident_date = row.get("date")
    non_incident_long = row.get("longitude")
    non_incident_lat = row.get("latitude")

    if not (-90 < non_incident_lat < 90):
        return {item: None for item in cimis.data_items.keys()}
    
    print(non_incident_lat, non_incident_long)
    min_distance = float("inf")
    zipcodes = None

    for key, value in cimis.zipcodes.items():
        long, lat = map(float, key.split(","))
        distance = geodesic((non_incident_lat, non_incident_long), (lat, long)).meters

        if distance < min_distance:
            min_distance = distance
            zipcodes = value

    # Fetch data for the last 14 days
    start_date = (datetime.strptime(non_incident_date, "%Y-%m-%d") - relativedelta(days=14)).strftime("%Y-%m-%d")
    station_records = cimis.get_data_zipcodes(zipcodes, start_date, non_incident_date)

    if station_records is None:
        return {item: None for item in cimis.data_items.keys()}
    
    records = station_records.get("Data", {}).get("Providers", [{}])[0].get("Records", [])

    if not records:
        return {item: None for item in cimis.data_items.keys()}

    results = {}

    # Iterate through the last 14 days and create feature columns
    for i in range(14):
        if i < len(records):  # Ensure we don't access out-of-range indexes
            record = records[i]
            for item in cimis.data_items.keys():
                value = record.get(item, {}).get("Value", None)
                results[f"{item}{str(i+1).zfill(2)}"] = float(value) if value is not None else None
        else:
            # If there are missing days, fill with None
            for item in cimis.data_items.keys():
                results[f"{item}{str(i+1).zfill(2)}"] = None

    return results

def process_non_wildfire_dates(input_file, output_file, start_row, end_row):
    print(f"PROCESSING ROWS {start_row} TO {end_row}...")

    with open(LONGLAT_ZIPCODES, "r") as f:  
        zipcodes = json.load(f)
        cimis = CIMIS(zipcodes)

        df = pd.read_csv(input_file)
        df_subset = df.iloc[start_row:end_row]
        process_results = df_subset.apply(lambda row: process_row_by_longlat(row, cimis), axis=1, result_type="expand")
        df_updated = pd.concat([df_subset, process_results], axis=1)

        # Save each chunk to a temporary file
        temp_output_file = output_file.replace(".csv", f"_{start_row}_{end_row}.csv")
        df_updated.to_csv(temp_output_file, index=False)

        print(f"Chunk saved to {temp_output_file}")

        # df_updated.to_csv(output_file, index=False)
        
    # print(f"PROCESSING COMPLETE. Data saved to {output_file}")

def combine_chunks(output_file, max_row):
    all_chunks = []
    for start_row in range(0, max_row, CHUNK_SIZE):
        end_row = min(start_row + CHUNK_SIZE, max_row)
        temp_output_file = output_file.replace(".csv", f"_{start_row}_{end_row}.csv")
        chunk_df = pd.read_csv(temp_output_file)
        all_chunks.append(chunk_df)
    
    # Combine all chunks into a single dataframe
    final_df = pd.concat(all_chunks, axis=0)
    final_df.to_csv(output_file, index=False)
    print(f"All data combined into {output_file}")

def main():
    start_row = 0
    end_row = start_row + CHUNK_SIZE  # Start with the first chunk
    
    while end_row <= MAX_END_ROW:
        process_non_wildfire_dates(NON_WILDFIRE_DATES_FILE, OUTPUT_FILE, start_row, end_row)
        start_row = end_row
        end_row = min(start_row + CHUNK_SIZE, MAX_END_ROW)
    
    # Combine all processed chunks into one final file
    combine_chunks(OUTPUT_FILE, 3100)

if __name__ == "__main__":
    main()