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
START_ROW = 0
END_ROW = 2063
COUNTY_ZIPCODES = BASE_PATH + "/zipcodes_by_county.json"
LONGLAT_ZIPCODES = BASE_PATH + "/zipcodes_by_longlat.json"
CALFIRE_DATA_FILE = BASE_PATH + "/data/processed/raw/mapdataall.csv"
PROCESSED_DATA_FILE = BASE_PATH + f"/data/processed/{START_ROW}_{END_ROW}.csv"

class CIMIS:

    def __init__(self, zipcodes=None):
        try:
            # Daily class
            self.data_items = {
                "DayAirTmpAvg": "day-air-tmp-avg",      # Average Air Temperature
                "DayPrecip": "day-precip",              # Precipitation
                "DayRelHumAvg": "day-rel-hum-avg",      # Average Relative Humidity
                "DaySoilTmpAvg": "day-soil-tmp-avg",    # Average Soil Temperature
                "DayWindSpdAvg": "day-wind-spd-avg",    # Average Wind Speed
            }
            self.zipcodes = zipcodes if zipcodes is not None else self.get_county_zipcodes()
        except Exception as e:
            print(f"Error occured when initializing CIMIS: {e}")

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

    def get_longlat_zipcodes(self):
        try:
            zipcodes = {}
            stations = self.make_request("/station").get("Stations", [])
            for station in stations:
                longitude = station.get("HmsLongitude", "").split(" / ")[-1]
                latitude = station.get("HmsLatitude", "").split(" / ")[-1]
                zipcodes[f"{longitude},{latitude}"] = ",".join(station.get("ZipCodes", []))
            
            return zipcodes
                
        except Exception as e:
            print(f"Error: {e}")

def process_row_by_county(row, cimis):
    incident_date = row.get("incident_dateonly_created")
    incident_county = row.get("incident_county")
    zipcodes = cimis.zipcodes.get(incident_county)

    if zipcodes is None:
        return {item: None for item in cimis.data_items.keys()}
    
    station_records = cimis.get_data_zipcodes(zipcodes, incident_date, incident_date)

    if station_records is None:
        return {item: None for item in cimis.data_items.keys()}
    
    records = station_records.get("Data", {}).get("Providers", [{}])[0].get("Records", None)

    if records is None:
        return {item: None for item in cimis.data_items.keys()}
    
    record = records[0]
    results = {}
    for item in cimis.data_items.keys():
        value = record.get(item, {}).get("Value", None)
        results[item] = float(value) if value is not None else None
    return results

def process_row_by_longlat(row, cimis):
    incident_date = row.get("incident_dateonly_created")
    incident_long = row.get("incident_longitude")
    incident_lat = row.get("incident_latitude")

    if not (-90 < incident_lat < 90):
        return {item: None for item in cimis.data_items.keys()}
    
    print(incident_lat, incident_long)
    min_distance = float("inf")
    zipcodes = None

    for key, value in cimis.zipcodes.items():
        long, lat = map(float, key.split(","))
        distance = geodesic((incident_lat, incident_long), (lat, long)).meters

        if distance < min_distance:
            min_distance = distance
            zipcodes = value

    # Fetch data for the last 14 days
    start_date = (datetime.strptime(incident_date, "%Y-%m-%d") - relativedelta(days=14)).strftime("%Y-%m-%d")
    station_records = cimis.get_data_zipcodes(zipcodes, start_date, incident_date)

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

def fetch_cimis_data(input_file, output_file, start_row, end_row):
    print(f"PROCESSING ROWS {start_row} TO {end_row}...")

    with open(LONGLAT_ZIPCODES, "r") as f:  
        zipcodes = json.load(f)
        cimis = CIMIS(zipcodes)

        df = pd.read_csv(input_file)
        df_subset = df.iloc[start_row:end_row]
        process_results = df_subset.apply(lambda row: process_row_by_longlat(row, cimis), axis=1, result_type="expand")
        df_updated = pd.concat([df_subset, process_results], axis=1)
        df_updated.to_csv(output_file, index=False)
        
    print("PROCESSING COMPLETE.")

def organize_csv(file):
    output_file = BASE_PATH + "/data/processed/organized_output.csv"

    df = pd.read_csv(file)

    last_70_cols = df.columns[-70:]
    filled_rows = df.dropna(subset=last_70_cols, how="any")
    unfilled_rows = df[~df.index.isin(filled_rows.index)]

    filled_rows_sorted = filled_rows.sort_values(by="incident_name", ascending=True)

    sorted_df = pd.concat([filled_rows_sorted, unfilled_rows], ignore_index=True)
    sorted_df.to_csv(output_file, index=False)

def merge_csvs(file1, file2, index):
    output_file = BASE_PATH + "/data/processed/merged_output.csv"

    df_a = pd.read_csv(file1)
    df_b = pd.read_csv(file2)

    df_a.set_index(index, inplace=True)
    df_b.set_index(index, inplace=True)

    df_a.update(df_b)

    df_a.reset_index(inplace=True)

    df_a.to_csv(output_file, index=False)

def find_incomplete_rows(file):
    output_file = BASE_PATH + "/data/processed/incomplete_rows_output.csv"

    df = pd.read_csv(file)
    
    # Select the last 14 columns
    last_14_cols = df.columns[-14:]
    
    # Filter rows where any of the last 14 columns have missing values
    filtered_df = df[df[last_14_cols].isnull().any(axis=1)]
    
    # Write the filtered rows to a new CSV file
    filtered_df.to_csv(output_file, index=False)
    
def remove_unwanted_features(file, target_columns):
    output_file = BASE_PATH + "/data/processed/only_wanted_features.csv"

    df = pd.read_csv(file)
    df = df.drop(columns=[col for col in target_columns if col in df.columns], errors='ignore')
    df.to_csv(output_file, index=False)

def remove_incomplete_rows(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df.dropna()
    df.to_csv(output_file, index=False)

if __name__ == "__main__":

    # Fetching data
    # fetch_cimis_data(CALFIRE_DATA_FILE, PROCESSED_DATA_FILE, START_ROW, END_ROW)

    # Extracting incompletes
    # input_file = BASE_PATH + "/data/processed/draft5.csv"
    # find_incomplete_rows(input_file)

    # Fetching data for incompletes
    # input_file = BASE_PATH + "/data/processed/incomplete_rows_output.csv"
    # output_file = BASE_PATH + "/data/processed/incomplete_rows_output_v2.csv"
    # fetch_cimis_data(input_file, output_file, 0, 322)

    # Remove unwanted columns
    # unwanted_columns = {
    #     "incident_is_final",
    #     "incident_administrative_unit",
    #     "incident_data_last_update,"
    #     "incident_date_created",
    #     "incident_is_final",
    #     "incident_administrative_unit_url",
    #     "incident_location",
    #     "incident_county",
    #     "incident_containment",
    #     "incident_control",
    #     "incident_cooperating_agencies",
    #     "incident_type",
    #     "incident_url",
    #     "incident_date_extinguished",
    #     "incident_dateonly_extinguished",
    #     "is_active",
    #     "calfire_incident",
    #     "notification_desired",
    # }

    # input_file = BASE_PATH + "/data/processed/draft5.csv"
    # remove_unwanted_features(input_file, unwanted_columns)

    # Keep only the completed rows
    input_file = BASE_PATH + "/data/processed/finalized/calfire_cimis_all_rows.csv"
    output_file = BASE_PATH + "/data/processed/finalized/calfire_cimis_completed_rows.csv"
    remove_incomplete_rows(input_file, output_file)