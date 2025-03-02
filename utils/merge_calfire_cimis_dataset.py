import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from geopy.distance import geodesic

load_dotenv()

BASE_PATH = os.path.abspath('.')
START_ROW = 2605
END_ROW = 2775
COUNTY_ZIPCODES = BASE_PATH + "/zipcodes_by_county.json"
LONGLAT_ZIPCODES = BASE_PATH + "/zipcodes_by_longlat.json"
CALFIRE_DATA_FILE = BASE_PATH + "/data/processed/draft2.csv"
PROCESSED_DATA_FILE = BASE_PATH + f"/data/processed/{START_ROW}_{END_ROW}.csv"

class CIMIS:

    def __init__(self, zipcodes=None):
        try:
            self.data_items = {
                "DayAirTmpAvg": "day-air-tmp-avg",
                "DayPrecip": "day-precip",
                "DayRelHumAvg": "day-rel-hum-avg",
                "DaySoilTmpAvg": "day-soil-tmp-avg",
                "DayWindSpdAvg": "day-wind-spd-avg",
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
                "appKey": os.getenv("CIMIS_API_KEY"),
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
        # print(f"JSON: {lat}, {long}")
        distance = geodesic((incident_lat, incident_long), (lat, long)).meters

        if distance < min_distance:
            min_distance = distance
            zipcodes = value

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

with open(LONGLAT_ZIPCODES, "r") as f:
    zipcodes = json.load(f)
    cimis = CIMIS(zipcodes)

    df = pd.read_csv(CALFIRE_DATA_FILE)
    df_subset = df.iloc[START_ROW:END_ROW]
    process_results = df_subset.apply(lambda row: process_row_by_longlat(row, cimis), axis=1, result_type="expand")
    df_updated = pd.concat([df_subset, process_results], axis=1)
    df_updated.to_csv(PROCESSED_DATA_FILE, index=False)