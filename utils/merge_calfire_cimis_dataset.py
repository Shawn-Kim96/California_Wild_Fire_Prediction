import os
import sys
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

current_path = os.path.abspath('.')
project_name = 'california_wild_fire_prediction'
BASE_PATH = os.path.join(current_path.split(project_name)[0], project_name)
sys.path.append(BASE_PATH)
load_dotenv('.env')


class CIMIS:

    def __init__(self, zipcodes=None):
        self.data_items = {
            "DayAirTmpAvg": "day-air-tmp-avg",
            "DayPrecip": "day-precip",
            "DayRelHumAvg": "day-rel-hum-avg",
            "DaySoilTmpAvg": "day-soil-tmp-avg",
            "DayWindSpdAvg": "day-wind-spd-avg",
        }
        self.zipcodes = zipcodes if zipcodes is not None else self.get_available_zipcodes()

    def make_request(self, endpoint, params=None):
        try:
            url = os.getenv("CIMIS_BASE_URL") + endpoint
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Request to CIMIS endpoint {endpoint} failed: {e}")

    def get_data_by_zipcodes(self, zipcodes, start, end):
        params = {
            "appKey": os.getenv("CIMIS_API_KEY"),
            "targets": zipcodes,
            "startDate": start,
            "endDate": end,
            "dataItems": ",".join(self.data_items.values()),
            "unitOfMeasure": "M",
        }
        return self.make_request("/data", params)
    
    def get_available_zipcodes(self):
        zipcodes = {}
        stations = self.make_request("/station").get("Stations", [])
        for station in stations:
            zipcodes[station.get("City")] = ",".join(station["ZipCodes"])
        return zipcodes


def process_row(row, cimis):
    incident_date = row.get("incident_dateonly_created")
    incident_county = row.get("incident_county")
    zipcodes = cimis.zipcodes.get(incident_county)

    if zipcodes is None:
        return {item: None for item in cimis.data_items.keys()}
    
    station_records = cimis.get_data_by_zipcodes(zipcodes, incident_date, incident_date)
    records = station_records.get("Data", {}).get("Providers", [{}])[0].get("Records", None)

    if records is None:
        return {item: None for item in cimis.data_items.keys()}
    
    record = records[0]
    results = {}
    for item in cimis.data_items.keys():
        value = record.get(item, {}).get("Value", None)
        results[item] = float(value) if value is not None else None
    return results

cimis = CIMIS()
start, end = 101, 200  # Change the values here to dictate what rows to fetch
calfire_data_file = os.path.join(BASE_PATH, "/data/raw/mapdataall.csv")
processed_data_file = os.path.join(BASE_PATH, f"/data/processed/{start}_{end}.csv")

df = pd.read_csv(calfire_data_file)
df_subset = df.iloc[start:end]
process_results = df_subset.apply(lambda row: process_row(row, cimis), axis=1, result_type="expand")
df_updated = pd.concat([df_subset, process_results], axis=1)
df_updated.to_csv(processed_data_file, index=False)
