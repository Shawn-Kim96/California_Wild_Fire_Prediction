## NOAA's NCEI API Request to Fetch Climate Data

### Guildline document
- [google drive link](https://drive.google.com/drive/u/0/folders/1lETRgX6_F8OhvQ1kRv6g2aK5mVprDFxG)
### API Endpoint:
```
https://www.ncei.noaa.gov/access/services/data/v1
```

### Request Parameters:
- `dataset`: daily-summaries
- `startDate`: 2025-01-01
- `endDate`: 2025-01-01
- `stations`: (All California stations, which we can extract [GHCN station list](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt))
- `dataTypes`: (Based on Table 4 in the documentation, we can retrieve all available parameters)
- `format`: csv

### Example API Request:
```
https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&startDate=2025-01-01&endDate=2025-01-01&locationCategory=STATE&locationid=FIPS:06&dataTypes=PRCP,SNOW,SNWD,TMAX,TMIN,AWND,EVAP,WDFG,WDFM,WSFG,TSUN&format=csv
```
- `dataset=daily-summaries` → Fetches daily summary weather data.
- `startDate=2025-01-01&endDate=2025-01-01` → Gets data for January 1, 2025.
- `locationCategory=STATE&locationid=FIPS:06` → Retrieves data specifically for California (FIPS code: 06).
- `dataTypes=...` → Includes key weather elements such as:
    - `PRCP` (Precipitation)
    - `SNOW` (Snowfall)
    - `SNWD` (Snow Depth)
    - `TMAX` (Max Temperature)
    - `TMIN` (Min Temperature)
    - `AWND` (Average Wind Speed)
    - `EVAP` (Evaporation)
    - `WDFG`, WDFM, WSFG (Wind Gusts, Fastest Mile Wind)
    - `TSUN` (Sunshine duration)
- `format=csv` → Retrieves data in CSV format for easier processing.