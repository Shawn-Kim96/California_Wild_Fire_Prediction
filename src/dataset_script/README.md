# NOAA's NCEI Climate Data Extraction

## Direct data download via Online Search

### Data Extraction
From [NOAA Climate Data Online Search](https://www.ncei.noaa.gov/cdo-web/search?datasetid=GHCND), you can select `Observation Type`, `Date Range`, `Search Term` and download the data.

To get CA Climate data, use `Daily Summaries` dataset and search for `State`, `California`.

The maximum dataset size per one download request is 1GB, which is about 4 months of data for daily summaries (when selecting all observation data).


## Data download via Web API v2 request
### Guildline document
- [NOAA Climate Data Online](https://www.ncdc.noaa.gov/cdo-web/webservices/v2#gettingStarted)

### Params for API
**Datatypes**
- From total [1566](https://www.ncei.noaa.gov/cdo-web/api/v2/datatypes) data, data that is included in GHCND dataset and date including after 2020-01-01 are [68](https://www.ncei.noaa.gov/cdo-web/api/v2/datatypes?datasetid=GHCND&startdate=2020-01-01&limit=80).
**LocationID**
- [California location id](https://www.ncei.noaa.gov/cdo-web/api/v2/locations?locationcategoryid=ST&offset=5&limit=1) = `FIPS:06`

### Data Extraction
- `src/dataset_script/extract_climate_data_via_webv2.py` is a script to extract dataset with WEB V2 API.
- However, there were errors while using API V2, so the code is deprecated.
