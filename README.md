# Wildfire Risk Prediction Model

## Project Summary

The Wildfire Risk Prediction Model is a machine learning project designed to predict wildfire occurrences across different regions in California. The model leverages historical wildfire data, meteorological conditions, and environmental indicators to estimate wildfire probability. By providing predictive insights, the model aims to enhance early warning systems, optimize resource allocation, and improve wildfire prevention strategies.

## Project Directory Structure

```
├── README.md
├── README_TREE.txt
├── data
│   ├── calfire_cimis_data
│   │   ├── finalized
│   │   │   ├── calfire_cimis_all_rows.csv
│   │   │   └── calfire_cimis_completed_rows.csv
│   │   └── processed
│   │       ├── calfire_cimis_merged.csv
│   │       └── rows_with_incompletes.csv
│   ├── calfire_cimis_non_wildfire_data
│   │   ├── finalized
│   │   │   └── cleaned_non_wildfire_dates.csv
│   │   └── processed
│   │       ├── non_wildfire_dates.csv
│   │       ├── processed_non_wildfire_dates.csv
│   │       ├── processed_non_wildfire_dates_0_100.csv
│   │       ├── processed_non_wildfire_dates_1000_1100.csv
│   │       ├── processed_non_wildfire_dates_100_200.csv
│   │       ├── processed_non_wildfire_dates_1100_1200.csv
│   │       ├── processed_non_wildfire_dates_1200_1300.csv
│   │       ├── processed_non_wildfire_dates_1300_1400.csv
│   │       ├── processed_non_wildfire_dates_1400_1500.csv
│   │       ├── processed_non_wildfire_dates_1500_1600.csv
│   │       ├── processed_non_wildfire_dates_1600_1700.csv
│   │       ├── processed_non_wildfire_dates_1700_1800.csv
│   │       ├── processed_non_wildfire_dates_1800_1900.csv
│   │       ├── processed_non_wildfire_dates_1900_2000.csv
│   │       ├── processed_non_wildfire_dates_2000_2100.csv
│   │       ├── processed_non_wildfire_dates_200_300.csv
│   │       ├── processed_non_wildfire_dates_2100_2200.csv
│   │       ├── processed_non_wildfire_dates_2200_2300.csv
│   │       ├── processed_non_wildfire_dates_2300_2400.csv
│   │       ├── processed_non_wildfire_dates_2400_2500.csv
│   │       ├── processed_non_wildfire_dates_2500_2600.csv
│   │       ├── processed_non_wildfire_dates_2600_2700.csv
│   │       ├── processed_non_wildfire_dates_2700_2800.csv
│   │       ├── processed_non_wildfire_dates_2800_2900.csv
│   │       ├── processed_non_wildfire_dates_2900_3000.csv
│   │       ├── processed_non_wildfire_dates_3000_3100.csv
│   │       ├── processed_non_wildfire_dates_300_400.csv
│   │       ├── processed_non_wildfire_dates_400_500.csv
│   │       ├── processed_non_wildfire_dates_500_600.csv
│   │       ├── processed_non_wildfire_dates_600_700.csv
│   │       ├── processed_non_wildfire_dates_700_800.csv
│   │       ├── processed_non_wildfire_dates_800_900.csv
│   │       ├── processed_non_wildfire_dates_900_1000.csv
│   │       └── wildfire_records(debug).csv
│   ├── final_data
│   │   └── total_data.csv
│   ├── landfire_data
│   │   ├── finalized
│   │   │   └── calfire_landfire_cimis_merged.csv
│   │   └── processed
│   │       ├── cbd_data.csv
│   │       ├── evc_data.csv
│   │       ├── fbfm_data.csv
│   │       ├── fdist_data.csv
│   │       └── fvc_data.csv
│   └── risk_data
│       └── finalized
│           ├── calfire_cimis_completed_rows_new_with_features.csv
│           └── calfire_cimis_completed_rows_new_with_features_no_wildfire.csv
├── notebooks
│   ├── 1.0-ShawnKim-analyze_climate_data.ipynb
│   ├── 2.0-ShawnKim-analyze_fuel_data.ipynb
│   ├── 3.0-AnthonyLuu-analyze_drought_fire_condition_data.ipynb
│   ├── 4.0-HashemJaber-add_risk_data.ipynb
│   ├── 5.0-Rodrigo-non_wildfire_cleaning.ipynb
│   ├── 6.0-ShanwKim-concat_and_clean_data.ipynb
│   └── 7.0-HashemJaber-Add-predictions.ipynb
├── poetry.lock
├── pyproject.toml
├── requirements.txt
├── src
│   └── dataset_script
│       ├── README.md
│       ├── add_landfire_data.py
│       ├── extract_climate_data_from_noaa.py
│       ├── extract_non_wildfire_incident_with_climate.py
│       └── extract_wildfire_with_climate_data.py
├── tools
│   ├── filling_nan_models
│   │   ├── model_fill_null_for_DayPrecip11.pth
│   │   ├── model_fill_null_for_DayPrecip14.pth
│   │   ├── model_fill_null_for_DayRelHumAvg01.pth
│   │   ├── model_fill_null_for_DayRelHumAvg02.pth
│   │   ├── model_fill_null_for_DayRelHumAvg03.pth
│   │   ├── model_fill_null_for_DayRelHumAvg04.pth
│   │   ├── model_fill_null_for_DayRelHumAvg05.pth
│   │   ├── model_fill_null_for_DayRelHumAvg06.pth
│   │   ├── model_fill_null_for_DayRelHumAvg07.pth
│   │   ├── model_fill_null_for_DayRelHumAvg08.pth
│   │   ├── model_fill_null_for_DayRelHumAvg09.pth
│   │   ├── model_fill_null_for_DayRelHumAvg10.pth
│   │   ├── model_fill_null_for_DayRelHumAvg11.pth
│   │   ├── model_fill_null_for_DayRelHumAvg12.pth
│   │   ├── model_fill_null_for_DayRelHumAvg13.pth
│   │   ├── model_fill_null_for_DayRelHumAvg14.pth
│   │   ├── model_fill_null_for_DaySoilTmpAvg01.pth
│   │   ├── model_fill_null_for_DaySoilTmpAvg02.pth
│   │   ├── model_fill_null_for_DaySoilTmpAvg03.pth
│   │   └── model_fill_null_for_DaySoilTmpAvg04.pth
│   └── tools.py
├── zipcodes_by_county.json
└── zipcodes_by_longlat.json

```

## Project Management Links

- **Jira Board:** [SJSU CMPE 257 Team Jira](https://sjsu-cmpe257-team.atlassian.net/jira/software/projects/SCRUM/boards/1?sprintStarted=true)
- **Google Drive:** [Project File Storage](https://drive.google.com/drive/u/0/folders/1vlctgZOaWCY1flVBnbs55Wa2WiuBbiED)

## Timelines

### **Phase 1: Project Proposal & Planning** (Jan - Feb 2025)
- Define project objectives, dataset selection, and preliminary research.
- Complete and submit the initial project report.

### **Phase 2: Data Collection & Preprocessing** (Feb - Mar 2025)
- Automate data extraction from external sources (NOAA, CAL FIRE, WRCC, MODIS).
- Identify missing values, outliers, and perform normalization.

### **Phase 3: Model Development & Training** (Mar - Apr 2025)
- Implement baseline models (Linear Regression, Decision Trees, etc.).
- Experiment with different feature engineering techniques.
- Optimize hyperparameters and evaluate model performance.

### **Phase 4: Model Evaluation & Refinement** (Apr - May 2025)
- Validate model using train-test split and k-fold cross-validation.
- Assess performance using metrics like Accuracy, F1-score, and ROC-AUC.
- Improve the model based on evaluation results.

### **Phase 5: Deployment & Report Submission** (May 2025)
- Prepare final project presentation.
- Visualize results and complete final report.
- Submit findings and model analysis.

## Git Workflow & Commit Message Convention

### Branching Strategy

- `main`: Stable and production-ready code.

- `feature/{jira-issue-name}`: New features are developed here before merging into develop.

- `bugfix/{jira-issue-name}`: Fixes for bugs found in develop.

- `hotfix/{jira-issue-name}`: Urgent bug fixes that need to be patched into main.

### Commit Message Format

To maintain consistency and readability, we follow this structured format:
```
[type]: [short description]

[Optional detailed description]
```

**Commit Types**:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation updates
- `refactor`: Code restructuring without functionality changes
- `chore`: Maintenance tasks (build processes, dependency updates, etc.)
