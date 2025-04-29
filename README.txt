# California Wildfire Risk Prediction Model

## Project Summary

The Wildfire Risk Prediction Model aims to predict wildfire occurrences across different regions in California by leveraging meteorological data, land characteristics, and environmental risk factors.

The goal is to enhance early warning systems, improve wildfire prevention, and optimize resource allocation.

[Github Repository Link](https://github.com/Shawn-Kim96/California_Wild_Fire_Prediction)

## Project Directory Structure

```bash
├── data/                                   # Raw, processed, and engineered datasets
│   ├── calfire_cimis_data/                 # Weather data for wildfire incidents
│   ├── calfire_cimis_non_wildfire_data/    # Weather data for non-wildfire days
│   ├── featured_data/                      # Outputs from feature engineering experiments
│   ├── final_data/                         # Final combined dataset used for model training
│   ├── landfire_data/                      # Landcover and vegetation data
│   ├── model_results/                      # Results from model training and evaluation
│   └── risk_data/                          # Processed datasets including risk-related features
├── models/                                 # Best model saved
├── notebooks/                              # Data analysis, feature engineering, model training notebooks
├── src/                                    # Source code for preprocessing, feature engineering, training
│   ├── dataset_extract/                    # Scripts to extract external data (NOAA, CALFIRE)
│   ├── dataset_preprocess/                 # Preprocessing and feature engineering functions
│   └── train_model.py                      # Model training pipeline
├── requirements.txt                        # Project dependencies
├── pyproject.toml / poetry.lock            # Python environment and package manager settings
├── zipcodes_by_county.json                 # Auxiliary mapping files
└── README.md                               # Project documentation (this file)
```

## How to Run
### Environment Setup
- Python version: 3.11.5
- Dependency installation
```bash
pip install poetry
poetry install
poetry shell  # initialize virtual environment
```

### Data Collecting
```bash
python src/dataset_extract/extract_climate_data_via_webv2.py  # collects climate data
python src/dataset_extract/add_landfire_data.py  # collects land data

# for risk data, should execute notebooks/4.0-HashemJaber...ipynb code
```

### Data Preprocessing & Feature Engineering
- Data preprocessing python script: `src/dataset_preprocess/preprocess_functions.py`
- Feature Engineering python functions: `src/dataset_preprocess/feature_engineering_functions.py`
- Example of feature engineering and data preprocessing usage in notebook: `notebooks/11.0-RodrigoShawn-test_feature_engineering.ipynb`

### Model Training
- Model training python script: `src/train_model.py`
- Example of model training in notebook: `notebooks/11.0-RodrigoShawn-test_feature_engineering.ipynb`

### Data and Model Analysis
- Example of data and model analysis in notebooks: `notebooks/8.0-ShawnKim-design_prediction_model`, `notebooks/12.0-ShawnKim-final_data_analysis`


## Highlights
- Datasets from multiple trusted sources (NOAA, CAL FIRE, LANDFIRE).

- Extensive feature engineering pipelines.

- Multiple model evaluations (Logistic Regression, Decision Tree, Gradient Boosting, XGBoost, Ensemble Voting Classifier).

- Focus on minimizing False Negatives to prevent undetected wildfires.


## Project Management Links

- **Jira Board:** [SJSU CMPE 257 Team Jira](https://sjsu-cmpe257-team.atlassian.net/jira/software/projects/SCRUM/boards/1?sprintStarted=true)
- **Google Drive:** [Project File Storage](https://drive.google.com/drive/u/0/folders/1vlctgZOaWCY1flVBnbs55Wa2WiuBbiED)

## Timelines


|Phase | Timeline | Details|
|---|---|---|
|Phase 1 | Jan - Feb 2025 | Project Proposal & Planning|
|Phase 2 | Feb - Mar 2025 | Data Collection & Preprocessing|
|Phase 3 | Mar - Apr 2025 | Model Development & Training|
|Phase 4 | Apr - Apr 2025 | Model Evaluation & Refinement|
|Phase 5 | Apr 2025 | Deployment & Final Report|


## Git Workflow & Commit Message Convention

### Branching Strategy

- `main`: Stable and production-ready code.

- `feature/{jira-issue-name}`: New features are developed here before merging into develop.

- `bugfix/{jira-issue-name}`: Bug fixes.

- `hotfix/{jira-issue-name}`: Urgent bug fixes that need to be patched into main.

### Commit Message Format
```csharp
[type]: [short description]

[Optional detailed description]
```

**Commit Types**:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation updates
- `refactor`: Code restructuring without functionality changes
- `chore`: Maintenance tasks (build processes, dependency updates, etc.)
