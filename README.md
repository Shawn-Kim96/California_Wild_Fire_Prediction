# Wildfire Risk Prediction Model

## Project Summary

The Wildfire Risk Prediction Model is a machine learning project designed to predict wildfire occurrences across different regions in California. The model leverages historical wildfire data, meteorological conditions, and environmental indicators to estimate wildfire probability. By providing predictive insights, the model aims to enhance early warning systems, optimize resource allocation, and improve wildfire prevention strategies.

## Project Directory Structure

```
├── notebooks       # Jupyter notebooks for experiments
│   ├── 1.0-data-prep.ipynb
│   ├── 1.1-eda.ipynb
│   ├── 2.0-model-training.ipynb
│   ├── 3.0-model-evaluation.ipynb
│   ├── 4.0-visualization.ipynb
│
├── data            # Raw and processed datasets
│   ├── raw/
│   ├── processed/
│
├── models          # Saved models and checkpoints
│
├── src             # Python source scripts for automation
│   ├── data_extraction.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│
├── reports         # Reports and visualizations
│
├── config          # Configuration files
│   ├── hyperparams.yaml
│
└── README.md       # Project documentation
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

## Task Distribution

All team members will collaborate on all major tasks, ensuring a balanced workload. Specific task responsibilities may vary but will include:

- **Data Preparation:** Automating extraction, handling missing values, and feature engineering.
- **Model Development:** Experimenting with different algorithms and architectures.
- **Implementation & Validation:** Setting up environments, training, and testing models.
- **Evaluation & Visualization:** Interpreting results, assessing performance, and generating visual reports.
- **Documentation & Reporting:** Writing reports and preparing presentations.

## Technologies Used

- **Programming Language:** Python
- **Machine Learning Libraries:** scikit-learn, TensorFlow/PyTorch, XGBoost
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Cloud & Storage:** Google Drive
- **Version Control:** GitHub
