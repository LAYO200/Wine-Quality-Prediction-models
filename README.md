# Wine-Quality-Prediction-Models

Machine learning project using the UCI Wine Quality dataset to predict wine quality scores through classification and regression models.

**Course:** CS 4120 – Machine Learning, Data Mining  
**Authors:** Chizurum Ewelike & 'Tomilayo Faloseyi  

---

##   Repository Structure
```text
data/                       # Dataset folder (winequality-red.csv, winequality-white.csv + data README)
  └─ wine+quality/
mlruns/                     # MLflow experiment tracking directory (auto-created by MLflow)
models/                     # Saved trained models (optional, added during experimentation)
notebooks/                  # Optional Jupyter notebooks for EDA
reports/                    # Proposal, midpoint report, final report
src/                        # All training, preprocessing, and evaluation scripts
  ├─ data.py                # Data loading, cleaning, and train/val/test splitting
  ├─ utils.py               # Config constants, paths, and evaluation helpers
  ├─ features.py            # Preprocessing pipelines & feature engineering
  ├─ evaluate.py            # Plotting and final table generation
  ├─ train_baselines.py     # Classical baseline models + MLflow logging
  └─ train_nn.py            # Classical + NN models + final report artefacts and MLflow logging
outputs/                    # Generated plots, tables, and other artefacts (created by scripts)
requirements.txt            # Pinned package versions for reproducibility
README.md                   # Setup and run instructions
LICENSE
.gitignore

```
---

##   Installation & Setup

To reproduce this project:
- Ensure Python is installed on your system.
- Create and activate a virtual environment (this is optional).
  
Install the required packages:
		pip install -r requirements.txt

This installs pinned versions of:
numpy
pandas
scikit-learn
matplotlib
mlflow
These versions match the development environment, ensuring reproducibility.

---

##   Running the Project

All commands below assume you’re in the project root directory.
1. Classical Baselines Only
Script: src/train_baselines.py
Run:
python src/train_baselines.py
This script:
Loads and preprocesses the red + white wine datasets
Creates train/validation/test splits
Trains classical baseline models:
LogisticRegression (classification)
DecisionTreeClassifier (classification)
LinearRegression (regression)
DecisionTreeRegressor (regression)
Evaluates models on validation and test sets (Accuracy, F1, MAE, RMSE, etc.)
Logs parameters, metrics, and models to MLflow
Saves an EDA correlation heatmap into the outputs/ directory
2. Classical + Neural Networks + Final Artefacts
Script: src/train_nn.py
Run:
python src/train_nn.py
This script:
Repeats the full preprocessing and splitting pipeline
Trains the same classical baselines as above
Trains neural network models:
MLPClassifier for classification
MLPRegressor for regression
(with scaling, input “dropout”-style noise, and early stopping)
Compares NN vs classical models on validation performance and selects best final models
Generates and saves:
Learning curve plots for NN classifier & regressor
Confusion matrix for the best classifier
Residual plots for the best regressor
Feature importance plots (e.g. from tree-based models)
Final summary tables comparing classical vs NN performance
Logs all metrics, parameters, models, and artefacts to MLflow
Stores plots and tables in the outputs/ directory


---
##   MLflow Experiment Tracking

The project uses MLflow to track experiments.

All experiment logs are stored in the mlruns/ directory, including:
- model parameters
- metrics (val_f1, test_f1, val_rmse, test_rmse, etc)
- saved models
- plots and other artifacts(from train_baselines.py and train_nn.py)

To launch the MLflow UI:
		mlflow ui

Then open the URL printed in the terminal to inspect and compare runs.

---

##   Reports

The reports/ directory contains all written documents related to the project:
- proposal_Wine Quality — G2.pdf – Project proposal
- midpoint_WineQuality-G2.pdf – Midpoint progress report
- final_report.pdf – Final report 
