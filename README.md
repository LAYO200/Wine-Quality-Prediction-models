# Wine-Quality-Prediction-Models

Machine learning project using the UCI Wine Quality dataset to predict wine quality scores through classification and regression models.

**Course:** CS 4120 – Machine Learning, Data Mining  
**Authors:** Chizurum Ewelike & 'Tomilayo Faloseyi  

---

##   Repository Structure

```text
data/               # Dataset folder (wine quality CSVs + data README)
mlruns/             # MLflow experiment tracking directory
models/             # Saved trained models (added during experimentation)
notebooks/          # Optional Jupyter notebooks for EDA
reports/            # Proposal, midpoint report, final report
src/                # All training, preprocessing, and evaluation scripts
requirements.txt    # Pinned package versions for reproducibility
README.md           # Setup and run instructions
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

Currently, the primary script is:
		src/midpoint_pipeline.py

Run it with:
		python src/midpoint_pipeline.py

This script performs:
- Loading and preprocessing the wine quality dataset
- Splitting into training and testing sets
- Training baseline models (e.g., Logistic Regression, Decision Tree)
- Evaluating model performance (accuracy, F1-score, RMSE, etc.)
- Logging parameters, metrics, and artifacts to MLflow

---

##   MLflow Experiment Tracking

The project uses MLflow to track experiments.

All experiment logs are stored in the mlruns/ directory, including:
- model parameters
- metrics (F1, RMSE, accuracy)
- saved models
- plots and other artifacts

To launch the MLflow UI:
		mlflow ui

Then open the URL printed in the terminal to inspect and compare runs.

---

##   Reports

The reports/ directory contains all written documents related to the project:
- proposal_Wine Quality — G2.pdf – Project proposal
- midpoint_WineQuality-G2.pdf – Midpoint progress report
- final_report.pdf – Final report 
