"""
Wine Quality — Midpoint
Tasks: (1) Classification: high quality (>=7) vs low; (2) Regression: predict quality (3–9).
Rationale: UCI wine (red+white) supports both targets from the same features.
Split: stratified on classification label, 60/20/20 (train/val/test), seed=42 for reproducibility.
Model selection: choose best CLS by val F1 (imbalance); best REG by val RMSE.
Tracking: MLflow logs params, metrics, figures, and model artifacts (with input example).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error
)

# minimal MLflow
import mlflow
import mlflow.sklearn


# Load data
np.random.seed(42)

red_df = pd.read_csv("data/wine+quality/winequality-red.csv", sep=";")
white_df = pd.read_csv("data/wine+quality/winequality-white.csv", sep=";")

df = pd.concat([red_df.assign(type="red"), white_df.assign(type="white")],
               ignore_index=True).drop_duplicates()

# numeric coercion, drop missing, simple dtypes
for c in df.columns:
    if c != "type":
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna().reset_index(drop=True)
df["quality"] = df["quality"].astype(int)
df["type"] = df["type"].astype("category")

# binary label for classification
df["target_cls"] = (df["quality"] >= 7).astype(int)

# features (exclude labels/helper columns)
features = [c for c in df.columns if c not in ["quality", "target_cls", "type"]]
X = df[features].astype("float64")  # FIX: use floats so MLflow schema is stable
y_cls = df["target_cls"].values
y_reg = df["quality"].values

# Split: train / val / test
X_tr, X_te, y_cls_tr, y_cls_te, y_reg_tr, y_reg_te = train_test_split(
    X, y_cls, y_reg, test_size=0.20, random_state=42, stratify=y_cls
)

X_tr, X_va, y_cls_tr, y_cls_va, y_reg_tr, y_reg_va = train_test_split(
    X_tr, y_cls_tr, y_reg_tr, test_size=0.25, random_state=42, stratify=y_cls_tr
)

# scaler in a simple ColumnTransformer
scaler = ColumnTransformer([("scale", StandardScaler(), features)], remainder="drop")

# Plot 1: target distribution

plt.figure()
pd.Series(y_cls).value_counts().sort_index().plot(kind="bar")
plt.title("Plot 1 — Target Distribution (0=Low, 1=High)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plot1_target_distribution.png", dpi=200, bbox_inches="tight")
plt.show()


# Plot 2: correlation matrix

num_df = df[features + ["quality"]].select_dtypes(include=[np.number])
corr = num_df.corr(numeric_only=True)

plt.figure()
plt.matshow(corr, fignum=False)          # simple matrix view, default style
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Plot 2 — Correlation Matrix")
plt.tight_layout()
plt.savefig("plot2_correlation_matrix.png")
plt.show()


# Baselines
# classification
logit = Pipeline([("prep", scaler), ("clf", LogisticRegression(max_iter=200, random_state=42))])
logit.fit(X_tr, y_cls_tr)

tree_cls = Pipeline([("prep", "passthrough"), ("clf", DecisionTreeClassifier(random_state=42))])
tree_cls.fit(X_tr, y_cls_tr)

# regression
linreg = Pipeline([("prep", scaler), ("reg", LinearRegression())])
linreg.fit(X_tr, y_reg_tr)

tree_reg = Pipeline([("prep", "passthrough"), ("reg", DecisionTreeRegressor(random_state=42))])
tree_reg.fit(X_tr, y_reg_tr)


# Tables: metrics on val + test
def cls_eval(model, Xv, yv, Xte, yte):
    pv = model.predict(Xv)
    pt = model.predict(Xte)
    return {
        "val_accuracy": accuracy_score(yv, pv),
        "val_f1": f1_score(yv, pv, zero_division=0),
        "test_accuracy": accuracy_score(yte, pt),
        "test_f1": f1_score(yte, pt, zero_division=0),
    }

def reg_eval(model, Xv, yv, Xte, yte):
    pv = model.predict(Xv)
    pt = model.predict(Xte)
    return {
        "val_mae": float(mean_absolute_error(yv, pv)),
        "val_rmse": float(np.sqrt(mean_squared_error(yv, pv))),
        "test_mae": float(mean_absolute_error(yte, pt)),
        "test_rmse": float(np.sqrt(mean_squared_error(yte, pt))),
    }

m_logit = cls_eval(logit, X_va, y_cls_va, X_te, y_cls_te)
m_treec = cls_eval(tree_cls, X_va, y_cls_va, X_te, y_cls_te)

m_linr = reg_eval(linreg, X_va, y_reg_va, X_te, y_reg_te)
m_treer = reg_eval(tree_reg, X_va, y_reg_va, X_te, y_reg_te)

table_cls = pd.DataFrame([
    ["LogisticRegression", m_logit["val_accuracy"], m_logit["val_f1"], m_logit["test_accuracy"], m_logit["test_f1"]],
    ["DecisionTreeClassifier", m_treec["val_accuracy"], m_treec["val_f1"], m_treec["test_accuracy"], m_treec["test_f1"]],
], columns=["Model", "Val_Accuracy", "Val_F1", "Test_Accuracy", "Test_F1"])

table_reg = pd.DataFrame([
    ["LinearRegression", m_linr["val_mae"], m_linr["val_rmse"], m_linr["test_mae"], m_linr["test_rmse"]],
    ["DecisionTreeRegressor", m_treer["val_mae"], m_treer["val_rmse"], m_treer["test_mae"], m_treer["test_rmse"]],
], columns=["Model", "Val_MAE", "Val_RMSE", "Test_MAE", "Test_RMSE"])

print("\n=== Table 1 — Classification metrics (Val/Test) ===")
print(table_cls.to_string(index=False))
print("\n=== Table 2 — Regression metrics (Val/Test) ===")
print(table_reg.to_string(index=False))

table_cls.to_csv("table1_classification_metrics.csv", index=False)
table_reg.to_csv("table2_regression_metrics.csv", index=False)


# Plot 3: confusion matrix
best_cls_model = logit if m_logit["val_f1"] >= m_treec["val_f1"] else tree_cls
best_cls_name = "LogisticRegression" if best_cls_model is logit else "DecisionTreeClassifier"

y_pred_cls_test = best_cls_model.predict(X_te)
cm = confusion_matrix(y_cls_te, y_pred_cls_test, labels=[0, 1])

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()                                # defaults (no colorbar, simple look)
plt.title(f"Plot 3 — Confusion Matrix (Test) — {best_cls_name}")
plt.tight_layout()
plt.savefig("plot3_confusion_matrix.png")
plt.show()


# Plot 4: residuals vs predicted
best_reg_model = linreg if m_linr["val_rmse"] <= m_treer["val_rmse"] else tree_reg
best_reg_name = "LinearRegression" if best_reg_model is linreg else "DecisionTreeRegressor"

y_pred_reg_test = best_reg_model.predict(X_te)
residuals = y_reg_te - y_pred_reg_test

plt.figure()
plt.scatter(y_pred_reg_test, residuals, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Quality (Test)")
plt.ylabel("Residuals (y - y_pred)")
plt.title(f"Plot 4 — Residuals vs Predicted (Test) — {best_reg_name}")
plt.tight_layout()
plt.savefig("plot4_residuals.png", dpi=200, bbox_inches="tight")
plt.show()


# Minimal MLflow logging
EXAMPLE_X = X_tr.head(5).astype("float64")  # FIX: small float sample for signature inference
mlflow.set_experiment("wine-quality-midpoint")

common = {
    "seed": 42,
    "test_pct": 0.20,
    "val_pct_of_train": 0.25,
    "n_features": len(features),
    "stratify_on": "target_cls",
}

# classification runs
with mlflow.start_run(run_name="CLS - LogisticRegression"):
    mlflow.log_params(common)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metrics(m_logit)
    mlflow.log_artifact("plot1_target_distribution.png")
    mlflow.log_artifact("plot2_correlation_matrix.png")  # FIX: match saved filename
    mlflow.log_artifact("plot3_confusion_matrix.png")
    mlflow.log_artifact("table1_classification_metrics.csv")
    mlflow.sklearn.log_model(logit, name="model", input_example=EXAMPLE_X)  # FIX: name= + input_example

with mlflow.start_run(run_name="CLS - DecisionTreeClassifier"):
    mlflow.log_params(common)
    mlflow.log_param("model", "DecisionTreeClassifier")
    mlflow.log_metrics(m_treec)
    mlflow.log_artifact("plot1_target_distribution.png")
    mlflow.log_artifact("plot2_correlation_matrix.png")  # FIX: match saved filename
    mlflow.log_artifact("plot3_confusion_matrix.png")
    mlflow.log_artifact("table1_classification_metrics.csv")
    mlflow.sklearn.log_model(tree_cls, name="model", input_example=EXAMPLE_X)  # FIX

# regression runs
with mlflow.start_run(run_name="REG - LinearRegression"):
    mlflow.log_params(common)
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metrics(m_linr)
    mlflow.log_artifact("plot4_residuals.png")
    mlflow.log_artifact("table2_regression_metrics.csv")
    mlflow.sklearn.log_model(linreg, name="model", input_example=EXAMPLE_X)  # FIX

with mlflow.start_run(run_name="REG - DecisionTreeRegressor"):
    mlflow.log_params(common)
    mlflow.log_param("model", "DecisionTreeRegressor")
    mlflow.log_metrics(m_treer)
    mlflow.log_artifact("plot4_residuals.png")
    mlflow.log_artifact("table2_regression_metrics.csv")
    mlflow.sklearn.log_model(tree_reg, name="model", input_example=EXAMPLE_X)  # FIX
