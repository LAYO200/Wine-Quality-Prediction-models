#%% md
# Midpoint — Wine Quality (interactive version)
# Shows 4 plots with plt.show(), prints 2 tables to console.


# Imports and basic setup
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

# Reproducibility: every split/model uses the same seed
random_seed = 42
np.random.seed(random_seed)


# Small helpers to keep code tidy
def metric_table(rows, columns):
    """Pretty convenience for building printed tables."""
    return pd.DataFrame(rows, columns=columns)

def standard_preprocessor(numeric_features):
    """
    Wrap a StandardScaler in a ColumnTransformer so we can reference
    columns by NAME (requires DataFrame inputs, not numpy arrays).
    """
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), numeric_features)],
        remainder="drop"
    )

def cls_metrics(model, Xv, yv, Xte, yte):
    """Classification metrics we actually care about: Accuracy + F1 on val/test."""
    pv = model.predict(Xv)
    pt = model.predict(Xte)
    return {
        "val_accuracy": accuracy_score(yv, pv),
        "val_f1": f1_score(yv, pv, zero_division=0),
        "test_accuracy": accuracy_score(yte, pt),
        "test_f1": f1_score(yte, pt, zero_division=0),
    }

def reg_metrics(model, Xv, yv, Xte, yte):
    """Regression metrics: MAE + RMSE on val/test."""
    pv = model.predict(Xv)
    pt = model.predict(Xte)
    return {
        "val_mae": float(mean_absolute_error(yv, pv)),
        "val_rmse": float(np.sqrt(mean_squared_error(yv, pv))),
        "test_mae": float(mean_absolute_error(yte, pt)),
        "test_rmse": float(np.sqrt(mean_squared_error(yte, pt))),
    }


# Load + clean the dataset
red_df = pd.read_csv("wine+quality/winequality-red.csv", sep=";")
red_df["type"] = "red"

white_df = pd.read_csv("wine+quality/winequality-white.csv", sep=";")
white_df["type"] = "white"

# Combine, de-dup, coerce numerics, drop rows with missing values
df = pd.concat([red_df, white_df], ignore_index=True).drop_duplicates()
for c in df.columns:
    if c != "type":
        df[c] = pd.to_numeric(df[c], errors="coerce")
df["quality"] = df["quality"].astype(int)
df["type"] = df["type"].astype("category")
df = df.dropna().reset_index(drop=True)

# Binary target for the classification task: 1 = “good” wine (>=7)
df["target_cls"] = (df["quality"] >= 7).astype(int)

# All numeric feature columns (exclude label + helper columns)
numeric_features = [c for c in df.columns if c not in ["quality", "target_cls", "type"]]


# Split: train/val/test using the rubric
# Keep X as a DataFrame (so transformers can use column names)
X = df[numeric_features]
y_cls = df["target_cls"].values
y_reg = df["quality"].values

# First split out the test set (20%); stratify on the class label
X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X, y_cls, y_reg, test_size=0.20, random_state=random_seed, stratify=y_cls
)

# Then carve out a validation set (25% of remaining = 20% overall)
X_train, X_val, y_cls_train, y_cls_val, y_reg_train, y_reg_val = train_test_split(
    X_train, y_cls_train, y_reg_train, test_size=0.25, random_state=random_seed, stratify=y_cls_train
)


#EDA — exactly two plots
# Plot 1: class balance (bar chart). Imbalance matters for F1 vs Accuracy.
plt.figure()
pd.Series(y_cls).value_counts().sort_index().plot(kind="bar")
plt.title("Plot 1 — Target Distribution (0=Low, 1=High)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Plot 2: correlation structure among features (+ quality).
# Gives a quick sense of which features drive quality (e.g., alcohol).
num_df = df[numeric_features + ["quality"]].select_dtypes(include=[np.number]).copy()

corr_df = num_df.corr(numeric_only=True)

# Drop rows/cols that are all NaN (can happen if a column is constant or non-numeric slipped in)
corr_df = corr_df.dropna(how="all", axis=0).dropna(how="all", axis=1)

if corr_df.shape[0] >= 2 and corr_df.shape[1] >= 2 and not np.isnan(corr_df.values).all():
    # Heatmap path
    plt.figure()
    mat = corr_df.values  # ensure numeric array for imshow
    plt.imshow(mat, vmin=-1, vmax=1)  # fix color scale for visibility
    plt.colorbar()
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90)
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.title("Plot 2 — Correlation Heatmap (Numeric + Quality)")
    plt.tight_layout()
    plt.show()
else:
    # Fallback: concise boxplot of the numeric features
    plt.figure()
    num_df.drop(columns=["quality"], errors="ignore").plot(kind="box", rot=90)
    plt.title("Plot 2 — Boxplot Summary (Fallback)")
    plt.tight_layout()
    plt.show()



#Baselines — classical models, no heroics
# Classification baselines:
# - Logistic Regression with scaling (sane default for tabular)
# - Decision Tree without scaling (trees don’t need it)
cls_logit = Pipeline([
    ("prep", standard_preprocessor(numeric_features)),
    ("clf", LogisticRegression(max_iter=200, random_state=random_seed))
])
cls_logit.fit(X_train, y_cls_train)
m_logit = cls_metrics(cls_logit, X_val, y_cls_val, X_test, y_cls_test)

cls_tree = Pipeline([
    ("prep", "passthrough"),
    ("clf", DecisionTreeClassifier(random_state=random_seed))
])
cls_tree.fit(X_train, y_cls_train)
m_tree = cls_metrics(cls_tree, X_val, y_cls_val, X_test, y_cls_test)

# Regression baselines:
# - Linear Regression with scaling (keeps coefficients honest)
# - Decision Tree Regressor (handles nonlinearity, risk of overfit)
reg_lin = Pipeline([
    ("prep", standard_preprocessor(numeric_features)),
    ("reg", LinearRegression())
])
reg_lin.fit(X_train, y_reg_train)
m_lin = reg_metrics(reg_lin, X_val, y_reg_val, X_test, y_reg_test)

reg_tree = Pipeline([
    ("prep", "passthrough"),
    ("reg", DecisionTreeRegressor(random_state=random_seed))
])
reg_tree.fit(X_train, y_reg_train)
m_rtree = reg_metrics(reg_tree, X_val, y_reg_val, X_test, y_reg_test)

# Print the two required tables directly to console
table1 = metric_table(
    rows=[
        ["LogisticRegression", m_logit["val_accuracy"], m_logit["val_f1"], m_logit["test_accuracy"], m_logit["test_f1"]],
        ["DecisionTreeClassifier", m_tree["val_accuracy"], m_tree["val_f1"], m_tree["test_accuracy"], m_tree["test_f1"]],
    ],
    columns=["Model", "Val_Accuracy", "Val_F1", "Test_Accuracy", "Test_F1"]
)
print("\n=== Table 1 — Classification metrics (Val/Test) ===")
print(table1.to_string(index=False))

table2 = metric_table(
    rows=[
        ["LinearRegression", m_lin["val_mae"], m_lin["val_rmse"], m_lin["test_mae"], m_lin["test_rmse"]],
        ["DecisionTreeRegressor", m_rtree["val_mae"], m_rtree["val_rmse"], m_rtree["test_mae"], m_rtree["test_rmse"]],
    ],
    columns=["Model", "Val_MAE", "Val_RMSE", "Test_MAE", "Test_RMSE"]
)
print("\n=== Table 2 — Regression metrics (Val/Test) ===")
print(table2.to_string(index=False))

# Plot 3 — Confusion matrix (best classifier by Val F1) on the test set
if m_logit["val_f1"] >= m_tree["val_f1"]:
    best_cls_model, best_cls_name = cls_logit, "LogisticRegression"
else:
    best_cls_model, best_cls_name = cls_tree, "DecisionTreeClassifier"

y_pred_test_cls = best_cls_model.predict(X_test)
cm = confusion_matrix(y_cls_test, y_pred_test_cls, labels=[0, 1])

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(values_format='d', ax=ax, colorbar=True)
ax.set_title(f"Plot 3 — Confusion Matrix (Test) — {best_cls_name}")
plt.tight_layout()
plt.show()

# Best regressor by Val RMSE → residuals vs predicted on the test set
if m_lin["val_rmse"] <= m_rtree["val_rmse"]:
    best_reg_model, best_reg_name = reg_lin, "LinearRegression"
else:
    best_reg_model, best_reg_name = reg_tree, "DecisionTreeRegressor"

y_pred_test_reg = best_reg_model.predict(X_test)
residuals = y_reg_test - y_pred_test_reg
plt.figure()
plt.scatter(y_pred_test_reg, residuals, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Quality (Test)")
plt.ylabel("Residuals (y - y_pred)")
plt.title(f"Plot 4 — Residuals vs Predicted (Test) — {best_reg_name}")
plt.tight_layout()
plt.show()
