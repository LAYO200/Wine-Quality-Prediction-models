# classical ML training for both tasks

import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import mlflow
import mlflow.sklearn

from utils import (
    SEED,
    OUTPUT_DIR,
    EXPERIMENT_NAME,
    cls_eval,
    reg_eval,
)
from data import load_wine_data, make_splits
from features import make_classical_preprocessor, plot_correlation_heatmap


def main():
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df, features = load_wine_data()
    (
        X_tr,
        X_va,
        X_te,
        y_cls_tr,
        y_cls_va,
        y_cls_te,
        y_reg_tr,
        y_reg_va,
        y_reg_te,
    ) = make_splits(df, features, seed=SEED)

    scaler_ct = make_classical_preprocessor(features)
    eda_corr_path = plot_correlation_heatmap(df, features, output_dir=OUTPUT_DIR)

    # classification
    logit = Pipeline(
        [("prep", scaler_ct), ("clf", LogisticRegression(max_iter=500, random_state=SEED))]
    )
    logit.fit(X_tr, y_cls_tr)

    tree_cls = Pipeline(
        [("prep", "passthrough"), ("clf", DecisionTreeClassifier(random_state=SEED))]
    )
    tree_cls.fit(X_tr, y_cls_tr)

    # regression
    linreg = Pipeline([("prep", scaler_ct), ("reg", LinearRegression())])
    linreg.fit(X_tr, y_reg_tr)

    tree_reg = Pipeline(
        [("prep", "passthrough"), ("reg", DecisionTreeRegressor(random_state=SEED))]
    )
    tree_reg.fit(X_tr, y_reg_tr)

    # metrics
    m_logit = cls_eval(logit, X_va, y_cls_va, X_te, y_cls_te)
    m_treec = cls_eval(tree_cls, X_va, y_cls_va, X_te, y_cls_te)

    m_linr = reg_eval(linreg, X_va, y_reg_va, X_te, y_reg_te)
    m_treer = reg_eval(tree_reg, X_va, y_reg_va, X_te, y_reg_te)

    print("\n=== Baseline — Classification ===")
    print("LogisticRegression:", m_logit)
    print("DecisionTreeClassifier:", m_treec)

    print("\n=== Baseline — Regression ===")
    print("LinearRegression:", m_linr)
    print("DecisionTreeRegressor:", m_treer)

    # MLflow logging (classical only)
    mlflow.set_experiment(EXPERIMENT_NAME)
    EXAMPLE_X = X_tr.head(5).astype("float64")

    common_params = {
        "seed": SEED,
        "test_pct": 0.20,
        "val_pct_of_train": 0.25,
        "n_features": len(features),
        "stratify_on": "target_cls",
    }

    with mlflow.start_run(run_name="CLS - LogisticRegression (baselines)"):
        mlflow.log_params(common_params)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metrics(m_logit)
        mlflow.sklearn.log_model(logit, name="model", input_example=EXAMPLE_X)

    with mlflow.start_run(run_name="CLS - DecisionTreeClassifier (baselines)"):
        mlflow.log_params(common_params)
        mlflow.log_param("model", "DecisionTreeClassifier")
        mlflow.log_metrics(m_treec)
        mlflow.sklearn.log_model(tree_cls, name="model", input_example=EXAMPLE_X)

    with mlflow.start_run(run_name="REG - LinearRegression (baselines)"):
        mlflow.log_params(common_params)
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metrics(m_linr)
        mlflow.sklearn.log_model(linreg, name="model", input_example=EXAMPLE_X)

    with mlflow.start_run(run_name="REG - DecisionTreeRegressor (baselines)"):
        mlflow.log_params(common_params)
        mlflow.log_param("model", "DecisionTreeRegressor")
        mlflow.log_metrics(m_treer)
        mlflow.sklearn.log_model(tree_reg, name="model", input_example=EXAMPLE_X)

    print(f"\nEDA correlation heatmap saved at: {eda_corr_path}")
    print("Baseline training finished.")


if __name__ == "__main__":
    main()
