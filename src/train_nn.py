# NN training for both tasks
import os
import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

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
from features import (
    make_classical_preprocessor,
    make_nn_scaled,
    plot_correlation_heatmap,
)
from evaluate import (
    plot_nn_learning_curves,
    plot_confusion_matrix_best,
    plot_residuals_best,
    plot_feature_importances,
    make_final_tables,
)


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

X = df[features].astype("float64")

scaler_ct = make_classical_preprocessor(features)
scaler_nn, X_tr_nn, X_va_nn, X_te_nn = make_nn_scaled(X_tr, X_va, X_te)
input_dim = X_tr_nn.shape[1]

eda_corr_path = plot_correlation_heatmap(df, features, output_dir=OUTPUT_DIR)

logit = Pipeline(
    [("prep", scaler_ct), ("clf", LogisticRegression(max_iter=500, random_state=SEED))]
)
logit.fit(X_tr, y_cls_tr)

tree_cls = Pipeline(
    [("prep", "passthrough"), ("clf", DecisionTreeClassifier(random_state=SEED))]
)
tree_cls.fit(X_tr, y_cls_tr)

linreg = Pipeline([("prep", scaler_ct), ("reg", LinearRegression())])
linreg.fit(X_tr, y_reg_tr)

tree_reg = Pipeline(
    [("prep", "passthrough"), ("reg", DecisionTreeRegressor(random_state=SEED))]
)
tree_reg.fit(X_tr, y_reg_tr)

m_logit = cls_eval(logit, X_va, y_cls_va, X_te, y_cls_te)
m_treec = cls_eval(tree_cls, X_va, y_cls_va, X_te, y_cls_te)

m_linr = reg_eval(linreg, X_va, y_reg_va, X_te, y_reg_te)
m_treer = reg_eval(tree_reg, X_va, y_reg_va, X_te, y_reg_te)

best_classical_cls = logit if m_logit["val_f1"] >= m_treec["val_f1"] else tree_cls
best_classical_cls_name = (
    "LogisticRegression" if best_classical_cls is logit else "DecisionTreeClassifier"
)
best_classical_cls_metrics = m_logit if best_classical_cls is logit else m_treec

best_classical_reg = linreg if m_linr["val_rmse"] <= m_treer["val_rmse"] else tree_reg
best_classical_reg_name = (
    "LinearRegression" if best_classical_reg is linreg else "DecisionTreeRegressor"
)
best_classical_reg_metrics = m_linr if best_classical_reg is linreg else m_treer

def train_mlp_classifier(
    X_tr,
    y_tr,
    X_va,
    y_va,
    hidden_layer_sizes=(64, 32),
    max_epochs=100,
    lr=1e-3,
    dropout_p=0.3,
    patience=10,
):

    np.random.seed(SEED)
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        max_iter=1,
        warm_start=True,
        random_state=SEED,
    )

    train_accs, val_accs = [], []
    best_val_f1 = -np.inf
    best_epoch = 0
    best_model = None
    epochs_no_improve = 0
    classes = np.unique(y_tr)

    for epoch in range(max_epochs):
        X_epoch = X_tr.copy()
        if dropout_p > 0:
            mask = np.random.binomial(1, 1 - dropout_p, size=X_epoch.shape)
            X_epoch = X_epoch * mask

        if epoch == 0:
            model.partial_fit(X_epoch, y_tr, classes=classes)
        else:
            model.partial_fit(X_epoch, y_tr)

        y_tr_pred = model.predict(X_tr)
        y_va_pred = model.predict(X_va)

        train_acc = accuracy_score(y_tr, y_tr_pred)
        val_acc = accuracy_score(y_va, y_va_pred)
        val_f1 = f1_score(y_va, y_va_pred, zero_division=0)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_model is None:
        best_model = model

    history = {
        "train_acc": train_accs,
        "val_acc": val_accs,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
    }
    return best_model, history


def train_mlp_regressor(
    X_tr,
    y_tr,
    X_va,
    y_va,
    hidden_layer_sizes=(64, 32),
    max_epochs=100,
    lr=1e-3,
    dropout_p=0.3,
    patience=10,
):

    np.random.seed(SEED)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        max_iter=1,
        warm_start=True,
        random_state=SEED,
    )

    train_losses, val_losses = [], []
    best_val_rmse = np.inf
    best_epoch = 0
    best_model = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        X_epoch = X_tr.copy()
        if dropout_p > 0:
            mask = np.random.binomial(1, 1 - dropout_p, size=X_epoch.shape)
            X_epoch = X_epoch * mask

        model.partial_fit(X_epoch, y_tr)

        y_tr_pred = model.predict(X_tr)
        y_va_pred = model.predict(X_va)

        train_mse = mean_squared_error(y_tr, y_tr_pred)
        val_mse = mean_squared_error(y_va, y_va_pred)
        train_losses.append(train_mse)
        val_losses.append(val_mse)

        val_rmse = np.sqrt(val_mse)

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_model is None:
        best_model = model

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
    }
    return best_model, history


NN_HIDDEN = (64, 32)
NN_LR = 1e-3
NN_EPOCHS = 100
NN_DROPOUT_P = 0.3
NN_PATIENCE = 10

nn_cls, history_cls = train_mlp_classifier(
    X_tr_nn,
    y_cls_tr,
    X_va_nn,
    y_cls_va,
    hidden_layer_sizes=NN_HIDDEN,
    max_epochs=NN_EPOCHS,
    lr=NN_LR,
    dropout_p=NN_DROPOUT_P,
    patience=NN_PATIENCE,
)

nn_reg, history_reg = train_mlp_regressor(
    X_tr_nn,
    y_reg_tr,
    X_va_nn,
    y_reg_va,
    hidden_layer_sizes=NN_HIDDEN,
    max_epochs=NN_EPOCHS,
    lr=NN_LR,
    dropout_p=NN_DROPOUT_P,
    patience=NN_PATIENCE,
)


plot1_path, plot2_path = plot_nn_learning_curves(
    history_cls, history_reg, output_dir=OUTPUT_DIR
)

m_nn_cls = cls_eval(nn_cls, X_va_nn, y_cls_va, X_te_nn, y_cls_te)
m_nn_reg = reg_eval(nn_reg, X_va_nn, y_reg_va, X_te_nn, y_reg_te)

if m_nn_cls["val_f1"] >= best_classical_cls_metrics["val_f1"]:
    best_final_cls_model = nn_cls
    best_final_cls_type = "nn"
    best_final_cls_name = "MLPClassifier"
    best_final_cls_metrics = m_nn_cls
else:
    best_final_cls_model = best_classical_cls
    best_final_cls_type = "classical"
    best_final_cls_name = best_classical_cls_name
    best_final_cls_metrics = best_classical_cls_metrics

if m_nn_reg["val_rmse"] <= best_classical_reg_metrics["val_rmse"]:
    best_final_reg_model = nn_reg
    best_final_reg_type = "nn"
    best_final_reg_name = "MLPRegressor"
    best_final_reg_metrics = m_nn_reg
else:
    best_final_reg_model = best_classical_reg
    best_final_reg_type = "classical"
    best_final_reg_name = best_classical_reg_name
    best_final_reg_metrics = best_classical_reg_metrics

plot3_path = plot_confusion_matrix_best(
    best_final_cls_model,
    best_final_cls_type,
    best_final_cls_name,
    X_te,
    X_te_nn,
    y_cls_te,
    output_dir=OUTPUT_DIR,
)

plot4_path = plot_residuals_best(
    best_final_reg_model,
    best_final_reg_type,
    best_final_reg_name,
    X_te,
    X_te_nn,
    y_reg_te,
    output_dir=OUTPUT_DIR,
)

plot5_path = plot_feature_importances(tree_cls, X, output_dir=OUTPUT_DIR)

table1_path, table2_path = make_final_tables(
    best_classical_cls_name,
    best_classical_cls_metrics,
    best_classical_reg_name,
    best_classical_reg_metrics,
    m_nn_cls,
    m_nn_reg,
    output_dir=OUTPUT_DIR,
)


EXAMPLE_X = X_tr.head(5).astype("float64")

mlflow.set_experiment(EXPERIMENT_NAME)

common_params = {
    "seed": SEED,
    "test_pct": 0.20,
    "val_pct_of_train": 0.25,
    "n_features": len(features),
    "stratify_on": "target_cls",
}

with mlflow.start_run(run_name="CLS - LogisticRegression"):
    mlflow.log_params(common_params)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metrics(m_logit)
    mlflow.sklearn.log_model(logit, name="model", input_example=EXAMPLE_X)

with mlflow.start_run(run_name="CLS - DecisionTreeClassifier"):
    mlflow.log_params(common_params)
    mlflow.log_param("model", "DecisionTreeClassifier")
    mlflow.log_metrics(m_treec)
    mlflow.sklearn.log_model(tree_cls, name="model", input_example=EXAMPLE_X)

with mlflow.start_run(run_name="REG - LinearRegression"):
    mlflow.log_params(common_params)
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metrics(m_linr)
    mlflow.sklearn.log_model(linreg, name="model", input_example=EXAMPLE_X)

with mlflow.start_run(run_name="REG - DecisionTreeRegressor"):
    mlflow.log_params(common_params)
    mlflow.log_param("model", "DecisionTreeRegressor")
    mlflow.log_metrics(m_treer)
    mlflow.sklearn.log_model(tree_reg, name="model", input_example=EXAMPLE_X)

with mlflow.start_run(run_name="NN - Classifier (MLPClassifier)"):
    mlflow.log_params(
        {
            **common_params,
            "architecture": f"MLPClassifier hidden={NN_HIDDEN}, relu",
            "dropout_input": NN_DROPOUT_P,
            "optimizer": "Adam",
            "learning_rate": NN_LR,
            "max_epochs": NN_EPOCHS,
            "patience": NN_PATIENCE,
        }
    )
    mlflow.log_metrics(m_nn_cls)
    mlflow.log_artifact(plot1_path)
    mlflow.sklearn.log_model(nn_cls, name="model", input_example=X_tr_nn[:5])

with mlflow.start_run(run_name="NN - Regressor (MLPRegressor)"):
    mlflow.log_params(
        {
            **common_params,
            "architecture": f"MLPRegressor hidden={NN_HIDDEN}, relu",
            "dropout_input": NN_DROPOUT_P,
            "optimizer": "Adam",
            "learning_rate": NN_LR,
            "max_epochs": NN_EPOCHS,
            "patience": NN_PATIENCE,
        }
    )
    mlflow.log_metrics(m_nn_reg)
    mlflow.log_artifact(plot2_path)
    mlflow.sklearn.log_model(nn_reg, name="model", input_example=X_tr_nn[:5])

with mlflow.start_run(run_name="Final - Best Models Summary"):
    mlflow.log_params(
        {
            **common_params,
            "best_final_cls_model": best_final_cls_name,
            "best_final_reg_model": best_final_reg_name,
        }
    )
    mlflow.log_metrics(
        {
            "best_final_cls_val_f1": best_final_cls_metrics["val_f1"],
            "best_final_cls_test_f1": best_final_cls_metrics["test_f1"],
            "best_final_reg_val_rmse": best_final_reg_metrics["val_rmse"],
            "best_final_reg_test_rmse": best_final_reg_metrics["test_rmse"],
        }
    )
    mlflow.log_artifact(plot1_path)
    mlflow.log_artifact(plot2_path)
    mlflow.log_artifact(plot3_path)
    mlflow.log_artifact(plot4_path)
    mlflow.log_artifact(plot5_path)
    mlflow.log_artifact(table1_path)
    mlflow.log_artifact(table2_path)
    mlflow.log_artifact(eda_corr_path)

print(f"\nAll figures and tables saved in: {OUTPUT_DIR}")
print(f"MLflow experiment name: '{EXPERIMENT_NAME}'")
