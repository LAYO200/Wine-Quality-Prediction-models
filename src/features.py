# preprocessing and engineered features

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from utils import OUTPUT_DIR


def make_classical_preprocessor(features):
    scaler_ct = ColumnTransformer(
        [("scale", StandardScaler(), features)],
        remainder="drop",
    )
    return scaler_ct


def make_nn_scaled(X_tr, X_va, X_te):
    scaler_nn = StandardScaler()
    X_tr_nn = scaler_nn.fit_transform(X_tr)
    X_va_nn = scaler_nn.transform(X_va)
    X_te_nn = scaler_nn.transform(X_te)
    return scaler_nn, X_tr_nn, X_va_nn, X_te_nn


def plot_correlation_heatmap(df, features, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    corr_df = df[features + ["quality"]].astype(float)
    corr = corr_df.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    cax = plt.matshow(corr, fignum=False)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(cax)
    plt.title("EDA — Correlation Matrix with Numeric Values")

    # overlay numeric values on each cell
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
            )

    plt.tight_layout()
    eda_corr_path = os.path.join(
        output_dir, "eda_correlation_matrix_with_values.png"
    )
    plt.savefig(eda_corr_path, dpi=200, bbox_inches="tight")
    plt.close()

    return eda_corr_path
