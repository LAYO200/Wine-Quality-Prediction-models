# shared helpers
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# data paths
DATA_DIR = PROJECT_ROOT / "data" / "wine+quality"
DATA_RED_PATH = DATA_DIR / "winequality-red.csv"
DATA_WHITE_PATH = DATA_DIR / "winequality-white.csv"
# output directory (used in train_baselines.py and train_nn.py)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
# MLflow experiment name (used in both training scripts)
EXPERIMENT_NAME = "wine_quality_experiments"  # any name you like

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
