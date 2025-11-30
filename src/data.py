# loading, cleaning, splitting
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import SEED, DATA_RED_PATH, DATA_WHITE_PATH


def load_wine_data():
    #Load red + white wine CSVs, clean, derive target_cls, return df and feature list.
    red_df = pd.read_csv(DATA_RED_PATH, sep=";")
    white_df = pd.read_csv(DATA_WHITE_PATH, sep=";")

    df = pd.concat(
        [red_df.assign(type="red"), white_df.assign(type="white")],
        ignore_index=True,
    ).drop_duplicates()

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

    return df, features


def make_splits(df, features, seed=SEED):
    X = df[features].astype("float64")
    y_cls = df["target_cls"].values
    y_reg = df["quality"].values

    X_tr, X_te, y_cls_tr, y_cls_te, y_reg_tr, y_reg_te = train_test_split(
        X, y_cls, y_reg, test_size=0.20, random_state=seed, stratify=y_cls
    )

    X_tr, X_va, y_cls_tr, y_cls_va, y_reg_tr, y_reg_va = train_test_split(
        X_tr,
        y_cls_tr,
        y_reg_tr,
        test_size=0.25,
        random_state=seed,
        stratify=y_cls_tr,
    )

    return (
        X_tr,
        X_va,
        X_te,
        y_cls_tr,
        y_cls_va,
        y_cls_te,
        y_reg_tr,
        y_reg_va,
        y_reg_te,
    )
