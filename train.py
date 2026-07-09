"""
train.py — Train an XGBoost model on German ImmobilienScout24 data
           (Rhineland-Palatinate focus).

Target  : obj_purchasePrice (EUR)
Features: see feature groups below
Tracking: MLflow  →  mlflow ui

Usage:
    python train.py
"""

import json
import sqlite3
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DB_PATH   = Path("data/housing.db")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "obj_purchasePrice"

# ── Feature groups ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "obj_livingSpace",      # m²
    "obj_noRooms",          # number of rooms
    "obj_lotArea",          # lot/plot area m²
    "obj_yearConstructed",  # year built
    "obj_noParkSpaces",     # parking spaces
    "obj_numberOfFloors",   # floors in building
    "obj_thermalChar",      # energy consumption kWh/m²a
    "obj_pricetrendbuy",    # local buy price trend (%)
    "geo_lat",              # latitude  (from zip lookup)
    "geo_lng",              # longitude (from zip lookup)
]

BINARY_FEATURES = [
    "obj_newlyConst",    # newly constructed
    "obj_cellar",        # has cellar
    "obj_barrierFree",   # barrier-free access
    "obj_rented",        # currently rented out
]

ORDERED_CAT_FEATURES = ["obj_condition", "obj_interiorQual"]
ORDERED_CAT_CATEGORIES = [
    # condition: worst → best
    ["need_of_renovation", "negotiable", "refurbished",
     "modernized", "well_kept", "mint_condition",
     "first_time_use_after_refurbishment", "first_time_use", "no_information"],
    # interiorQual: worst → best
    ["simple", "normal", "sophisticated", "luxury", "no_information"],
]

ONEHOT_FEATURES = [
    "obj_heatingType",
    "obj_buildingType",
    "obj_firingTypes",
    "obj_constructionPhase",
]

ALL_FEATURES = (NUMERIC_FEATURES + BINARY_FEATURES +
                ORDERED_CAT_FEATURES + ONEHOT_FEATURES)


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run `python setup_db.py` first."
        )
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM housing", conn)
    conn.close()
    print(f"Loaded {len(df):,} rows | target median: €{df[TARGET].median():,.0f}")
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numeric: fill nulls with median
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    # Binary: fill nulls with 0
    for col in BINARY_FEATURES:
        df[col] = df.get(col, pd.Series(0, index=df.index)).fillna(0).astype(int)

    # Categorical: fill nulls with "no_information"
    for col in ORDERED_CAT_FEATURES + ONEHOT_FEATURES:
        df[col] = df.get(col, pd.Series("no_information", index=df.index)) \
                    .fillna("no_information").astype(str)

    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough",   BINARY_FEATURES),
            ("ord", OrdinalEncoder(
                        categories=ORDERED_CAT_CATEGORIES,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),               ORDERED_CAT_FEATURES),
            ("ohe", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),               ONEHOT_FEATURES),
        ],
        remainder="drop",
    )


# ── Training ───────────────────────────────────────────────────────────────────
def train() -> None:
    df = load_data()
    df = prepare(df)

    X = df[ALL_FEATURES]
    y = np.log1p(df[TARGET])   # log transform: reduces price skew

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    mlflow.set_experiment("immo-preisrechner-rlp")

    xgb_params = {
        "n_estimators":          600,
        "max_depth":             6,
        "learning_rate":         0.04,
        "subsample":             0.8,
        "colsample_bytree":      0.8,
        "min_child_weight":      5,
        "reg_alpha":             0.1,
        "reg_lambda":            1.0,
        "random_state":          42,
        "n_jobs":                -1,
        "eval_metric":           "rmse",
        "early_stopping_rounds": 40,
    }

    with mlflow.start_run():
        model = XGBRegressor(**{k: v for k, v in xgb_params.items()
                                if k != "early_stopping_rounds"})
        model.set_params(early_stopping_rounds=xgb_params["early_stopping_rounds"])
        model.fit(
            X_train_t, y_train,
            eval_set=[(X_test_t, y_test)],
            verbose=50,
        )

        y_pred = np.expm1(model.predict(X_test_t))
        y_true = np.expm1(y_test)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)

        print("\n" + "=" * 44)
        print("  RESULTS — Rheinland-Pfalz")
        print("=" * 44)
        print(f"  R²   : {r2:.4f}")
        print(f"  RMSE : €{rmse:,.0f}")
        print(f"  MAE  : €{mae:,.0f}")
        print(f"  MAPE : {mape:.2f}%")
        print("=" * 44)

        mlflow.log_params(xgb_params)
        mlflow.log_metrics({"r2": r2, "rmse": rmse, "mae": mae, "mape": mape})
        mlflow.xgboost.log_model(model, "model")

        joblib.dump(model,        MODEL_DIR / "model.joblib")
        joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

        (MODEL_DIR / "metrics.json").write_text(
            json.dumps({"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}, indent=2)
        )
        (MODEL_DIR / "features.json").write_text(
            json.dumps({
                "numeric":     NUMERIC_FEATURES,
                "binary":      BINARY_FEATURES,
                "ordered_cat": ORDERED_CAT_FEATURES,
                "onehot_cat":  ONEHOT_FEATURES,
                "all":         ALL_FEATURES,
            }, indent=2)
        )

        print(f"\n💾 Saved to {MODEL_DIR}/")
        print("🎉 Done!  Run: streamlit run app.py")


if __name__ == "__main__":
    train()
