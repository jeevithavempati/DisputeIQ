import os
import csv
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sqlalchemy import create_engine
from lightgbm import LGBMClassifier

# ==========================================================
# CONFIGURATION
# ==========================================================

DB_HOST = "fintech-pg-server.postgres.database.azure.com"
DB_NAME = "disputeiqdb"
DB_USER = "pgadmin"
DB_PASS = "disputeIQ1"
DB_PORT = 5432

# Training input file (your CSV)
TRANSACTION_FILE = r"C:\Users\Rohan.Sangodkar\Downloads\training_clean.csv"

# Save artifacts locally
SAVE_LOCALLY = True
LOCAL_MODEL_DIR = r"C:\Users\Rohan.Sangodkar\Desktop\dispute\models"

# Versioning
USE_TIMESTAMP_VERSIONING = True
BASE_MODEL_NAME = "dispute_model"
BASE_FEATURE_NAME = "model_features"

# ==========================================================
# DATABASE CONNECTION
# ==========================================================

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ==========================================================
# LOAD FEATURE TABLES
# ==========================================================

def load_customer_features(customer_ids):
    query = """
        SELECT *
        FROM dispute_poc.customer_behavior_current
        WHERE customer_id = ANY(%(ids)s)
    """
    return pd.read_sql(query, engine, params={"ids": list(customer_ids)})


def load_merchant_features(merchant_ids):
    query = """
        SELECT *
        FROM dispute_poc.merchant_behavior_current
        WHERE merchant_id = ANY(%(ids)s)
    """
    return pd.read_sql(query, engine, params={"ids": list(merchant_ids)})


def load_merchant_reputation(merchant_ids):
    query = """
        SELECT *
        FROM dispute_poc.merchant_reputation
        WHERE merchant_id = ANY(%(ids)s)
    """
    return pd.read_sql(query, engine, params={"ids": list(merchant_ids)})

# ==========================================================
# ROBUST CSV READER (NO pandas sep=None)
# ==========================================================

def read_training_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

# ==========================================================
# BUILD TRAINING DATASET
# ==========================================================

def build_dataset(transaction_file: str):
    txns = read_training_csv(transaction_file)

    required_cols = {
        "customer_id",
        "merchant_id",
        "amount",
        "ecommerce_flag",
        "cross_border_flag",
        "dispute_flag"
    }

    missing = required_cols - set(txns.columns)
    if missing:
        raise ValueError(
            f"Training file missing columns: {missing}\n"
            f"Found columns: {list(txns.columns)}"
        )

    # Normalize dtypes
    txns["amount"] = pd.to_numeric(txns["amount"], errors="coerce")
    txns["ecommerce_flag"] = pd.to_numeric(txns["ecommerce_flag"], errors="coerce").fillna(0).astype(int)
    txns["cross_border_flag"] = pd.to_numeric(txns["cross_border_flag"], errors="coerce").fillna(0).astype(int)
    txns["dispute_flag"] = pd.to_numeric(txns["dispute_flag"], errors="coerce").fillna(0).astype(int)

    # Basic transformations
    txns["amount_major"] = txns["amount"] / 100.0
    txns["log_amount"] = np.log1p(txns["amount_major"])

    customer_ids = txns["customer_id"].unique()
    merchant_ids = txns["merchant_id"].unique()

    # Fetch current features
    cust_df = load_customer_features(customer_ids)
    merch_df = load_merchant_features(merchant_ids)
    rep_df = load_merchant_reputation(merchant_ids)

    # Merge
    df = txns.merge(cust_df, on="customer_id", how="left")
    df = df.merge(merch_df, on="merchant_id", how="left")
    df = df.merge(rep_df, on="merchant_id", how="left")

    # Target
    y = df["dispute_flag"]

    # Drop non-feature columns
    drop_cols = {"dispute_flag", "customer_id", "merchant_id"}

    # Drop any last_updated_at columns (could be last_updated_at, last_updated_at_x, last_updated_at_y)
    for c in list(df.columns):
        if c.startswith("last_updated_at"):
            drop_cols.add(c)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # LightGBM can use pandas 'category' dtype
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")

    # Leave NaNs as-is (LightGBM handles missing values)
    return X, y

# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(X: pd.DataFrame, y: pd.Series):
    model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X, y)
    return model

# ==========================================================
# SAVE MODEL + FEATURES + POINTER FILE
# ==========================================================

def save_artifacts(model, feature_columns):
    if USE_TIMESTAMP_VERSIONING:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{BASE_MODEL_NAME}_{ts}.pkl"
        feature_filename = f"{BASE_FEATURE_NAME}_{ts}.pkl"
    else:
        model_filename = f"{BASE_MODEL_NAME}.pkl"
        feature_filename = f"{BASE_FEATURE_NAME}.pkl"

    if SAVE_LOCALLY:
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        model_path = os.path.join(LOCAL_MODEL_DIR, model_filename)
        feature_path = os.path.join(LOCAL_MODEL_DIR, feature_filename)
        pointer_path = os.path.join(LOCAL_MODEL_DIR, "active_model.txt")
    else:
        model_path = model_filename
        feature_path = feature_filename
        pointer_path = "active_model.txt"

    joblib.dump(model, model_path)
    joblib.dump(feature_columns, feature_path)

    # pointer file for prediction script
    with open(pointer_path, "w", encoding="utf-8") as f:
        f.write(model_filename + "\n")
        f.write(feature_filename + "\n")

    print(f"Model saved:    {model_path}")
    print(f"Features saved: {feature_path}")
    print(f"Pointer saved:  {pointer_path}")

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    print("Building dataset...")
    X, y = build_dataset(TRANSACTION_FILE)

    print(f"Dataset ready. X={X.shape}, y={y.shape}, dispute_rate={y.mean():.4f}")

    print("Training model...")
    model = train_model(X, y)

    print("Saving artifacts...")
    save_artifacts(model, list(X.columns))

    print("Training complete.")
