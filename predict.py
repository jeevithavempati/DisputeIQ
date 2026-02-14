import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine

# ==========================================================
# CONFIGURATION
# ==========================================================

DB_HOST = "your-server.postgres.database.azure.com"
DB_NAME = "your_db"
DB_USER = "your_user"
DB_PASS = "your_password"
DB_PORT = 5432

# Local model directory (same as training script)
LOCAL_MODEL_DIR = r"C:\Users\yourname\models"

MODEL_FILENAME = "dispute_model_20260212_101530.pkl"      # <-- use your actual saved filename
FEATURE_FILENAME = "model_features_20260212_101530.pkl"   # <-- matching feature file

INPUT_FILE = "incoming_transactions.csv"
OUTPUT_FILE = "scored_transactions.csv"

# ==========================================================
# LOAD MODEL + FEATURES
# ==========================================================

model_path = os.path.join(LOCAL_MODEL_DIR, MODEL_FILENAME)
feature_path = os.path.join(LOCAL_MODEL_DIR, FEATURE_FILENAME)

model = joblib.load(model_path)
feature_columns = joblib.load(feature_path)

print("Model and feature schema loaded.")

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
# BUILD FEATURE MATRIX
# ==========================================================

def build_features(transaction_file):

    txns = pd.read_csv(transaction_file)

    customer_ids = txns["customer_id"].unique()
    merchant_ids = txns["merchant_id"].unique()

    cust_df = load_customer_features(customer_ids)
    merch_df = load_merchant_features(merchant_ids)
    rep_df = load_merchant_reputation(merchant_ids)

    df = txns.merge(cust_df, on="customer_id", how="left")
    df = df.merge(merch_df, on="merchant_id", how="left")
    df = df.merge(rep_df, on="merchant_id", how="left")

    # Same transformation as training
    df["amount_major"] = df["amount"] / 100
    df["log_amount"] = np.log1p(df["amount_major"])

    # Drop identifiers
    drop_cols = [
        "customer_id",
        "merchant_id",
        "last_updated_at_x",
        "last_updated_at_y"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Convert categoricals exactly like training
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")

    return X, df

# ==========================================================
# PREDICTION
# ==========================================================

def predict():

    print("Building feature matrix...")
    X, full_df = build_features(INPUT_FILE)

    # Align columns exactly to training order
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_columns]

    print("Scoring transactions...")
    preds = model.predict(X)

    # Clip to valid probability range
    preds = np.clip(preds, 0, 1)

    full_df["dispute_coefficient"] = preds

    full_df.to_csv(OUTPUT_FILE, index=False)

    print("Scoring complete.")
    print(f"Output saved to {OUTPUT_FILE}")

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    predict()
