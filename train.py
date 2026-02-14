import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sqlalchemy import create_engine
from lightgbm import LGBMRegressor

# ==========================================================
# CONFIGURATION
# ==========================================================

DB_HOST = "fintech-pg-server.postgres.database.azure.com"
DB_NAME = "disputeiqdb"
DB_USER = "pgadmin"
DB_PASS = "disputeIQ1"
DB_PORT = 5432

# Toggle local saving
SAVE_LOCALLY = True

# Local directory for model storage
LOCAL_MODEL_DIR = r"C:\Users\yourname\models"   # change this

# Versioning
USE_TIMESTAMP_VERSIONING = True

BASE_MODEL_NAME = "dispute_model"
BASE_FEATURE_NAME = "model_features"

TRANSACTION_FILE = "labeled_transactions.csv"

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
# BUILD TRAINING DATASET
# ==========================================================

def build_dataset(transaction_file):

    txns = pd.read_csv(transaction_file)

    customer_ids = txns["customer_id"].unique()
    merchant_ids = txns["merchant_id"].unique()

    cust_df = load_customer_features(customer_ids)
    merch_df = load_merchant_features(merchant_ids)
    rep_df = load_merchant_reputation(merchant_ids)

    df = txns.merge(cust_df, on="customer_id", how="left")
    df = df.merge(merch_df, on="merchant_id", how="left")
    df = df.merge(rep_df, on="merchant_id", how="left")

    # Transform amount
    df["amount_major"] = df["amount"] / 100
    df["log_amount"] = np.log1p(df["amount_major"])

    # Target
    y = df["dispute_coefficient"]

    # Drop non-feature columns
    drop_cols = [
        "dispute_coefficient",
        "customer_id",
        "merchant_id",
        "last_updated_at_x",
        "last_updated_at_y"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Convert categoricals for LightGBM
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")

    return X, y

# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(X, y):

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X, y)

    return model

# ==========================================================
# SAVE MODEL + FEATURES
# ==========================================================

def save_artifacts(model, feature_columns):

    if USE_TIMESTAMP_VERSIONING:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{BASE_MODEL_NAME}_{timestamp}.pkl"
        feature_filename = f"{BASE_FEATURE_NAME}_{timestamp}.pkl"
    else:
        model_filename = f"{BASE_MODEL_NAME}.pkl"
        feature_filename = f"{BASE_FEATURE_NAME}.pkl"

    if SAVE_LOCALLY:
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        model_path = os.path.join(LOCAL_MODEL_DIR, model_filename)
        feature_path = os.path.join(LOCAL_MODEL_DIR, feature_filename)
    else:
        model_path = model_filename
        feature_path = feature_filename

    joblib.dump(model, model_path)
    joblib.dump(feature_columns, feature_path)

    print(f"Model saved at: {model_path}")
    print(f"Feature list saved at: {feature_path}")

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("Building dataset...")
    X, y = build_dataset(TRANSACTION_FILE)

    print("Training model...")
    model = train_model(X, y)

    print("Saving artifacts...")
    save_artifacts(model, list(X.columns))

    print("Training complete.")
