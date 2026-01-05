import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    RobustScaler, 
    OneHotEncoder, 
    FunctionTransformer, 
    TargetEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
DATA_PATH = "data/orders.csv"
MODEL_OUTPUT_PATH = "models/final_artifact.joblib"
RANDOM_SEED = 42

def prepare_features(df):
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    df["checkout_date"] = pd.to_datetime(df["checkout_date"])
    
    df['order_month'] = df['order_date'].dt.month
    df['order_day_of_week'] = df['order_date'].dt.dayofweek
    df['is_weekend'] = (df['order_date'].dt.dayofweek >= 5).astype(int)
    df['customer_order_count'] = df.groupby('email')['email'].transform('count')
    df["lead_time"] = (df["checkin_date"] - df["order_date"]).dt.days
    df["stay_duration"] = (df["checkout_date"] - df["checkin_date"]).dt.days
    
    safe_stay = df["stay_duration"].replace(0, 1)
    df["price_per_room"] = df["total_price"] / (df["room_qty"] * safe_stay)
    
    df["deposit_ratio"] = (df["prepaid"] / df["total_price"]).clip(0, 1)
    df["group_size"] = df["room_qty"]
    
    df['is_cancelled'] = (df['cancel_date'] != "1900-01-01 00:00:00").astype(int)
    
    drop_cols = [
        'reservation_id', 'reservation_no', 'email', 'order_date', 
        'checkin_date', 'checkout_date', 'cancel_date', 'room_qty'
    ]
    df.drop(columns=drop_cols, inplace=True)
    return df

def build_preprocessor():
    log_transformer = FunctionTransformer(np.log1p)
    
    return ColumnTransformer(
        transformers=[
            ('num_log', log_transformer, ['price_per_room', 'customer_order_count']),
            ('num_scale', RobustScaler(), ['lead_time', 'deposit_ratio', 'stay_duration']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['payment_type', 'brand_id']),
            ('hotel_target', TargetEncoder(target_type='binary'), ['hotel_id'])
        ])

def train():
    df_raw = pd.read_csv(DATA_PATH)
    df = prepare_features(df_raw)
    
    X = df.drop('is_cancelled', axis=1)
    y = df['is_cancelled']

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train_full
    )

    preprocessor = build_preprocessor()
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_val_probs = pipeline.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0, 1, 101)
    f1_scores = [f1_score(y_val, (y_val_probs >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    y_test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_probs)
    
    print(f"--- Results ---")
    print(f"Best Val Threshold: {best_threshold:.2f}")
    print(f"Final Test AUC:      {test_auc:.4f}")

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "threshold": best_threshold,
        "metadata": {
            "test_auc": test_auc,
            "feature_names": X.columns.tolist()
        }
    }
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print(f"Artifact saved to {MODEL_OUTPUT_PATH}")

train()