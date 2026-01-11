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

from src.utils import (
    process_booking_data, 
    TRAINING_COLUMNS,
    RANDOM_SEED,
    MODEL_PATH,
    DATA_PATH
)

def prepare_features(df_raw):
    df_raw = pd.read_csv(DATA_PATH)
    
    df_raw['is_cancelled'] = (df_raw['cancel_date'] != "1900-01-01 00:00:00").astype(int)
    df_raw['customer_order_count'] = df_raw.groupby('email')['email'].transform('count')
    
    X_processed = process_booking_data(df_raw)
    
    X_processed['is_cancelled'] = df_raw['is_cancelled'].values
    return X_processed

def build_preprocessor():
    log_transformer = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    
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
    
    rf_model = pipeline.named_steps['classifier']
    try:
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    except Exception as e:
        print(f"Warning: Could not extract feature names automatically: {e}")
        feature_names = TRAINING_COLUMNS
    
    importances = rf_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    print("\n--- Global Feature Weights (Top 10) ---")
    print(feature_importance_df.head(10))

    artifact = {
        "pipeline": pipeline,
        "threshold": best_threshold,
        "metadata": {
            "test_auc": test_auc,
            "feature_names": X.columns.tolist(),
            "global_importances": feature_importance_df.to_dict(orient="records")
        }
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Artifact saved to {MODEL_PATH}")

train()