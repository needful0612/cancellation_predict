import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from src.utils import (
    process_booking_data, 
    BatchBookingRequest,
    BookingRequest, 
    MODEL_PATH
)

app = FastAPI()

pipeline = None
threshold = 0.5
metadata = {}

try:
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    metadata = artifact.get("metadata", {})
except FileNotFoundError:
    print(f"Warning: Model file not found at {MODEL_PATH}. Please run training first.")

@app.post("/predict")
def predict(request: BookingRequest):
    """
    1. Validation: Handled by BookingRequest (Pydantic)
    2. Processing: Handled by shared process_booking_data
    3. Prediction: Handled by the loaded pipeline
    """
    df_raw = pd.DataFrame([request.model_dump()])
    
    df_processed = process_booking_data(df_raw)
    
    y_prob = pipeline.predict_proba(df_processed)[0, 1]
    
    is_cancelled = bool(y_prob >= threshold)

    return {
        "cancel_prob": round(float(y_prob), 4),
        "prediction": "Cancelled" if is_cancelled else "Not Cancelled",
        "is_cancelled": is_cancelled,
        "threshold_used": threshold
    }

# VERY IMPORTANT: THE ORDER HERE MATTERS.
@app.post("/predict_batch")
def predict_batch(request: BatchBookingRequest):
    input_data = [b.model_dump() for b in request.bookings]
    
    df_raw = pd.DataFrame(input_data)
    
    df_for_model = df_raw.copy()
    df_processed = process_booking_data(df_for_model)
    
    y_probs = pipeline.predict_proba(df_processed)[:, 1]
    
    df_raw["cancel_prob"] = np.round(y_probs.astype(float), 4)
    df_raw["is_cancelled"] = (df_raw["cancel_prob"] >= threshold).astype(bool)
    df_raw["prediction"] = df_raw["is_cancelled"].map({True: "Cancelled", False: "Not Cancelled"})
    
    return {"results": df_raw.to_dict(orient="records")}

@app.get("/model_info")
def get_model_info():
    """
    Returns global feature importance and model performance metrics.
    """
    return {
        "threshold": threshold,
        "test_auc": metadata.get("test_auc"),
        "global_importances": metadata.get("global_importances", []),
        "features_used": metadata.get("feature_names", [])
    }