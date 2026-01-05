import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_OUTPUT_PATH = "models/final_artifact.joblib"

artifact = joblib.load(MODEL_OUTPUT_PATH)
pipeline = artifact["pipeline"]
threshold = artifact["threshold"]

app = FastAPI()

TRAINING_COLUMNS = [
    'brand_id', 'hotel_id', 'total_price', 'prepaid', 
    'payment_type', 'order_month', 'order_day_of_week', 
    'is_weekend', 'lead_time', 'stay_duration', 
    'price_per_room', 'deposit_ratio', 'group_size', 
    'customer_order_count'
]

class BookingRequest(BaseModel):
    brand_id: int
    hotel_id: int
    room_qty: int = Field(..., gt=0) # Must be greater than 0
    total_price: float = Field(..., ge=0) # Cannot be negative
    prepaid: float = Field(..., ge=0)
    payment_type: str
    order_date: str         # "2024-01-01"
    checkin_date: str       # "2024-01-10"
    checkout_date: str      # "2024-01-12"
    customer_order_count: int 

@app.post("/predict")
def predict(request: BookingRequest):
    raw_data = request.model_dump()
    df = pd.DataFrame([raw_data])

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    df["checkout_date"] = pd.to_datetime(df["checkout_date"])

    df['order_month'] = df['order_date'].dt.month
    df['order_day_of_week'] = df['order_date'].dt.dayofweek
    df['is_weekend'] = (df['order_date'].dt.dayofweek >= 5).astype(int)
    
    df["lead_time"] = (df["checkin_date"] - df["order_date"]).dt.days
    df["stay_duration"] = (df["checkout_date"] - df["checkin_date"]).dt.days
    
    safe_stay = df["stay_duration"].replace(0, 1)
    df["price_per_room"] = df["total_price"] / (df["room_qty"] * safe_stay)
    df["deposit_ratio"] = (df["prepaid"] / df["total_price"]).clip(0, 1)
    df["group_size"] = df["room_qty"]

    cols_to_drop = ['order_date', 'checkin_date', 'checkout_date', 'room_qty']
    df = df.drop(columns=cols_to_drop)
    df_processed = df[TRAINING_COLUMNS]

    y_prob = pipeline.predict_proba(df_processed)[0, 1]
    
    is_cancelled = bool(y_prob >= threshold)

    return {
        "cancel prob": round(float(y_prob), 4),
        "predict": "Cancelled" if is_cancelled else "Not Cancelled",
        "is_cancelled": is_cancelled
    }