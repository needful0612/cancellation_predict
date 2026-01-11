import os
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List

TRAINING_COLUMNS = [
    'brand_id', 'hotel_id', 'total_price', 'prepaid', 
    'payment_type', 'order_month', 'order_day_of_week', 
    'is_weekend', 'lead_time', 'stay_duration', 
    'price_per_room', 'deposit_ratio', 'group_size', 
    'customer_order_count'
]
MODEL_PATH = "models/final_artifact.joblib"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
RANDOM_SEED = 42
DATA_PATH = "data/orders.csv"

class BookingRequest(BaseModel):
    reservation_id: int
    brand_id: int
    hotel_id: int
    room_qty: int = Field(..., gt=0)
    total_price: float = Field(..., ge=0)
    prepaid: float = Field(..., ge=0)
    payment_type: str
    order_date: str         # Expected: "YYYY-MM-DD"
    checkin_date: str
    checkout_date: str
    customer_order_count: int = Field(default=1, ge=0)

    @field_validator('order_date', 'checkin_date', 'checkout_date')
    @classmethod
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

class BatchBookingRequest(BaseModel):
    bookings: List[BookingRequest]

def process_booking_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes input data to match the training feature set and column order.
    """
    df = df.copy()

    date_cols = ["order_date", "checkin_date", "checkout_date"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    df['order_month'] = df['order_date'].dt.month
    df['order_day_of_week'] = df['order_date'].dt.dayofweek
    df['is_weekend'] = (df['order_day_of_week'] >= 5).astype(int)
    
    df["lead_time"] = (df["checkin_date"] - df["order_date"]).dt.days
    df["stay_duration"] = (df["checkout_date"] - df["checkin_date"]).dt.days
    
    safe_stay = df["stay_duration"].replace(0, 1)
    df["group_size"] = df["room_qty"]
    df["price_per_room"] = df["total_price"] / (df["room_qty"] * safe_stay)
    
    df["deposit_ratio"] = (df["prepaid"] / df["total_price"]).fillna(0).clip(0, 1)

    for col in TRAINING_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[TRAINING_COLUMNS]
            
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    return df