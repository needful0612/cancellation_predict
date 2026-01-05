import os
import random
import pandas as pd
from faker import Faker
from datetime import timedelta
import string

RANDOM_SEED = 42
os.makedirs("data", exist_ok=True)
DATA_PATH = "data/orders.csv"

fake = Faker()
Faker.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def generate_reservation_no():
    chars = string.ascii_uppercase + string.digits
    suffix = ''.join(random.choices(chars, k=12))
    return f"T{suffix}"

def get_email_pool(n_target):
    """
    Generates a list of emails that follows the provided distribution.
    Total emails in your table ~41,541. 
    Total rows represented ~50,000.
    """
    dist = {
        1: 0.8484, 2: 0.1033, 3: 0.0254, 4: 0.0087,
        5: 0.0049, 6: 0.0025, 7: 0.0020, 8: 0.0010,
        9: 0.0008, 10: 0.0005, 11: 0.0007, 12: 0.0004,
        15: 0.0013, 101:0.0001
    }
    
    counts = list(dist.keys())
    weights = list(dist.values())
    
    pool = []
    while len(pool) < n_target:
        email = fake.email()
        freq = random.choices(counts, weights=weights, k=1)[0]
        pool.extend([email] * freq)
    
    random.shuffle(pool)
    return pool[:n_target]

def generate_mock_data(n=50000):
    data = []
    hotel_ids = [i for i in range(1, 121)]
    hotel_weights = [1/i for i in range(1, 121)] 
    
    email_pool = get_email_pool(n)

    for i in range(n):
        order_date = fake.date_time_between(start_date='-1y', end_date='now')
        rand_val = random.random()
        
        # Risk Logic
        if rand_val < 0.15:
            lead_time = random.randint(61, 120)
            prepaid_ratio = random.uniform(0, 0.2)
            room_qty = random.randint(5, 10)
            cancel_prob = 0.85
        elif rand_val < 0.35:
            lead_time = random.randint(31, 60)
            prepaid_ratio = random.uniform(0.3, 0.7)
            room_qty = random.randint(3, 4)
            cancel_prob = 0.30
        else:
            lead_time = random.randint(1, 30)
            prepaid_ratio = random.uniform(0.8, 1.0)
            room_qty = random.randint(1, 2)
            cancel_prob = 0.04

        price_multiplier = 1.3 if order_date.month == 12 else 1.0
        total_price = int(random.randint(2000, 45000) * price_multiplier)
        
        checkin_date = (order_date + timedelta(days=lead_time)).date()
        checkout_date = checkin_date + timedelta(days=random.randint(1, 5))
        
        is_cancelled = random.random() < cancel_prob
        cancel_date = "1900-01-01 00:00:00"
        if is_cancelled:
            cancel_date = order_date + timedelta(days=random.randint(1, lead_time))

        data.append({
            "reservation_id": i,
            "brand_id": random.randint(1, 15),
            "hotel_id": random.choices(hotel_ids, weights=hotel_weights, k=1)[0],
            "reservation_no": generate_reservation_no(),
            "room_qty": room_qty,
            "total_price": total_price,
            "prepaid": int(total_price * prepaid_ratio),
            "email": email_pool[i], # Pull from our distributed pool
            "order_date": order_date,
            "checkin_date": checkin_date,
            "checkout_date": checkout_date,
            "cancel_date": cancel_date,
            "payment_type": random.choice(['Credit Card', 'Bank Transfer', 'On-site']),
        })
        
    return pd.DataFrame(data)

df = generate_mock_data(53612)
df.to_csv(DATA_PATH, index=False)