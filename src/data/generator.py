"""
Data generation module for Smart Demand Forecasting Platform.
Generates 2 full years (2023-01-01 to 2024-12-30) of realistic retail e-commerce transactions.
Allows models to learn multi-year annual seasonality, holiday lifts (Black Friday, Cyber Week, Christmas),
and promotional dynamics.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_ecommerce_data(seed: int = 42, num_days: int = 730, base_date: str = "2023-01-01") -> pd.DataFrame:
    np.random.seed(seed)
    start_date = datetime.strptime(base_date, "%Y-%m-%d")
    
    products = [
        {"code": "SKU_1001", "desc": "Vintage Ceramic Coffee Mug", "price": 12.50, "base_qty": 35, "category": "Home & Kitchen"},
        {"code": "SKU_1002", "desc": "Wireless Bluetooth Earbuds", "price": 49.99, "base_qty": 25, "category": "Electronics"},
        {"code": "SKU_1003", "desc": "Ergonomic Memory Foam Chair Cushion", "price": 29.95, "base_qty": 18, "category": "Office"},
        {"code": "SKU_1004", "desc": "Organic Bamboo Bath Towel Set", "price": 34.00, "base_qty": 20, "category": "Home & Kitchen"},
        {"code": "SKU_1005", "desc": "Insulated Stainless Water Bottle 32oz", "price": 22.00, "base_qty": 40, "category": "Sports & Outdoors"},
        {"code": "SKU_1006", "desc": "Minimalist Smart LED Desk Lamp", "price": 39.50, "base_qty": 15, "category": "Lighting"},
        {"code": "SKU_1007", "desc": "Mechanical Gaming Keyboard RGB", "price": 79.99, "base_qty": 12, "category": "Electronics"},
        {"code": "SKU_1008", "desc": "Aromatherapy Essential Oil Diffuser", "price": 24.99, "base_qty": 28, "category": "Wellness"},
        {"code": "SKU_1009", "desc": "Heavy-Duty Canvas Travel Backpack", "price": 55.00, "base_qty": 14, "category": "Luggage & Travel"},
        {"code": "SKU_1010", "desc": "Non-Stick Cast Iron Skillet 12-inch", "price": 42.50, "base_qty": 22, "category": "Cookware"}
    ]
    
    countries = ["United Kingdom", "Germany", "France", "United States", "Australia"]
    country_weights = [0.65, 0.12, 0.10, 0.08, 0.05]
    regions = ["Region-North", "Region-South", "Region-East", "Region-West"]
    
    records = []
    invoice_seq = 500000
    customer_ids = [f"CUST_{i:05d}" for i in range(1001, 1800)]
    
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        day_of_week = current_date.weekday()
        month = current_date.month
        day_of_month = current_date.day
        
        # Day of week seasonality (higher on weekends/Fridays)
        dow_mult = 1.0 + 0.25 * (day_of_week in [4, 5, 6]) - 0.1 * (day_of_week == 0)
        
        # Monthly seasonality (peak in Nov/Dec for holiday season)
        season_mult = 1.0
        if month in [11, 12]:
            season_mult = 1.45
        elif month in [7, 8]:
            season_mult = 1.15
        elif month in [1, 2]:
            season_mult = 0.85
            
        # Specific festival/event surges
        is_black_friday = (month == 11 and 24 <= day_of_month <= 29)
        is_cyber_week = (month == 12 and 1 <= day_of_month <= 4)
        is_xmas_rush = (month == 12 and 15 <= day_of_month <= 23)
        is_summer_sale = (month == 7 and 10 <= day_of_month <= 20)
        
        event_mult = 1.0
        if is_black_friday:
            event_mult = 2.2
        elif is_cyber_week:
            event_mult = 1.9
        elif is_xmas_rush:
            event_mult = 2.1
        elif is_summer_sale:
            event_mult = 1.4
            
        # Daily orders per product
        for p in products:
            expected_demand = p["base_qty"] * dow_mult * season_mult * event_mult
            daily_orders_count = max(2, int(np.random.normal(loc=expected_demand / 3.0, scale=max(1.0, expected_demand * 0.12))))
            
            for _ in range(daily_orders_count):
                invoice_seq += 1
                inv_no = str(invoice_seq)
                cust_id = np.random.choice(customer_ids) if np.random.rand() > 0.08 else None  # missing IDs
                country = np.random.choice(countries, p=country_weights)
                region = np.random.choice(regions)
                
                qty = int(np.random.choice([1, 2, 3, 4, 5, 8, 10, 12], p=[0.45, 0.25, 0.12, 0.08, 0.04, 0.03, 0.02, 0.01]))
                
                price = p["price"]
                if is_black_friday or is_cyber_week:
                    price = round(price * 0.80, 2)
                elif is_xmas_rush:
                    price = round(price * 0.90, 2)
                elif is_summer_sale:
                    price = round(price * 0.85, 2)
                    
                hour = np.random.randint(8, 22)
                minute = np.random.randint(0, 60)
                second = np.random.randint(0, 60)
                inv_date = current_date.replace(hour=hour, minute=minute, second=second)
                
                records.append({
                    "InvoiceNo": inv_no,
                    "StockCode": p["code"],
                    "Description": p["desc"],
                    "Quantity": qty,
                    "InvoiceDate": inv_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "UnitPrice": price,
                    "CustomerID": cust_id,
                    "Country": country,
                    "StoreRegion": region,
                    "Category": p["category"]
                })
                
                # 3% chance of cancellation order
                if np.random.rand() < 0.03:
                    cancel_inv_no = f"C{invoice_seq}"
                    records.append({
                        "InvoiceNo": cancel_inv_no,
                        "StockCode": p["code"],
                        "Description": p["desc"],
                        "Quantity": -qty,
                        "InvoiceDate": (inv_date + timedelta(hours=np.random.randint(1, 24))).strftime("%Y-%m-%d %H:%M:%S"),
                        "UnitPrice": price,
                        "CustomerID": cust_id,
                        "Country": country,
                        "StoreRegion": region,
                        "Category": p["category"]
                    })
                    
    df = pd.DataFrame(records)
    
    # Add duplicates
    duplicates = df.sample(frac=0.005, random_state=seed)
    df = pd.concat([df, duplicates], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    out_dir = "/working_dir/smart-demand-forecasting/data/raw"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ecommerce_transactions.csv")
    print("Generating 2-year e-commerce raw transactions dataset...")
    df = generate_ecommerce_data()
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} records saved to {out_path}")
