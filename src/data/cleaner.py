"""
Data Cleaning Module.
Implements retail transactional data cleansing rules:
- Filters out cancellations (InvoiceNo starting with 'C' or Quantity <= 0)
- Cleans missing customer IDs and drops invalid product codes/prices
- Parses datetime fields
- Removes duplicate rows
- Calculates transaction revenue: revenue = Quantity * UnitPrice
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

class DataCleaner:
    def __init__(self, raw_filepath: str):
        self.raw_filepath = raw_filepath
        self.audit_log: Dict[str, Any] = {}

    def clean(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = pd.read_csv(self.raw_filepath)
        initial_rows = len(df)
        self.audit_log['initial_rows'] = initial_rows

        # 1. Deduplication
        duplicates_count = df.duplicated().sum()
        df = df.drop_duplicates()
        self.audit_log['duplicates_removed'] = int(duplicates_count)

        # 2. Datetime parsing
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        # 3. Filter cancellations and negative/zero quantities
        is_cancellation = df['InvoiceNo'].astype(str).str.upper().str.startswith('C')
        is_invalid_qty = df['Quantity'] <= 0
        is_invalid_price = df['UnitPrice'] <= 0
        
        invalid_mask = is_cancellation | is_invalid_qty | is_invalid_price
        self.audit_log['cancellations_and_invalid_records'] = int(invalid_mask.sum())
        df = df[~invalid_mask].copy()

        # 4. Handle Missing Values
        missing_customers = df['CustomerID'].isna().sum()
        self.audit_log['missing_customer_ids_imputed'] = int(missing_customers)
        df['CustomerID'] = df['CustomerID'].fillna('GUEST_CHECKOUT')
        
        # Drop rows missing essential product identifiers
        df = df.dropna(subset=['StockCode', 'Description'])

        # 5. Financial metrics
        df['Revenue'] = (df['Quantity'] * df['UnitPrice']).round(2)
        
        # Extract calendar date
        df['Date'] = pd.to_datetime(df['InvoiceDate'].dt.date)

        self.audit_log['final_clean_rows'] = len(df)
        self.audit_log['total_revenue'] = float(df['Revenue'].sum())
        self.audit_log['unique_skus'] = int(df['StockCode'].nunique())
        self.audit_log['date_range'] = (str(df['Date'].min().date()), str(df['Date'].max().date()))

        return df, self.audit_log
