"""
Feature Engineering Pipeline for Retail Demand Forecasting.
Creates:
- Strict Product x Date continuous grid (fill missing dates with 0 sales)
- Calendar features (dayofweek, month, quarter, day, is_weekend)
- Holiday and festival flags (Black Friday, Cyber Monday, Christmas rush, New Year)
- Promotional / discount indicators
- Lag sales: lag_1, lag_7, lag_14, lag_28 (preventing data leakage)
- Rolling statistics: rolling_mean_7, 14, 28; rolling_std_7, 14, 28; rolling_min_7, rolling_max_7
- Previous month demand
- Product category encoding and average unit price
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

class FeaturePipeline:
    def __init__(self, target_col: str = "Quantity"):
        self.target_col = target_col
        self.feature_columns: List[str] = []

    def build_grid(self, clean_transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates transactions by product and date, ensuring a full calendar grid."""
        df = clean_transactions_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Aggregate daily sales per product
        daily = df.groupby(["Date", "StockCode"]).agg(
            Quantity=("Quantity", "sum"),
            Revenue=("Revenue", "sum"),
            AvgUnitPrice=("UnitPrice", "mean"),
            TransactionsCount=("InvoiceNo", "nunique"),
            Category=("Category", "first"),
            Description=("Description", "first")
        ).reset_index()

        # Build Cartesian product grid of all unique dates x all unique products
        all_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
        all_skus = daily["StockCode"].unique()
        
        idx = pd.MultiIndex.from_product([all_dates, all_skus], names=["Date", "StockCode"])
        full_grid = pd.DataFrame(index=idx).reset_index()

        # Merge daily aggregated sales
        merged = pd.merge(full_grid, daily, on=["Date", "StockCode"], how="left")
        merged["Quantity"] = merged["Quantity"].fillna(0)
        merged["Revenue"] = merged["Revenue"].fillna(0)
        merged["TransactionsCount"] = merged["TransactionsCount"].fillna(0)

        # Impute static product meta (AvgUnitPrice, Category, Description)
        sku_meta = daily.groupby("StockCode").agg(
            Category=("Category", "first"),
            Description=("Description", "first"),
            GlobalAvgPrice=("AvgUnitPrice", "mean")
        ).reset_index()

        merged = pd.merge(merged.drop(columns=["Category", "Description", "AvgUnitPrice"]), sku_meta, on="StockCode", how="left")
        merged["AvgUnitPrice"] = merged["GlobalAvgPrice"]
        merged = merged.drop(columns=["GlobalAvgPrice"])

        # Sort chronologically by product
        merged = merged.sort_values(["StockCode", "Date"]).reset_index(drop=True)
        return merged

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month
        df["Quarter"] = df["Date"].dt.quarter
        df["DayOfMonth"] = df["Date"].dt.day
        df["DayOfYear"] = df["Date"].dt.dayofyear
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

        # Cyclical transformations for seasonality
        df["Sin_DayOfWeek"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
        df["Cos_DayOfWeek"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
        df["Sin_Month"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Cos_Month"] = np.cos(2 * np.pi * df["Month"] / 12)

        return df

    def add_holiday_and_promotions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flags major commercial festivals and promotional surges."""
        month = df["Month"]
        day = df["DayOfMonth"]

        # Black Friday (Late November, approx Nov 24-29)
        df["IsBlackFriday"] = ((month == 11) & (day >= 24) & (day <= 29)).astype(int)

        # Cyber Week (Early December, Dec 1-4)
        df["IsCyberWeek"] = ((month == 12) & (day >= 1) & (day <= 4)).astype(int)

        # Christmas Shopping Rush (Dec 15-23)
        df["IsChristmasRush"] = ((month == 12) & (day >= 15) & (day <= 23)).astype(int)

        # New Year Holiday (Dec 31 - Jan 2)
        df["IsNewYear"] = (((month == 12) & (day >= 31)) | ((month == 1) & (day <= 2))).astype(int)

        # Summer Sale (July 10-20)
        df["IsSummerSale"] = ((month == 7) & (day >= 10) & (day <= 20)).astype(int)

        # Composite festival/promo indicator
        df["IsPromotion"] = (
            df["IsBlackFriday"] | df["IsCyberWeek"] | df["IsChristmasRush"] | df["IsSummerSale"]
        ).astype(int)

        return df

    def add_lag_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates lag and rolling features strictly using historical demand
        (shifted by at least 1 day) to avoid future lookahead bias / data leakage.
        """
        grouped = df.groupby("StockCode")[self.target_col]

        # Lags
        for lag in [1, 7, 14, 28]:
            df[f"Lag_{lag}"] = grouped.shift(lag)

        # Rolling statistics using shift(1)
        shifted_series = grouped.shift(1)
        # Using groupby on shifted values
        for window in [7, 14, 28]:
            df[f"RollingMean_{window}"] = df.groupby("StockCode")["Quantity"].transform(
                lambda s: s.shift(1).rolling(window, min_periods=3).mean()
            )
            df[f"RollingStd_{window}"] = df.groupby("StockCode")["Quantity"].transform(
                lambda s: s.shift(1).rolling(window, min_periods=3).std()
            )

        # 7-day Min/Max
        df["RollingMin_7"] = df.groupby("StockCode")["Quantity"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=3).min()
        )
        df["RollingMax_7"] = df.groupby("StockCode")["Quantity"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=3).max()
        )

        # Previous month total demand (~30 days sum shifted)
        df["PrevMonthDemand"] = df.groupby("StockCode")["Quantity"].transform(
            lambda s: s.shift(1).rolling(30, min_periods=7).sum()
        )

        # Rolling volatility ratio (std / (mean + 1e-5))
        df["DemandVolatility_7"] = df["RollingStd_7"] / (df["RollingMean_7"] + 1e-5)

        return df

    def add_product_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encodes categories and adds price-tier features."""
        # One-hot encode category
        categories = df["Category"].unique()
        for cat in sorted(categories):
            safe_cat = "".join([c if c.isalnum() else "_" for c in cat])
            df[f"Cat_{safe_cat}"] = (df["Category"] == cat).astype(int)
        
        # Log-transformed price
        df["LogUnitPrice"] = np.log1p(df["AvgUnitPrice"])
        return df

    def transform(self, clean_transactions_df: pd.DataFrame) -> pd.DataFrame:
        print("Transforming clean transactions into daily demand feature set...")
        grid = self.build_grid(clean_transactions_df)
        grid = self.add_calendar_features(grid)
        grid = self.add_holiday_and_promotions(grid)
        grid = self.add_lag_and_rolling_features(grid)
        grid = self.add_product_embeddings(grid)

        # Drop initial burn-in period where 28-day lags are NaN
        initial_count = len(grid)
        valid_df = grid.dropna().copy().reset_index(drop=True)
        print(f"Burn-in rows dropped: {initial_count - len(valid_df)}. Valid feature rows: {len(valid_df)}")

        # Store feature columns (excluding target, meta, dates)
        exclude_cols = ["Date", "StockCode", "Description", "Category", "Quantity", "Revenue", "TransactionsCount"]
        self.feature_columns = [c for c in valid_df.columns if c not in exclude_cols]
        return valid_df

if __name__ == "__main__":
    import os
    clean_path = "/working_dir/smart-demand-forecasting/data/processed/cleaned_transactions.csv"
    processed_dir = "/working_dir/smart-demand-forecasting/data/processed"
    out_features_path = os.path.join(processed_dir, "daily_demand_features.csv")

    clean_df = pd.read_csv(clean_path)
    pipeline = FeaturePipeline()
    featured_df = pipeline.transform(clean_df)
    featured_df.to_csv(out_features_path, index=False)
    print(f"Features saved to {out_features_path}")
    print("Engineered feature columns count:", len(pipeline.feature_columns))
    print("Sample feature columns:", pipeline.feature_columns[:10])
