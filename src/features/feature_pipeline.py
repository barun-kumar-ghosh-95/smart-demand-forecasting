"""
Feature engineering for the demand forecasting pipeline.

Builds a Product × Date grid, then layers on calendar features,
holiday/promo flags, lag demand, rolling stats, and category encodings.

One important design decision: all lag and rolling features use shift(1)
or higher, so there's zero lookahead into the target day's actual sales.
This was a deliberate choice after noticing early experiments had
suspiciously good validation scores — turned out lag_0 was leaking.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


class FeaturePipeline:
    def __init__(self, target_col: str = "Quantity"):
        self.target_col = target_col
        self.feature_columns: List[str] = []

    def build_grid(self, clean_transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily sales per (Date, StockCode) and fill in missing
        date/product combinations with zero sales. Without this, a product
        that had no sales on a Tuesday would just be absent from the data,
        which breaks lag features.
        """
        df = clean_transactions_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        daily = df.groupby(["Date", "StockCode"]).agg(
            Quantity=("Quantity", "sum"),
            Revenue=("Revenue", "sum"),
            AvgUnitPrice=("UnitPrice", "mean"),
            TransactionsCount=("InvoiceNo", "nunique"),
            Category=("Category", "first"),
            Description=("Description", "first")
        ).reset_index()

        # Cartesian product so every product has an entry for every date
        all_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
        all_skus = daily["StockCode"].unique()
        idx = pd.MultiIndex.from_product([all_dates, all_skus], names=["Date", "StockCode"])
        full_grid = pd.DataFrame(index=idx).reset_index()

        merged = pd.merge(full_grid, daily, on=["Date", "StockCode"], how="left")
        merged["Quantity"] = merged["Quantity"].fillna(0)
        merged["Revenue"] = merged["Revenue"].fillna(0)
        merged["TransactionsCount"] = merged["TransactionsCount"].fillna(0)

        # Forward-fill product metadata (price, category, description)
        sku_meta = daily.groupby("StockCode").agg(
            Category=("Category", "first"),
            Description=("Description", "first"),
            GlobalAvgPrice=("AvgUnitPrice", "mean")
        ).reset_index()

        merged = pd.merge(
            merged.drop(columns=["Category", "Description", "AvgUnitPrice"]),
            sku_meta, on="StockCode", how="left"
        )
        merged["AvgUnitPrice"] = merged["GlobalAvgPrice"]
        merged = merged.drop(columns=["GlobalAvgPrice"])
        merged = merged.sort_values(["StockCode", "Date"]).reset_index(drop=True)
        return merged

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["DayOfWeek"]  = df["Date"].dt.dayofweek
        df["Month"]      = df["Date"].dt.month
        df["Quarter"]    = df["Date"].dt.quarter
        df["DayOfMonth"] = df["Date"].dt.day
        df["DayOfYear"]  = df["Date"].dt.dayofyear
        df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)

        # Cyclical encoding so Mon and Sun are "close" in the feature space
        df["Sin_DayOfWeek"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
        df["Cos_DayOfWeek"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
        df["Sin_Month"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Cos_Month"] = np.cos(2 * np.pi * df["Month"] / 12)
        return df

    def add_holiday_and_promotions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag the major commercial events. Dates are approximate — Black Friday
        moves around each year, but the Nov 24-29 window catches it reliably
        in most years.
        """
        month = df["Month"]
        day = df["DayOfMonth"]

        df["IsBlackFriday"]  = ((month == 11) & (day >= 24) & (day <= 29)).astype(int)
        df["IsCyberWeek"]    = ((month == 12) & (day >= 1)  & (day <= 4)).astype(int)
        df["IsChristmasRush"]= ((month == 12) & (day >= 15) & (day <= 23)).astype(int)
        df["IsNewYear"]      = (((month == 12) & (day >= 31)) | ((month == 1) & (day <= 2))).astype(int)
        df["IsSummerSale"]   = ((month == 7)  & (day >= 10) & (day <= 20)).astype(int)

        # Combined flag — useful as a single "is any promo running" signal
        df["IsPromotion"] = (
            df["IsBlackFriday"] | df["IsCyberWeek"] | df["IsChristmasRush"] | df["IsSummerSale"]
        ).astype(int)
        return df

    def add_lag_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lag and rolling features. All shifts are >= 1 day, so the model
        never sees today's actual sales when predicting today's demand.

        Using shift(1) inside the rolling window (not after) ensures the
        window only covers past days, not the current one.
        """
        grouped = df.groupby("StockCode")[self.target_col]

        for lag in [1, 7, 14, 28]:
            df[f"Lag_{lag}"] = grouped.shift(lag)

        for window in [7, 14, 28]:
            df[f"RollingMean_{window}"] = df.groupby("StockCode")["Quantity"].transform(
                lambda s: s.shift(1).rolling(window, min_periods=3).mean()
            )
            df[f"RollingStd_{window}"] = df.groupby("StockCode")["Quantity"].transform(
                lambda s: s.shift(1).rolling(window, min_periods=3).std()
            )

        df["RollingMin_7"] = df.groupby("StockCode")["Quantity"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=3).min()
        )
        df["RollingMax_7"] = df.groupby("StockCode")["Quantity"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=3).max()
        )

        # Previous 30-day demand total — captures monthly seasonality
        df["PrevMonthDemand"] = df.groupby("StockCode")["Quantity"].transform(
            lambda s: s.shift(1).rolling(30, min_periods=7).sum()
        )

        # Demand volatility ratio — high values signal noisy/unpredictable SKUs
        df["DemandVolatility_7"] = df["RollingStd_7"] / (df["RollingMean_7"] + 1e-5)
        return df

    def add_product_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode product category and add log-transformed price."""
        categories = df["Category"].unique()
        for cat in sorted(categories):
            col_name = "Cat_" + "".join(c if c.isalnum() else "_" for c in cat)
            df[col_name] = (df["Category"] == cat).astype(int)

        df["LogUnitPrice"] = np.log1p(df["AvgUnitPrice"])
        return df

    def transform(self, clean_transactions_df: pd.DataFrame) -> pd.DataFrame:
        print("Building feature grid from clean transactions...")
        grid = self.build_grid(clean_transactions_df)
        grid = self.add_calendar_features(grid)
        grid = self.add_holiday_and_promotions(grid)
        grid = self.add_lag_and_rolling_features(grid)
        grid = self.add_product_embeddings(grid)

        # Drop the first ~28 days per product where lag features are NaN
        initial_count = len(grid)
        valid_df = grid.dropna().copy().reset_index(drop=True)
        print(f"Dropped {initial_count - len(valid_df)} burn-in rows. Remaining: {len(valid_df)}")

        exclude_cols = ["Date", "StockCode", "Description", "Category", "Quantity", "Revenue", "TransactionsCount"]
        self.feature_columns = [c for c in valid_df.columns if c not in exclude_cols]
        return valid_df


if __name__ == "__main__":
    import os
    clean_path = "/working_dir/smart-demand-forecasting/data/processed/cleaned_transactions.csv"
    out_path = "/working_dir/smart-demand-forecasting/data/processed/daily_demand_features.csv"

    clean_df = pd.read_csv(clean_path)
    pipe = FeaturePipeline()
    featured_df = pipe.transform(clean_df)
    featured_df.to_csv(out_path, index=False)
    print(f"Saved features to {out_path}")
    print(f"Total feature columns: {len(pipe.feature_columns)}")
    print(f"First 10: {pipe.feature_columns[:10]}")
