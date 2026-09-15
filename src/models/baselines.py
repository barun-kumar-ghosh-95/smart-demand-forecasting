"""
Baseline Forecasting Models.
Provides standard benchmark references:
1. Last Period (Naive Lag-1)
2. Moving Average (7-Day Rolling Mean)
3. Seasonal Naive (Lag-7: Same day last week)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class BaselineForecaster:
    @staticmethod
    def predict_last_period(df: pd.DataFrame) -> np.ndarray:
        """Predicts yesterday's demand (Lag_1)."""
        if "Lag_1" in df.columns:
            return df["Lag_1"].values
        return df.groupby("StockCode")["Quantity"].shift(1).fillna(0).values

    @staticmethod
    def predict_moving_average_7(df: pd.DataFrame) -> np.ndarray:
        """Predicts average of past 7 days."""
        if "RollingMean_7" in df.columns:
            return df["RollingMean_7"].values
        return df.groupby("StockCode")["Quantity"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=1).mean()
        ).values

    @staticmethod
    def predict_seasonal_naive(df: pd.DataFrame) -> np.ndarray:
        """Predicts demand from same day last week (Lag_7)."""
        if "Lag_7" in df.columns:
            return df["Lag_7"].values
        return df.groupby("StockCode")["Quantity"].shift(7).fillna(0).values
