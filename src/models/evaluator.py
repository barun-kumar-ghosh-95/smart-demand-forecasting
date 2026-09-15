"""
Evaluation metrics module for retail demand forecasting.
Computes:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- sMAPE (Symmetric Mean Absolute Percentage Error)
- WMAPE (Weighted Mean Absolute Percentage Error)
- Forecast Bias (Normalized tracking signal / under-over prediction percentage)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-5) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        # Demand cannot be negative in retail; clip predictions at 0
        y_pred = np.clip(y_pred, 0, None)

        errors = y_pred - y_true
        abs_errors = np.abs(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        # sMAPE: 200 * |y - y_hat| / (|y| + |y_hat| + eps)
        denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
        smape = float(np.mean(2.0 * abs_errors / denominator) * 100.0)

        # WMAPE: sum(|y - y_hat|) / sum(y)
        sum_y = float(np.sum(y_true))
        wmape = float((np.sum(abs_errors) / (sum_y + epsilon)) * 100.0)

        # Forecast Bias: sum(y_hat - y) / sum(y)
        # Positive bias means overforecasting; negative bias means underforecasting (stockout hazard)
        bias = float((np.sum(errors) / (sum_y + epsilon)) * 100.0)

        return {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "sMAPE": round(smape, 2),
            "WMAPE": round(wmape, 2),
            "Bias": round(bias, 2)
        }

    @staticmethod
    def evaluate_by_segment(df: pd.DataFrame, true_col: str, pred_col: str, group_col: str = "StockCode") -> pd.DataFrame:
        results = []
        for name, group in df.groupby(group_col):
            m = ModelEvaluator.calculate_metrics(group[true_col].values, group[pred_col].values)
            m[group_col] = name
            results.append(m)
        res_df = pd.DataFrame(results)
        # Rearrange columns
        cols = [group_col, "MAE", "RMSE", "sMAPE", "WMAPE", "Bias"]
        return res_df[cols]
