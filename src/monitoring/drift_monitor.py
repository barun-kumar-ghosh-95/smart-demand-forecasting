"""
Model & Data Drift Monitoring Module.
Tracks:
- Forecast bias tracking signal (TS = Running Sum of Forecast Errors / MAD)
- Rolling MAE and RMSE drift
- Mean sales distribution shift detection
- Generates monitoring status alerts (GREEN, YELLOW, RED)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

class DriftMonitor:
    def __init__(self, test_predictions_path: str):
        self.test_predictions_path = test_predictions_path

    def run_drift_audit(self) -> Dict[str, Any]:
        df = pd.read_csv(self.test_predictions_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        df["Error"] = df["Pred_GBDT"] - df["Quantity"]
        df["AbsError"] = np.abs(df["Error"])

        # Tracking signal: RSFE / MAD
        rsfe = df["Error"].sum()
        mad = df["AbsError"].mean()
        tracking_signal = float(rsfe / (mad + 1e-5))

        # Rolling error stats
        overall_mae = float(df["AbsError"].mean())
        overall_rmse = float(np.sqrt(np.mean(df["Error"] ** 2)))

        # Split into early test vs late test
        mid_point = len(df) // 2
        early_df = df.iloc[:mid_point]
        late_df = df.iloc[mid_point:]

        early_mae = float(early_df["AbsError"].mean())
        late_mae = float(late_df["AbsError"].mean())
        mae_shift_pct = round(((late_mae - early_mae) / (early_mae + 1e-5)) * 100, 2)

        # Status categorization
        # In inventory control, a tracking signal within [-4, +4] is considered in control
        if abs(tracking_signal) > 6.0 or abs(mae_shift_pct) > 30.0:
            status = "RED (Severe Drift Detected - Retraining Recommended)"
        elif abs(tracking_signal) > 4.0 or abs(mae_shift_pct) > 15.0:
            status = "YELLOW (Moderate Shift - Monitor Closely)"
        else:
            status = "GREEN (Model In Statistical Process Control)"

        sku_alerts = []
        for sku, group in df.groupby("StockCode"):
            sku_ts = group["Error"].sum() / (group["AbsError"].mean() + 1e-5)
            sku_alerts.append({
                "sku": sku,
                "tracking_signal": round(float(sku_ts), 2),
                "in_control": bool(abs(sku_ts) <= 4.0),
                "mae": round(float(group["AbsError"].mean()), 2)
            })

        return {
            "monitoring_status": status,
            "overall_mae": round(overall_mae, 2),
            "overall_rmse": round(overall_rmse, 2),
            "tracking_signal": round(tracking_signal, 2),
            "mae_shift_pct": mae_shift_pct,
            "sku_level_health": sku_alerts
        }
