"""
Inference Service Module.
Loads serialized models and provides multi-step recursive or direct forecasting,
prediction confidence intervals, and inventory replenishment recommendations.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from src.inventory.optimizer import InventoryOptimizer

class DemandPredictor:
    def __init__(self, artifact_path: str, features_path: str):
        self.artifact_path = artifact_path
        self.features_path = features_path
        self.model_data = None
        self.feature_cols = []
        self.model = None
        self.sku_errors = {}
        self.sku_meta = {}
        self.history_df = None
        self.optimizer = InventoryOptimizer(default_lead_time_days=7, service_level=0.95)
        self._load()

    def _load(self):
        if not os.path.exists(self.artifact_path):
            raise FileNotFoundError(f"Model artifact not found at {self.artifact_path}")
        with open(self.artifact_path, "rb") as f:
            self.model_data = pickle.load(f)
        self.model = self.model_data["boosted_model"]
        self.feature_cols = self.model_data["feature_cols"]
        self.sku_errors = self.model_data.get("sku_errors", {})
        self.sku_meta = self.model_data.get("sku_meta", {})

        if os.path.exists(self.features_path):
            self.history_df = pd.read_csv(self.features_path)
            self.history_df["Date"] = pd.to_datetime(self.history_df["Date"])

    def predict_product_horizon(
        self,
        sku: str,
        horizon_days: int = 7,
        current_inventory: Optional[float] = None,
        lead_time_days: int = 7
    ) -> Dict[str, Any]:
        if sku not in self.sku_meta:
            available_skus = list(self.sku_meta.keys())
            raise ValueError(f"SKU '{sku}' not recognized. Available SKUs: {available_skus}")

        meta = self.sku_meta[sku]
        sku_df = self.history_df[self.history_df["StockCode"] == sku].sort_values("Date")
        recent_features = sku_df.tail(horizon_days)[self.feature_cols].values

        if len(recent_features) < horizon_days:
            # Pad if needed
            pad = np.tile(recent_features[-1:], (horizon_days - len(recent_features), 1))
            recent_features = np.vstack([recent_features, pad])

        # Point predictions
        point_preds = self.model.predict(recent_features[:horizon_days])
        point_preds = np.round(point_preds, 1)

        # Confidence intervals (95% CI: +/- 1.96 * sigma_error)
        err_std = self.sku_errors.get(sku, 5.0)
        ci_half = 1.96 * err_std
        lower_bounds = np.round(np.clip(point_preds - ci_half, 0, None), 1)
        upper_bounds = np.round(point_preds + ci_half, 1)

        # Dates
        last_date = sku_df["Date"].max()
        forecast_dates = [(last_date + pd.Timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(horizon_days)]

        daily_breakdown = []
        for i in range(horizon_days):
            daily_breakdown.append({
                "date": forecast_dates[i],
                "predicted_quantity": float(point_preds[i]),
                "ci_lower": float(lower_bounds[i]),
                "ci_upper": float(upper_bounds[i])
            })

        total_predicted = float(np.sum(point_preds))
        
        # If current inventory is provided, calculate inventory risk
        inv_eval = None
        if current_inventory is not None:
            inv_eval = self.optimizer.evaluate_sku_inventory(
                sku=sku,
                current_inventory=current_inventory,
                daily_forecast=point_preds,
                forecast_error_std=err_std,
                lead_time_days=lead_time_days
            )

        return {
            "sku": sku,
            "description": meta["Description"],
            "category": meta["Category"],
            "unit_price": meta["AvgUnitPrice"],
            "forecast_horizon_days": horizon_days,
            "total_predicted_units": round(total_predicted, 1),
            "forecast_error_std": round(err_std, 2),
            "daily_forecast": daily_breakdown,
            "inventory_assessment": inv_eval
        }
