"""
Inventory optimization: safety stock, reorder points, stockout risk.

Core formula:  SS = Z * sigma_error * sqrt(lead_time)
               ROP = lead_time_demand + SS

Using per-SKU forecast error standard deviation (sigma_error) rather than
a global average makes a meaningful difference for products with very different
demand volatility profiles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


class InventoryOptimizer:
    # Z-scores for common service level targets
    Z_SCORES = {
        0.90: 1.282,
        0.95: 1.645,
        0.975: 1.960,
        0.99: 2.326
    }

    def __init__(self, default_lead_time_days: int = 7, service_level: float = 0.95):
        self.lead_time_days = default_lead_time_days
        self.service_level = service_level
        self.z_score = self.Z_SCORES.get(service_level, 1.645)

    def calculate_safety_stock(self, forecast_error_std: float, lead_time_days: Optional[int] = None) -> float:
        lt = lead_time_days or self.lead_time_days
        ss = self.z_score * forecast_error_std * np.sqrt(lt)
        return float(np.ceil(max(0.0, ss)))

    def evaluate_sku_inventory(
        self,
        sku: str,
        current_inventory: float,
        daily_forecast: np.ndarray,
        forecast_error_std: float,
        lead_time_days: Optional[int] = None
    ) -> Dict[str, Any]:
        lt = lead_time_days or self.lead_time_days

        lt_forecast   = daily_forecast[:lt]
        lt_demand     = float(np.sum(lt_forecast))
        safety_stock  = self.calculate_safety_stock(forecast_error_std, lt)
        reorder_point = lt_demand + safety_stock

        is_stockout_risk   = current_inventory < lt_demand
        is_reorder_needed  = current_inventory <= reorder_point
        reorder_qty        = float(np.ceil(max(0.0, reorder_point - current_inventory)))

        avg_daily = float(np.mean(daily_forecast)) if len(daily_forecast) > 0 else 1.0
        days_of_supply = round(current_inventory / (avg_daily + 1e-5), 1)

        if current_inventory <= 0:
            severity = "CRITICAL (Stockout)"
        elif current_inventory < lt_demand:
            severity = "HIGH (Stockout Imminent)"
        elif current_inventory <= reorder_point:
            severity = "MEDIUM (Reorder Needed)"
        else:
            severity = "LOW (Healthy)"

        return {
            "sku":                      sku,
            "current_inventory":        int(current_inventory),
            "lead_time_days":           lt,
            "lead_time_predicted_demand": round(lt_demand, 1),
            "safety_stock":             int(safety_stock),
            "reorder_point":            int(reorder_point),
            "stockout_risk":            bool(is_stockout_risk),
            "reorder_needed":           bool(is_reorder_needed),
            "recommended_reorder_qty":  int(reorder_qty),
            "days_of_supply":           days_of_supply,
            "risk_severity":            severity,
            "inventory_source":         "SIMULATED_LAYER"
        }

    def generate_simulated_inventory(
        self,
        skus: List[str],
        avg_daily_demands: Dict[str, float],
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Create a simulated inventory snapshot for all SKUs.
        Randomizes days-of-stock between 3.5 and 20 days so we get a
        realistic mix of stockout-risk and healthy products.
        """
        np.random.seed(seed)
        records = []
        for sku in skus:
            avg_d = avg_daily_demands.get(sku, 30.0)
            sim_days  = np.random.uniform(3.5, 20.0)
            curr_stock = int(np.round(sim_days * avg_d))
            records.append({
                "StockCode":       sku,
                "CurrentInventory": curr_stock,
                "LeadTimeDays":    self.lead_time_days,
                "UnitCost":        round(avg_d * 0.6, 2)
            })
        return pd.DataFrame(records)
