"""
Inventory Optimization and Stockout Risk Module.
Implements:
- Safety Stock: SS = Z * sigma_error * sqrt(lead_time)
- Reorder Point (ROP): ROP = Lead_Time_Demand + Safety_Stock
- Stockout Risk Indicator: (Predicted Demand over Lead Time > Current Inventory)
- Recommended Reorder Quantity: max(0, ROP - Current Inventory)
- Simulated Inventory Layer for backtesting and demonstration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

class InventoryOptimizer:
    # Standard normal quantile mapping for service levels
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
        
        # Demand over lead time window
        lt_forecast = daily_forecast[:lt]
        lead_time_demand = float(np.sum(lt_forecast))
        
        # Safety Stock
        safety_stock = self.calculate_safety_stock(forecast_error_std, lt)
        
        # Reorder Point (ROP)
        reorder_point = float(lead_time_demand + safety_stock)
        
        # Stockout Risk condition
        is_stockout_risk = current_inventory < lead_time_demand
        is_reorder_triggered = current_inventory <= reorder_point
        
        # Recommended reorder quantity
        reorder_qty = float(np.ceil(max(0.0, reorder_point - current_inventory)))
        
        # Days of Supply remaining based on average daily forecast
        avg_daily_demand = float(np.mean(daily_forecast)) if len(daily_forecast) > 0 else 1.0
        days_of_supply = round(current_inventory / (avg_daily_demand + 1e-5), 1)

        # Risk severity
        if current_inventory <= 0:
            severity = "CRITICAL (Stockout)"
        elif current_inventory < lead_time_demand:
            severity = "HIGH (Stockout Imminent)"
        elif current_inventory <= reorder_point:
            severity = "MEDIUM (Reorder Needed)"
        else:
            severity = "LOW (Healthy)"

        return {
            "sku": sku,
            "current_inventory": int(current_inventory),
            "lead_time_days": lt,
            "lead_time_predicted_demand": round(lead_time_demand, 1),
            "safety_stock": int(safety_stock),
            "reorder_point": int(reorder_point),
            "stockout_risk": bool(is_stockout_risk),
            "reorder_needed": bool(is_reorder_triggered),
            "recommended_reorder_qty": int(reorder_qty),
            "days_of_supply": days_of_supply,
            "risk_severity": severity,
            "inventory_source": "SIMULATED_LAYER"
        }

    def generate_simulated_inventory(
        self,
        skus: List[str],
        avg_daily_demands: Dict[str, float],
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Creates a realistic simulated inventory snapshot for all SKUs,
        with varying buffer levels (some well-stocked, some near stockout).
        """
        np.random.seed(seed)
        records = []
        for sku in skus:
            avg_d = avg_daily_demands.get(sku, 30.0)
            # Random days of stock between 3 days (stockout risk) and 25 days (overstocked)
            simulated_days = np.random.uniform(3.5, 20.0)
            curr_stock = int(np.round(simulated_days * avg_d))
            records.append({
                "StockCode": sku,
                "CurrentInventory": curr_stock,
                "LeadTimeDays": self.lead_time_days,
                "UnitCost": round(avg_d * 0.6, 2)
            })
        return pd.DataFrame(records)
