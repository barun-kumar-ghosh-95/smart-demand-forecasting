import unittest
import numpy as np
from src.inventory.optimizer import InventoryOptimizer

class TestInventoryOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = InventoryOptimizer(default_lead_time_days=7, service_level=0.95)

    def test_safety_stock_formula(self):
        # SS = Z * sigma * sqrt(L)
        # Z for 0.95 = 1.645, sigma = 10, L = 7 -> 1.645 * 10 * 2.64575 = ~43.52 -> ceil 44
        ss = self.optimizer.calculate_safety_stock(forecast_error_std=10.0, lead_time_days=7)
        self.assertEqual(ss, 44.0)

    def test_stockout_risk_triggered(self):
        forecast_7d = np.array([20.0] * 7) # Total demand = 140
        res = self.optimizer.evaluate_sku_inventory(
            sku="SKU_TEST",
            current_inventory=50.0, # 50 < 140 -> imminent stockout
            daily_forecast=forecast_7d,
            forecast_error_std=5.0,
            lead_time_days=7
        )
        self.assertTrue(res["stockout_risk"])
        self.assertTrue(res["reorder_needed"])
        self.assertIn("HIGH", res["risk_severity"])
        self.assertGreater(res["recommended_reorder_qty"], 90)

    def test_healthy_inventory(self):
        forecast_7d = np.array([10.0] * 7) # Total demand = 70
        res = self.optimizer.evaluate_sku_inventory(
            sku="SKU_HEALTHY",
            current_inventory=500.0, # plenty of stock
            daily_forecast=forecast_7d,
            forecast_error_std=5.0,
            lead_time_days=7
        )
        self.assertFalse(res["stockout_risk"])
        self.assertFalse(res["reorder_needed"])
        self.assertEqual(res["recommended_reorder_qty"], 0)
        self.assertEqual(res["risk_severity"], "LOW (Healthy)")

if __name__ == "__main__":
    unittest.main()
