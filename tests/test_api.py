import unittest
from starlette.testclient import TestClient
from api.main import app

class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_products_endpoint(self):
        resp = self.client.get("/products")
        self.assertEqual(resp.status_code, 200)
        skus = resp.json()
        self.assertGreater(len(skus), 0)
        self.assertIn("sku", skus[0])

    def test_predict_endpoint(self):
        payload = {
            "sku": "SKU_1001",
            "horizon_days": 7,
            "current_inventory": 150.0,
            "lead_time_days": 7
        }
        resp = self.client.post("/predict/product", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["sku"], "SKU_1001")
        self.assertEqual(len(data["daily_forecast"]), 7)
        self.assertIn("inventory_assessment", data)

    def test_inventory_recommendations(self):
        resp = self.client.get("/inventory/recommendations")
        self.assertEqual(resp.status_code, 200)
        recs = resp.json()
        self.assertGreater(len(recs), 0)
        self.assertIn("reorder_point", recs[0])

    def test_drift_monitoring(self):
        resp = self.client.get("/monitoring/drift")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("tracking_signal", data)
        self.assertIn("monitoring_status", data)

if __name__ == "__main__":
    unittest.main()
