import unittest
import pandas as pd
import numpy as np
from src.features.feature_pipeline import FeaturePipeline

class TestFeaturePipeline(unittest.TestCase):
    def test_feature_generation_no_leakage(self):
        dates = pd.date_range("2024-01-01", periods=45, freq="D")
        records = []
        for i, d in enumerate(dates):
            records.append({
                "InvoiceNo": f"INV_{i}",
                "StockCode": "SKU_TEST",
                "Description": "Test Product",
                "Quantity": 10 + (i % 5),
                "InvoiceDate": d.strftime("%Y-%m-%d 10:00:00"),
                "Date": d.strftime("%Y-%m-%d"),
                "UnitPrice": 25.0,
                "CustomerID": "CUST_1",
                "Country": "UK",
                "StoreRegion": "North",
                "Category": "Electronics",
                "Revenue": (10 + (i % 5)) * 25.0
            })
        df = pd.DataFrame(records)
        pipeline = FeaturePipeline()
        featured = pipeline.transform(df)

        self.assertGreater(len(featured), 0)
        self.assertIn("Lag_1", featured.columns)
        self.assertIn("Lag_7", featured.columns)
        self.assertIn("RollingMean_7", featured.columns)
        self.assertIn("IsWeekend", featured.columns)
        self.assertIn("IsBlackFriday", featured.columns)

        # Verify no NaN values
        self.assertEqual(featured.isna().sum().sum(), 0)

if __name__ == "__main__":
    unittest.main()
