import unittest
import pandas as pd
import tempfile
import os
from src.data.cleaner import DataCleaner

class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        sample_data = {
            "InvoiceNo": ["50001", "C50002", "50003", "50003", "50004"],
            "StockCode": ["SKU_1", "SKU_1", "SKU_2", "SKU_2", "SKU_3"],
            "Description": ["Item 1", "Item 1", "Item 2", "Item 2", "Item 3"],
            "Quantity": [10, -5, 4, 4, 0],
            "InvoiceDate": ["2024-01-01 10:00:00", "2024-01-01 11:00:00", "2024-01-02 12:00:00", "2024-01-02 12:00:00", "2024-01-03 14:00:00"],
            "UnitPrice": [15.0, 15.0, 20.0, 20.0, 10.0],
            "CustomerID": ["CUST_1", "CUST_1", None, None, "CUST_2"],
            "Country": ["UK", "UK", "UK", "UK", "UK"],
            "StoreRegion": ["North", "North", "North", "North", "North"],
            "Category": ["General", "General", "General", "General", "General"]
        }
        pd.DataFrame(sample_data).to_csv(self.temp_file.name, index=False)

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_cleaning_pipeline(self):
        cleaner = DataCleaner(self.temp_file.name)
        df_clean, audit = cleaner.clean()

        # Deduplication: Row 4 is duplicate of Row 3 -> 1 duplicate removed
        self.assertEqual(audit["duplicates_removed"], 1)
        # Cancellations / non-positive quantities:
        # C50002 (-5) is cancellation
        # 50004 (qty=0) is invalid
        # Cleaned dataset should only contain 50001 and 50003
        self.assertEqual(len(df_clean), 2)
        self.assertTrue(all(df_clean["Quantity"] > 0))
        self.assertFalse(any(df_clean["InvoiceNo"].str.startswith("C")))
        # Missing CustomerID imputed
        self.assertEqual(df_clean.loc[df_clean["InvoiceNo"] == "50003", "CustomerID"].iloc[0], "GUEST_CHECKOUT")
        # Revenue check
        self.assertEqual(df_clean.loc[df_clean["InvoiceNo"] == "50001", "Revenue"].iloc[0], 150.0)

if __name__ == "__main__":
    unittest.main()
