"""
Model training and experiment orchestration.

Runs the full training pipeline:
  - Chronological split (train / val / test)
  - Baseline models for comparison
  - Ridge regression and gradient boosted forecaster
  - Holiday feature ablation experiment
  - Inventory safety stock and stockout analysis
  - Exports trained model + metrics for the API

The test set is intentionally Q4 (Nov-Dec) to capture peak retail volatility.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.models.baselines import BaselineForecaster
from src.models.ml_models import RidgeRegressionForecaster, RetailForecastingModel
from src.models.evaluator import ModelEvaluator
from src.inventory.optimizer import InventoryOptimizer


class DemandModelTrainer:
    def __init__(self, features_path: str, artifacts_dir: str):
        self.features_path = features_path
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("Loading feature data...")
        df = pd.read_csv(self.features_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Date", "StockCode"]).reset_index(drop=True)

        # Split: Train → Val → Test (Q4 2024 holiday peak)
        # Keeping the test set as Q4 is deliberate — it's the hardest period
        # to forecast and gives the most informative benchmark.
        train_mask = df["Date"] <= "2024-08-31"
        val_mask   = (df["Date"] > "2024-08-31") & (df["Date"] <= "2024-10-31")
        test_mask  = df["Date"] > "2024-10-31"

        train_df = df[train_mask].copy()
        val_df   = df[val_mask].copy()
        test_df  = df[test_mask].copy()

        print(f"Split sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        exclude_cols = ["Date", "StockCode", "Description", "Category", "Quantity", "Revenue", "TransactionsCount"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X_train = train_df[feature_cols].values
        y_train = train_df["Quantity"].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df["Quantity"].values

        results = {}
        predictions = {}

        # --- Baselines ---
        print("\nEvaluating baselines...")
        pred_lag1 = BaselineForecaster.predict_last_period(test_df)
        results["Baseline: Last Period"]          = ModelEvaluator.calculate_metrics(y_test, pred_lag1)
        predictions["baseline_last_period"]       = pred_lag1

        pred_ma7 = BaselineForecaster.predict_moving_average_7(test_df)
        results["Baseline: 7-Day Moving Avg"]     = ModelEvaluator.calculate_metrics(y_test, pred_ma7)
        predictions["baseline_ma7"]               = pred_ma7

        pred_snaive = BaselineForecaster.predict_seasonal_naive(test_df)
        results["Baseline: Seasonal Naive (Lag 7)"] = ModelEvaluator.calculate_metrics(y_test, pred_snaive)
        predictions["baseline_seasonal_naive"]    = pred_snaive

        # --- ML models ---
        print("\nTraining ML models...")
        ridge = RidgeRegressionForecaster(alpha=2.5)
        ridge.fit(X_train, y_train, feature_names=feature_cols)
        pred_ridge = ridge.predict(X_test)
        results["ML: Regularized Linear (Ridge)"] = ModelEvaluator.calculate_metrics(y_test, pred_ridge)
        predictions["ml_ridge"]                   = pred_ridge

        boosted = RetailForecastingModel(model_type="boosted")
        boosted.fit(X_train, y_train, feature_names=feature_cols)
        pred_boosted = boosted.predict(X_test)
        results["ML: Gradient Boosted Forecaster"] = ModelEvaluator.calculate_metrics(y_test, pred_boosted)
        predictions["ml_boosted"]                  = pred_boosted

        # --- Festival feature ablation ---
        # Remove holiday flags and retrain to quantify their contribution
        print("\nRunning festival feature ablation...")
        festival_cols = ["IsBlackFriday", "IsCyberWeek", "IsChristmasRush", "IsNewYear", "IsSummerSale", "IsPromotion"]
        features_no_fest = [c for c in feature_cols if c not in festival_cols]

        boosted_no_fest = RetailForecastingModel(model_type="boosted")
        boosted_no_fest.fit(train_df[features_no_fest].values, y_train, feature_names=features_no_fest)
        pred_no_fest = boosted_no_fest.predict(test_df[features_no_fest].values)
        results["Ablation: GBDT (Without Festival Features)"] = ModelEvaluator.calculate_metrics(y_test, pred_no_fest)

        # --- Save test predictions ---
        test_out = test_df[["Date", "StockCode", "Description", "Category", "Quantity"]].copy()
        test_out["Pred_Seasonal_Naive"]      = np.round(pred_snaive, 1)
        test_out["Pred_Moving_Avg_7"]        = np.round(pred_ma7, 1)
        test_out["Pred_Ridge"]               = np.round(pred_ridge, 1)
        test_out["Pred_GBDT"]                = np.round(pred_boosted, 1)
        test_out["Pred_GBDT_No_Festivals"]   = np.round(pred_no_fest, 1)

        test_pred_path = os.path.join(self.artifacts_dir, "test_predictions.csv")
        test_out.to_csv(test_pred_path, index=False)
        print(f"Saved test predictions → {test_pred_path}")

        # --- Inventory analysis ---
        print("\nCalculating stockout risk and safety stock per SKU...")
        optimizer = InventoryOptimizer(default_lead_time_days=7, service_level=0.95)

        error_df = test_out.copy()
        error_df["Error"] = error_df["Quantity"] - error_df["Pred_GBDT"]
        sku_errors    = error_df.groupby("StockCode")["Error"].std().to_dict()
        sku_avg_demand = train_df.groupby("StockCode")["Quantity"].mean().to_dict()

        all_skus = sorted(df["StockCode"].unique())
        sim_inv_df = optimizer.generate_simulated_inventory(all_skus, sku_avg_demand)

        inventory_records = []
        for _, row in sim_inv_df.iterrows():
            sku = row["StockCode"]
            err_std = sku_errors.get(sku, 5.0)

            sku_features = df[df["StockCode"] == sku].tail(7)[feature_cols].values
            sku_forecast  = boosted.predict(sku_features)

            status = optimizer.evaluate_sku_inventory(
                sku=sku,
                current_inventory=row["CurrentInventory"],
                daily_forecast=sku_forecast,
                forecast_error_std=err_std,
                lead_time_days=7
            )
            meta = df[df["StockCode"] == sku].iloc[0]
            status["description"] = meta["Description"]
            status["category"]    = meta["Category"]
            status["unit_price"]  = meta["AvgUnitPrice"]
            inventory_records.append(status)

        inv_df = pd.DataFrame(inventory_records)
        inv_path = os.path.join(self.artifacts_dir, "inventory_status.csv")
        inv_df.to_csv(inv_path, index=False)
        print(f"Saved inventory status → {inv_path}")

        # --- Save model artifacts ---
        model_payload = {
            "feature_cols":  feature_cols,
            "boosted_model": boosted,
            "ridge_model":   ridge,
            "sku_errors":    sku_errors,
            "sku_meta": df.groupby("StockCode").agg(
                Description=("Description", "first"),
                Category=("Category", "first"),
                AvgUnitPrice=("AvgUnitPrice", "first")
            ).to_dict(orient="index"),
            "results": results
        }
        model_path = os.path.join(self.artifacts_dir, "forecasting_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_payload, f)
        print(f"Saved model → {model_path}")

        metrics_path = os.path.join(self.artifacts_dir, "model_comparison_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)

        # Print results summary
        print("\n" + "=" * 55)
        print("          MODEL PERFORMANCE SUMMARY")
        print("=" * 55)
        summary = pd.DataFrame(results).T[["MAE", "RMSE", "sMAPE", "WMAPE", "Bias"]]
        print(summary.to_string())
        print("=" * 55)

        snaive_smape = results["Baseline: Seasonal Naive (Lag 7)"]["sMAPE"]
        gbdt_smape   = results["ML: Gradient Boosted Forecaster"]["sMAPE"]
        lift_pct     = round(((snaive_smape - gbdt_smape) / snaive_smape) * 100, 1)
        print(f"\nsMAPE improvement vs seasonal naive: {lift_pct}%")

        no_fest_smape = results["Ablation: GBDT (Without Festival Features)"]["sMAPE"]
        fest_lift     = round(((no_fest_smape - gbdt_smape) / no_fest_smape) * 100, 1)
        print(f"Festival feature contribution: {fest_lift}% sMAPE improvement")

        return {
            "metrics": results,
            "lift_vs_baseline_pct": lift_pct,
            "festival_lift_pct": fest_lift
        }


if __name__ == "__main__":
    feat_path  = "/working_dir/smart-demand-forecasting/data/processed/daily_demand_features.csv"
    artifacts  = "/working_dir/smart-demand-forecasting/data/processed"
    trainer = DemandModelTrainer(feat_path, artifacts)
    trainer.run()
