"""
Model Training and Experiment Orchestration.
Performs:
- Chronological train/validation/test split
- Trains baselines (Last Period, Moving Average, Seasonal Naive)
- Trains Machine Learning models (Ridge Regression, Gradient Boosted Forecaster)
- Evaluates on out-of-time Test set (including holiday peak)
- Conducts Festival/Promotion feature ablation experiment
- Calculates inventory safety stock and stockout risk recommendations
- Exports model artifacts and metrics for API & Dashboard
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
        print("Loading engineered features...")
        df = pd.read_csv(self.features_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Date", "StockCode"]).reset_index(drop=True)

        # 1. Chronological Split
        # Train: 2023-01-29 to 2024-08-31
        # Validation: 2024-09-01 to 2024-10-31
        # Test: 2024-11-01 to 2024-12-30 (Holiday Peak 2024)
        train_mask = df["Date"] <= "2024-08-31"
        val_mask = (df["Date"] > "2024-08-31") & (df["Date"] <= "2024-10-31")
        test_mask = df["Date"] > "2024-10-31"

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        print(f"Data Splits: Train={len(train_df)} rows, Val={len(val_df)} rows, Test={len(test_df)} rows")

        # Exclude metadata and target columns from features
        exclude_cols = ["Date", "StockCode", "Description", "Category", "Quantity", "Revenue", "TransactionsCount"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X_train = train_df[feature_cols].values
        y_train = train_df["Quantity"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["Quantity"].values

        results = {}
        predictions = {}

        # 2. Baseline Models
        print("\nEvaluating Baseline Models...")
        pred_last_period = BaselineForecaster.predict_last_period(test_df)
        results["Baseline: Last Period"] = ModelEvaluator.calculate_metrics(y_test, pred_last_period)
        predictions["baseline_last_period"] = pred_last_period

        pred_ma7 = BaselineForecaster.predict_moving_average_7(test_df)
        results["Baseline: 7-Day Moving Avg"] = ModelEvaluator.calculate_metrics(y_test, pred_ma7)
        predictions["baseline_ma7"] = pred_ma7

        pred_snaive = BaselineForecaster.predict_seasonal_naive(test_df)
        results["Baseline: Seasonal Naive (Lag 7)"] = ModelEvaluator.calculate_metrics(y_test, pred_snaive)
        predictions["baseline_seasonal_naive"] = pred_snaive

        # 3. Machine Learning Models
        print("\nTraining ML Models...")
        # Ridge Regression
        ridge = RidgeRegressionForecaster(alpha=2.5)
        ridge.fit(X_train, y_train, feature_names=feature_cols)
        pred_ridge = ridge.predict(X_test)
        results["ML: Regularized Linear (Ridge)"] = ModelEvaluator.calculate_metrics(y_test, pred_ridge)
        predictions["ml_ridge"] = pred_ridge

        # Boosted Forecaster
        boosted = RetailForecastingModel(model_type="boosted")
        boosted.fit(X_train, y_train, feature_names=feature_cols)
        pred_boosted = boosted.predict(X_test)
        results["ML: Gradient Boosted Forecaster"] = ModelEvaluator.calculate_metrics(y_test, pred_boosted)
        predictions["ml_boosted"] = pred_boosted

        # 4. Seasonality & Festival Upgrade Ablation
        print("\nRunning Seasonality & Festival Feature Ablation...")
        festival_cols = ["IsBlackFriday", "IsCyberWeek", "IsChristmasRush", "IsNewYear", "IsSummerSale", "IsPromotion"]
        features_no_festival = [c for c in feature_cols if c not in festival_cols]

        X_train_no_fest = train_df[features_no_festival].values
        X_test_no_fest = test_df[features_no_festival].values

        boosted_no_fest = RetailForecastingModel(model_type="boosted")
        boosted_no_fest.fit(X_train_no_fest, y_train, feature_names=features_no_festival)
        pred_no_fest = boosted_no_fest.predict(X_test_no_fest)
        results["Ablation: GBDT (Without Festival Features)"] = ModelEvaluator.calculate_metrics(y_test, pred_no_fest)

        # 5. Save Test Predictions DataFrame
        test_out_df = test_df[["Date", "StockCode", "Description", "Category", "Quantity"]].copy()
        test_out_df["Pred_Seasonal_Naive"] = np.round(pred_snaive, 1)
        test_out_df["Pred_Moving_Avg_7"] = np.round(pred_ma7, 1)
        test_out_df["Pred_Ridge"] = np.round(pred_ridge, 1)
        test_out_df["Pred_GBDT"] = np.round(pred_boosted, 1)
        test_out_df["Pred_GBDT_No_Festivals"] = np.round(pred_no_fest, 1)

        test_pred_path = os.path.join(self.artifacts_dir, "test_predictions.csv")
        test_out_df.to_csv(test_pred_path, index=False)
        print(f"Test predictions saved to {test_pred_path}")

        # 6. Inventory Safety Stock and Stockout Analysis
        print("\nCalculating Inventory Safety Stock and Stockout Risk...")
        optimizer = InventoryOptimizer(default_lead_time_days=7, service_level=0.95)
        
        error_df = test_out_df.copy()
        error_df["Error"] = error_df["Quantity"] - error_df["Pred_GBDT"]
        sku_errors = error_df.groupby("StockCode")["Error"].std().to_dict()
        sku_avg_demand = train_df.groupby("StockCode")["Quantity"].mean().to_dict()

        all_skus = sorted(df["StockCode"].unique())
        sim_inv_df = optimizer.generate_simulated_inventory(all_skus, sku_avg_demand)

        inventory_status_list = []
        for _, row in sim_inv_df.iterrows():
            sku = row["StockCode"]
            curr_stock = row["CurrentInventory"]
            err_std = sku_errors.get(sku, 5.0)
            
            sku_recent_features = df[df["StockCode"] == sku].tail(7)[feature_cols].values
            sku_7d_forecast = boosted.predict(sku_recent_features)
            
            status = optimizer.evaluate_sku_inventory(
                sku=sku,
                current_inventory=curr_stock,
                daily_forecast=sku_7d_forecast,
                forecast_error_std=err_std,
                lead_time_days=7
            )
            meta = df[df["StockCode"] == sku].iloc[0]
            status["description"] = meta["Description"]
            status["category"] = meta["Category"]
            status["unit_price"] = meta["AvgUnitPrice"]
            inventory_status_list.append(status)

        inv_df = pd.DataFrame(inventory_status_list)
        inv_path = os.path.join(self.artifacts_dir, "inventory_status.csv")
        inv_df.to_csv(inv_path, index=False)
        print(f"Inventory status saved to {inv_path}")

        # 7. Save Models and Metadata
        model_payload = {
            "feature_cols": feature_cols,
            "boosted_model": boosted,
            "ridge_model": ridge,
            "sku_errors": sku_errors,
            "sku_meta": df.groupby("StockCode").agg(
                Description=("Description", "first"),
                Category=("Category", "first"),
                AvgUnitPrice=("AvgUnitPrice", "first")
            ).to_dict(orient="index"),
            "results": results
        }
        model_artifact_path = os.path.join(self.artifacts_dir, "forecasting_model.pkl")
        with open(model_artifact_path, "wb") as f:
            pickle.dump(model_payload, f)
        print(f"Trained model artifacts saved to {model_artifact_path}")

        metrics_path = os.path.join(self.artifacts_dir, "model_comparison_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)

        print("\n=======================================================")
        print("               MODEL PERFORMANCE SUMMARY               ")
        print("=======================================================")
        summary_table = pd.DataFrame(results).T[["MAE", "RMSE", "sMAPE", "WMAPE", "Bias"]]
        print(summary_table.to_string())
        print("=======================================================")

        snaive_smape = results["Baseline: Seasonal Naive (Lag 7)"]["sMAPE"]
        gbdt_smape = results["ML: Gradient Boosted Forecaster"]["sMAPE"]
        pct_reduction = round(((snaive_smape - gbdt_smape) / snaive_smape) * 100, 1)
        print(f"\nModel Lift: Reduced sMAPE by {pct_reduction}% vs. Seasonal Naive baseline!")

        no_fest_smape = results["Ablation: GBDT (Without Festival Features)"]["sMAPE"]
        fest_lift = round(((no_fest_smape - gbdt_smape) / no_fest_smape) * 100, 1)
        print(f"Festival Feature Lift: Adding Holiday/Festival features improved sMAPE by {fest_lift}% during test period!")

        return {
            "metrics": results,
            "lift_vs_baseline_pct": pct_reduction,
            "festival_lift_pct": fest_lift
        }

if __name__ == "__main__":
    feat_path = "/working_dir/smart-demand-forecasting/data/processed/daily_demand_features.csv"
    artifacts = "/working_dir/smart-demand-forecasting/data/processed"
    trainer = DemandModelTrainer(feat_path, artifacts)
    trainer.run()
