"""
Machine Learning Forecasting Models.
Provides:
1. Ridge Regression with L2 Regularization and standardization
2. Boosted Tree / Ensemble Regressor
3. LightGBM / XGBoost wrapper with automatic graceful fallback
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any

class RidgeRegressionForecaster:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        # Feature standardization
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds == 0] = 1.0

        X_norm = (X - self.means) / self.stds
        n, p = X_norm.shape

        # Closed form: w = (X^T X + alpha * I)^(-1) X^T (y - y_mean)
        y_mean = np.mean(y)
        self.bias = float(y_mean)

        XtX = X_norm.T @ X_norm
        reg_matrix = XtX + self.alpha * np.eye(p)
        try:
            self.weights = np.linalg.solve(reg_matrix, X_norm.T @ (y - y_mean))
        except np.linalg.LinAlgError:
            self.weights = np.linalg.pinv(reg_matrix) @ (X_norm.T @ (y - y_mean))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X_norm = (X - self.means) / self.stds
        preds = self.bias + X_norm @ self.weights
        return np.clip(preds, 0, None)

    def get_feature_importance(self) -> pd.DataFrame:
        if self.weights is None:
            return pd.DataFrame()
        imp = np.abs(self.weights)
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": imp
        }).sort_values("importance", ascending=False).reset_index(drop=True)


class DecisionStump:
    def __init__(self):
        self.feature_idx: int = 0
        self.threshold: float = 0.0
        self.left_val: float = 0.0
        self.right_val: float = 0.0

    def fit(self, X: np.ndarray, residuals: np.ndarray, sample_features: int = 15):
        n, p = X.shape
        best_loss = float("inf")
        feat_indices = np.random.choice(p, min(p, sample_features), replace=False)
        
        for feat in feat_indices:
            vals = X[:, feat]
            thresholds = np.percentile(vals, [20, 40, 60, 80])
            for th in thresholds:
                left_mask = vals <= th
                right_mask = ~left_mask
                if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
                    continue
                left_mean = np.mean(residuals[left_mask])
                right_mean = np.mean(residuals[right_mask])
                loss = np.sum((residuals[left_mask] - left_mean) ** 2) + np.sum((residuals[right_mask] - right_mean) ** 2)
                if loss < best_loss:
                    best_loss = loss
                    self.feature_idx = feat
                    self.threshold = th
                    self.left_val = left_mean
                    self.right_val = right_mean

    def predict(self, X: np.ndarray) -> np.ndarray:
        left_mask = X[:, self.feature_idx] <= self.threshold
        preds = np.empty(len(X), dtype=float)
        preds[left_mask] = self.left_val
        preds[~left_mask] = self.right_val
        return preds


class BoostedRegressionForecaster:
    """
    Gradient Boosted Regression Tree forecaster.
    Trains an initial regularized base model followed by shrinkage trees on pseudo-residuals.
    """
    def __init__(self, n_estimators: int = 35, learning_rate: float = 0.08):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_model = RidgeRegressionForecaster(alpha=2.0)
        self.trees: List[DecisionStump] = []
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        self.base_model.fit(X, y, self.feature_names)
        current_preds = self.base_model.predict(X)

        self.trees = []
        for _ in range(self.n_estimators):
            residuals = y - current_preds
            tree = DecisionStump()
            tree.fit(X, residuals, sample_features=18)
            update = tree.predict(X)
            current_preds += self.learning_rate * update
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        preds = self.base_model.predict(X)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return np.clip(preds, 0, None)


class RetailForecastingModel:
    """
    Top-level wrapper that automatically uses LightGBM/XGBoost if installed,
    or BoostedRegressionForecaster as robust native fallback.
    """
    def __init__(self, model_type: str = "boosted"):
        self.model_type = model_type
        self.model = None
        self.feature_names: List[str] = []
        self.backend = "native"

        if model_type == "ridge":
            self.model = RidgeRegressionForecaster(alpha=1.5)
        else:
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
                )
                self.backend = "lightgbm"
            except ImportError:
                try:
                    import xgboost as xgb
                    self.model = xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=5,
                        random_state=42
                    )
                    self.backend = "xgboost"
                except ImportError:
                    self.model = BoostedRegressionForecaster(n_estimators=40, learning_rate=0.07)
                    self.backend = "native_boosted"

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or []
        if self.backend in ["lightgbm", "xgboost"]:
            self.model.fit(X, y)
        else:
            self.model.fit(X, y, feature_names=self.feature_names)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.backend in ["lightgbm", "xgboost"]:
            return np.clip(self.model.predict(X), 0, None)
        return self.model.predict(X)
