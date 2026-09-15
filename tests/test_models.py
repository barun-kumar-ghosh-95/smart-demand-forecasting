import unittest
import numpy as np
from src.models.evaluator import ModelEvaluator
from src.models.ml_models import RidgeRegressionForecaster

class TestModelEvaluation(unittest.TestCase):
    def test_metrics_perfect_prediction(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        self.assertEqual(metrics["MAE"], 0.0)
        self.assertEqual(metrics["RMSE"], 0.0)
        self.assertEqual(metrics["sMAPE"], 0.0)
        self.assertEqual(metrics["Bias"], 0.0)

    def test_metrics_bias_direction(self):
        y_true = np.array([100.0, 100.0])
        # Overforecasting
        metrics_over = ModelEvaluator.calculate_metrics(y_true, np.array([110.0, 110.0]))
        self.assertGreater(metrics_over["Bias"], 0.0)

        # Underforecasting
        metrics_under = ModelEvaluator.calculate_metrics(y_true, np.array([90.0, 90.0]))
        self.assertLess(metrics_under["Bias"], 0.0)

    def test_ridge_regression_fit_predict(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        true_w = np.array([2.0, -1.5, 0.5, 3.0, -0.8])
        y = np.maximum(0, X @ true_w + 10.0 + np.random.normal(0, 0.2, 100))

        model = RidgeRegressionForecaster(alpha=0.1)
        model.fit(X, y)
        preds = model.predict(X)

        self.assertEqual(len(preds), 100)
        self.assertTrue(all(preds >= 0.0))
        metrics = ModelEvaluator.calculate_metrics(y, preds)
        self.assertLess(metrics["MAE"], 2.0)

if __name__ == "__main__":
    unittest.main()
