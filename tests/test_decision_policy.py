"""Regression test for the user-facing UP/NOT_UP decision policy."""

import unittest

import pandas as pd
from sklearn.dummy import DummyClassifier

from config.settings import FEATURE_COLUMNS
from services.model_tuning import run_cv_metrics


class DecisionPolicyTests(unittest.TestCase):
    def test_cv_chooses_constrained_oof_probability_cutoff(self):
        rows = 60
        dates = pd.date_range("2025-01-01", periods=rows, freq="D")
        train = pd.DataFrame(
            {
                **{column: range(rows) for column in FEATURE_COLUMNS},
                "symbol": ["FPT"] * rows,
                "trading_date": dates.strftime("%Y-%m-%d"),
                "label_end_date": dates.strftime("%Y-%m-%d"),
                "target": [0, 0, 1] * (rows // 3),
            }
        )

        metrics = run_cv_metrics(DummyClassifier(strategy="prior"), train)

        self.assertTrue(0 < metrics["decision_threshold"] < 1)
        self.assertLessEqual(metrics["oof_predicted_up_ratio"], 0.5)
        self.assertGreaterEqual(metrics["oof_precision_up"], metrics["oof_up_rate"])


if __name__ == "__main__":
    unittest.main()
