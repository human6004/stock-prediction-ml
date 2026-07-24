"""Contracts for targets, rolling boundaries and date-based CV."""

import unittest

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier

from config.settings import FEATURE_COLUMNS
from services.feature_engineering import (
    create_labels,
    protocol_time_split,
)
from services.time_splitting import iter_purged_date_splits
from services.model_tuning import run_cv_metrics
from services.protocol_dates import resolve_protocol_dates


class FeatureProbabilityEstimator(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        probability = ((X[FEATURE_COLUMNS[0]].to_numpy() % 10) + 0.5) / 10
        return np.column_stack([1 - probability, probability])


class CommonMarketTargetTests(unittest.TestCase):
    def test_target_uses_exact_fifth_common_market_session(self):
        dates = pd.date_range("2025-01-01", periods=8, freq="D").strftime("%Y-%m-%d")
        clean = pd.DataFrame(
            [
                {"symbol": symbol, "trading_date": day, "close": close}
                for symbol, closes in (("AAA", range(100, 108)), ("BBB", range(200, 208)))
                for day, close in zip(dates, closes)
            ]
        )
        features = clean.iloc[[0, 1, 2, 8, 9, 10]].copy()
        features["feature"] = 1.0

        labeled, report = create_labels(clean, features, horizon=5, up_threshold=0.01)

        aaa = labeled[labeled["symbol"] == "AAA"].set_index("trading_date")
        self.assertEqual(aaa.loc["2025-01-01", "label_end_date"], "2025-01-06")
        self.assertEqual(aaa.loc["2025-01-01", "future_close_5d"], 105)
        self.assertEqual(aaa.loc["2025-01-01", "target"], 1)
        self.assertEqual(report["horizon"], 5)

    def test_missing_price_on_exact_future_session_is_dropped_not_skipped(self):
        dates = pd.date_range("2025-01-01", periods=8, freq="D").strftime("%Y-%m-%d")
        clean = pd.DataFrame(
            [
                {"symbol": "AAA", "trading_date": day, "close": 100 + index}
                for index, day in enumerate(dates)
            ]
            + [
                {"symbol": "BBB", "trading_date": day, "close": 200 + index}
                for index, day in enumerate(dates)
                if day != "2025-01-06"
            ]
        )
        features = clean.copy()
        features["feature"] = 1.0

        labeled, report = create_labels(clean, features, horizon=5, up_threshold=0.01)

        kept = set(zip(labeled["symbol"], labeled["trading_date"]))
        self.assertIn(("AAA", "2025-01-01"), kept)
        self.assertNotIn(("BBB", "2025-01-01"), kept)
        self.assertGreaterEqual(report["rows_missing_symbol_future_close"], 1)


class HoldoutBoundaryTests(unittest.TestCase):
    def test_default_boundaries_roll_with_latest_labeled_session(self):
        first = pd.DataFrame(
            {"trading_date": ["2024-01-02", "2026-07-20"]}
        )
        second = pd.concat(
            [first, pd.DataFrame({"trading_date": ["2026-07-21"]})],
            ignore_index=True,
        )

        before = resolve_protocol_dates(first)
        after = resolve_protocol_dates(second)

        self.assertEqual(before["source"], "rolling_max_session")
        self.assertEqual(before["test_end_date"], "2026-07-20")
        self.assertEqual(after["test_end_date"], "2026-07-21")
        self.assertNotEqual(before, after)

    def test_protocol_split_respects_explicit_train_validation_and_test_dates(self):
        dataset = pd.DataFrame(
            [
                {"symbol": "AAA", "trading_date": "2025-06-20", "label_end_date": "2025-06-27", "target": 0},
                {"symbol": "AAA", "trading_date": "2025-06-27", "label_end_date": "2025-07-04", "target": 1},
                {"symbol": "AAA", "trading_date": "2025-07-01", "label_end_date": "2025-07-08", "target": 1},
                {"symbol": "AAA", "trading_date": "2026-03-24", "label_end_date": "2026-03-31", "target": 0},
                {"symbol": "AAA", "trading_date": "2026-03-30", "label_end_date": "2026-04-07", "target": 1},
                {"symbol": "AAA", "trading_date": "2026-04-01", "label_end_date": "2026-04-08", "target": 0},
                {"symbol": "AAA", "trading_date": "2026-07-06", "label_end_date": "2026-07-13", "target": 1},
            ]
        )

        train, validation, test, report = protocol_time_split(
            dataset,
            train_end_date="2025-06-30",
            validation_end_date="2026-03-31",
            test_end_date="2026-07-03",
        )

        self.assertEqual(train["trading_date"].tolist(), ["2025-06-20"])
        self.assertEqual(
            validation["trading_date"].tolist(), ["2025-07-01", "2026-03-24"]
        )
        self.assertEqual(test["trading_date"].tolist(), ["2026-04-01"])
        self.assertEqual(report["purged_train_validation_rows"], 1)
        self.assertEqual(report["purged_validation_test_rows"], 1)
        self.assertEqual(report["post_test_inference_rows"], 1)
        self.assertLess(train["label_end_date"].max(), validation["trading_date"].min())
        self.assertLess(validation["label_end_date"].max(), test["trading_date"].min())


class PurgedDateFoldTests(unittest.TestCase):
    def test_folds_use_dates_gap_sessions_and_purge_label_overlap(self):
        dates = pd.date_range("2025-01-01", periods=36, freq="D")
        rows = []
        for date_index, day in enumerate(dates):
            label_end = dates[min(date_index + 5, len(dates) - 1)]
            for symbol in ("AAA", "BBB"):
                rows.append(
                    {
                        "symbol": symbol,
                        "trading_date": day.strftime("%Y-%m-%d"),
                        "label_end_date": label_end.strftime("%Y-%m-%d"),
                    }
                )
        frame = pd.DataFrame(rows)

        folds = list(iter_purged_date_splits(frame, n_splits=3, gap_sessions=5))

        self.assertEqual(len(folds), 3)
        market_dates = sorted(frame["trading_date"].unique())
        for train_idx, validation_idx in folds:
            train = frame.iloc[train_idx]
            validation = frame.iloc[validation_idx]
            validation_start = validation["trading_date"].min()
            train_end_position = market_dates.index(train["trading_date"].max())
            validation_start_position = market_dates.index(validation_start)
            self.assertEqual(validation_start_position - train_end_position - 1, 5)
            self.assertLess(train["label_end_date"].max(), validation_start)
            self.assertEqual(validation.groupby("trading_date")["symbol"].nunique().min(), 2)

    def test_official_cv_uses_panel_dates_and_records_fold_ranges(self):
        dates = pd.date_range("2025-01-01", periods=36, freq="D")
        rows = []
        for date_index, day in enumerate(dates):
            for symbol_index, symbol in enumerate(("AAA", "BBB")):
                row = {column: float(date_index + symbol_index) for column in FEATURE_COLUMNS}
                row.update(
                    symbol=symbol,
                    trading_date=day.strftime("%Y-%m-%d"),
                    label_end_date=dates[min(date_index + 5, 35)].strftime("%Y-%m-%d"),
                    target=int((date_index + symbol_index) % 10 >= 5),
                )
                rows.append(row)
        frame = pd.DataFrame(rows)

        metrics = run_cv_metrics(FeatureProbabilityEstimator(), frame)

        self.assertTrue(metrics["threshold_constraint_passed"])
        self.assertEqual(len(metrics["fold_date_ranges"]), 4)
        for fold in metrics["fold_date_ranges"]:
            train_end = dates.get_loc(pd.Timestamp(fold["train_end"]))
            validation_start = dates.get_loc(pd.Timestamp(fold["validation_start"]))
            self.assertEqual(validation_start - train_end - 1, 5)
            self.assertLess(fold["train_label_end_max"], fold["validation_start"])


if __name__ == "__main__":
    unittest.main()
