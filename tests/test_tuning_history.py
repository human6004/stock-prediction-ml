"""Tests for the compact Tuning Lab history controls."""

import json
import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from werkzeug.datastructures import MultiDict

from app import _build_history_page, _parse_history_query, _tuning_context, app
from config.settings import CV_N_SPLITS, EXPERIMENT_POLICY_ID, TUNABLE_PARAM_SCHEMA


CURRENT_FP = "current123456"
OLD_FP = "old000000000"


def history_row(
    run_id,
    *,
    model_key="random_forest",
    fingerprint=CURRENT_FP,
    status="ok",
    timestamp="2026-07-18T10:00:00",
    is_best=False,
):
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_key": model_key,
        "model_name": model_key,
        "params": {
            "n_estimators": 100,
            "max_depth": 4,
            "min_samples_leaf": 10,
            "max_features": 0.35,
        },
        "params_json": '{"max_depth": 4}',
        "cv_f1_up_mean": 0.5,
        "cv_f1_up_std": 0.01,
        "decision_threshold": 0.6,
        "train_seconds": 1.0,
        "dataset_fingerprint": fingerprint,
        "policy_id": EXPERIMENT_POLICY_ID,
        "status": status,
        "note": "",
        "is_best": is_best,
    }


class HistoryQueryTests(unittest.TestCase):
    def test_dataset_defaults_to_all_history(self):
        filters = _parse_history_query(
            MultiDict(), "random_forest", TUNABLE_PARAM_SCHEMA
        )

        self.assertEqual(filters["dataset"], "all")

    def test_best_selected_and_page_are_preserved(self):
        filters = _parse_history_query(
            MultiDict({"best": "1", "selected": "1", "page": "3"}),
            "random_forest",
            TUNABLE_PARAM_SCHEMA,
        )

        self.assertTrue(filters["best"])
        self.assertTrue(filters["selected"])
        self.assertEqual(filters["page"], 3)
        self.assertEqual(filters["query_args"]["best"], "1")
        self.assertEqual(filters["query_args"]["selected"], "1")
        self.assertEqual(filters["query_args"]["page"], 3)

    def test_invalid_page_falls_back_to_first(self):
        filters = _parse_history_query(
            MultiDict({"page": "invalid"}),
            "random_forest",
            TUNABLE_PARAM_SCHEMA,
        )

        self.assertEqual(filters["page"], 1)
        self.assertFalse(filters["best"])
        self.assertFalse(filters["selected"])

    def test_legacy_query_is_ignored(self):
        filters = _parse_history_query(
            MultiDict({"legacy": "1"}),
            "random_forest",
            TUNABLE_PARAM_SCHEMA,
        )

        self.assertNotIn("legacy", filters)
        self.assertNotIn("legacy", filters["query_args"])


class HistoryPageTests(unittest.TestCase):
    def build(self, rows, values=None, selected_run_id="selected"):
        filters = _parse_history_query(
            MultiDict(values or {}),
            "random_forest",
            TUNABLE_PARAM_SCHEMA,
        )
        return _build_history_page(
            rows,
            "random_forest",
            CURRENT_FP,
            selected_run_id,
            filters,
        )

    def test_scope_is_current_ok_active_model_and_newest_first(self):
        rows = [
            history_row("older", timestamp="2026-07-17T10:00:00"),
            history_row("newer", timestamp="2026-07-18T10:00:00"),
            history_row("old", fingerprint=OLD_FP),
            history_row("error", status="error"),
            history_row("other", model_key="gradient_boosting"),
        ]

        page = self.build(rows, {"dataset": "current"})

        self.assertEqual([row["run_id"] for row in page["rows"]], ["newer", "older"])

    def test_best_and_selected_can_be_combined(self):
        rows = [
            history_row("selected", is_best=True),
            history_row("best-only", is_best=True),
            history_row("ordinary"),
        ]

        page = self.build(rows, {"best": "1", "selected": "1"})

        self.assertEqual([row["run_id"] for row in page["rows"]], ["selected"])

    def test_page_size_is_50_and_page_above_range_clamps(self):
        rows = [
            history_row(
                f"run-{index:03d}",
                timestamp=f"2026-07-18T10:{index % 60:02d}:00",
            )
            for index in range(120)
        ]

        page = self.build(rows, {"page": "999"})

        self.assertEqual(page["total"], 120)
        self.assertEqual(page["total_pages"], 3)
        self.assertEqual(page["page"], 3)
        self.assertEqual(len(page["rows"]), 20)
        self.assertEqual((page["start"], page["end"]), (101, 120))

    def test_empty_result_has_safe_metadata(self):
        page = self.build([])

        self.assertEqual(page["rows"], [])
        self.assertEqual(page["page"], 1)
        self.assertEqual((page["start"], page["end"]), (0, 0))

    def test_legacy_rows_never_appear_even_if_query_requests_them(self):
        legacy = history_row("legacy", fingerprint=OLD_FP)
        legacy["policy_id"] = "legacy"

        current_page = self.build([legacy], {"dataset": "current"})
        legacy_page = self.build([legacy], {"legacy": "1"})

        self.assertEqual(current_page["rows"], [])
        self.assertEqual(legacy_page["rows"], [])


class HistoryTemplateTests(unittest.TestCase):
    def test_history_form_only_exposes_best_and_selected(self):
        current = history_row("current", is_best=True)
        stale = history_row("stale", fingerprint=OLD_FP)

        app.config.update(TESTING=True)
        with app.test_request_context("/tuning?history_model=random_forest"):
            with (
                patch(
                    "app.compute_dataset_fingerprint",
                    return_value={"hash": CURRENT_FP, "parts": {}},
                ),
                patch(
                    "app.read_manual_config",
                    return_value={
                        "dataset_fingerprint": CURRENT_FP,
                        "selected_models": {},
                        "cv_settings": {"n_splits": CV_N_SPLITS, "gap": 5},
                    },
                ),
                patch("app.read_history", return_value=[current, stale]),
                patch("app.mark_best", side_effect=lambda rows: rows),
                patch("app.is_config_complete", return_value=False),
                patch("app.has_evaluated_snapshot", return_value=False),
                patch("app.is_pipeline_running", return_value=False),
                patch("app.is_fetch_running", return_value=False),
            ):
                response = app.full_dispatch_request()

        html = response.get_data(as_text=True)
        filter_form = html.split('<form class="history-filter-form"', 1)[1].split(
            "</form>", 1
        )[0]
        self.assertEqual(response.status_code, 200)
        self.assertIn('name="best"', filter_form)
        self.assertIn('name="selected"', filter_form)
        self.assertNotIn('name="dataset"', filter_form)
        self.assertNotIn('name="status"', filter_form)
        self.assertNotIn('name="f1_min"', filter_form)
        self.assertNotIn('name="sort"', filter_form)
        self.assertNotIn('name="p_', filter_form)
        self.assertIn("current", html)
        self.assertIn("stale", html)
        self.assertIn("(2 lần)", html)
        self.assertNotIn("official-config-picker", html)
        self.assertIn("Chạy thử CV", html)
        self.assertIn("Dùng cấu hình này", html)
        self.assertNotIn("Xem lịch sử legacy", html)
        self.assertNotIn("Quay lại lịch sử cũ", html)


class HistoryTrendTests(unittest.TestCase):
    """`history_trend` — chuỗi F1 theo thời gian cho biểu đồ hội tụ."""

    def trend(self, rows, *, query="?history_model=random_forest", cfg_extra=None):
        """Gọi _tuning_context trong request context rồi lấy ra history_trend.

        Patch đúng bộ như HistoryTemplateTests: mark_best để nguyên rows (cờ
        is_best do chính test đặt qua history_row), mọi cờ lock trả False.
        """
        cfg = {
            "dataset_fingerprint": CURRENT_FP,
            "selected_models": {},
            "cv_settings": {"n_splits": CV_N_SPLITS, "gap": 5},
        }
        cfg.update(cfg_extra or {})
        app.config.update(TESTING=True)
        with app.test_request_context(f"/tuning{query}"):
            with (
                patch(
                    "app.compute_dataset_fingerprint",
                    return_value={"hash": CURRENT_FP, "parts": {}},
                ),
                patch("app.read_manual_config", return_value=cfg),
                patch("app.read_history", return_value=rows),
                patch("app.mark_best", side_effect=lambda rows: rows),
                patch("app.is_config_complete", return_value=False),
                patch("app.has_evaluated_snapshot", return_value=False),
                patch("app.is_pipeline_running", return_value=False),
                patch("app.is_fetch_running", return_value=False),
            ):
                return _tuning_context()["history_trend"]

    @staticmethod
    def scored(run_id, f1, timestamp, **kwargs):
        row = history_row(run_id, timestamp=timestamp, **kwargs)
        row["cv_f1_up_mean"] = f1
        return row

    def test_order_follows_timestamp_not_score(self):
        rows = [
            self.scored("second", 0.70, "2026-07-18T11:00:00"),
            self.scored("third", 0.40, "2026-07-18T12:00:00"),
            self.scored("first", 0.55, "2026-07-18T10:00:00"),
        ]

        trend = self.trend(rows)

        self.assertEqual([item["n"] for item in trend], [1, 2, 3])
        self.assertEqual([item["f1"] for item in trend], [0.55, 0.70, 0.40])
        self.assertEqual(
            [item["timestamp"] for item in trend],
            ["2026-07-18T10:00:00", "2026-07-18T11:00:00", "2026-07-18T12:00:00"],
        )

    def test_sort_query_does_not_reorder_the_trend(self):
        rows = [
            self.scored("low", 0.30, "2026-07-18T10:00:00"),
            self.scored("high", 0.90, "2026-07-18T11:00:00"),
        ]

        trend = self.trend(
            rows, query="?history_model=random_forest&sort=f1_desc"
        )

        self.assertEqual([item["f1"] for item in trend], [0.30, 0.90])

    def test_other_fingerprint_is_excluded(self):
        rows = [
            self.scored("current", 0.50, "2026-07-18T10:00:00"),
            self.scored("stale", 0.99, "2026-07-18T11:00:00", fingerprint=OLD_FP),
        ]

        trend = self.trend(rows)

        self.assertEqual([item["f1"] for item in trend], [0.50])

    def test_other_model_is_excluded(self):
        rows = [
            self.scored("rf", 0.50, "2026-07-18T10:00:00"),
            self.scored(
                "gb", 0.99, "2026-07-18T11:00:00", model_key="gradient_boosting"
            ),
        ]

        trend = self.trend(rows)

        self.assertEqual([item["f1"] for item in trend], [0.50])

    def test_non_finite_and_missing_f1_are_excluded(self):
        rows = [
            self.scored("ok", 0.50, "2026-07-18T10:00:00"),
            self.scored("nan", float("nan"), "2026-07-18T11:00:00"),
            self.scored("inf", float("inf"), "2026-07-18T12:00:00"),
            self.scored("empty", None, "2026-07-18T13:00:00"),
            self.scored("text", "n/a", "2026-07-18T14:00:00"),
        ]

        trend = self.trend(rows)

        self.assertEqual([item["f1"] for item in trend], [0.50])
        # n phải liên tục 1..N sau khi loại, không giữ khoảng trống của run bị bỏ.
        self.assertEqual([item["n"] for item in trend], [1])

    def test_legacy_policy_rows_are_excluded(self):
        legacy = self.scored("legacy", 0.99, "2026-07-18T11:00:00")
        legacy["policy_id"] = "legacy"
        rows = [self.scored("ok", 0.50, "2026-07-18T10:00:00"), legacy]

        trend = self.trend(rows)

        self.assertEqual([item["f1"] for item in trend], [0.50])

    def test_best_and_selected_flags_are_marked(self):
        rows = [
            self.scored("plain", 0.40, "2026-07-18T10:00:00"),
            self.scored("chosen", 0.60, "2026-07-18T11:00:00"),
            self.scored("peak", 0.80, "2026-07-18T12:00:00", is_best=True),
        ]

        trend = self.trend(
            rows,
            cfg_extra={
                "selected_models": {
                    "random_forest": {
                        "run_id": "chosen",
                        "selection_method": "manual",
                    }
                }
            },
        )

        self.assertEqual(
            [(item["is_best"], item["is_selected"]) for item in trend],
            [(False, False), (False, True), (True, False)],
        )

    def test_empty_history_gives_empty_list_not_none(self):
        trend = self.trend([])

        self.assertEqual(trend, [])

    def test_result_is_json_serializable(self):
        rows = [
            self.scored("ok", 0.50, "2026-07-18T10:00:00", is_best=True),
            self.scored("nan", float("nan"), "2026-07-18T11:00:00"),
        ]

        trend = self.trend(rows)

        # allow_nan=False mô phỏng đúng chỗ dễ vỡ: Jinja |tojson sinh ra NaN thì
        # JSON.parse phía client ném lỗi và mất biểu đồ.
        self.assertEqual(
            json.loads(json.dumps(trend, allow_nan=False)),
            [
                {
                    "n": 1,
                    "f1": 0.50,
                    "timestamp": "2026-07-18T10:00:00",
                    "is_best": True,
                    "is_selected": False,
                }
            ],
        )


class HistoryUseConfigTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_redirect_keeps_checkbox_state_and_rejects_open_redirect(self):
        run = history_row("current")
        with (
            patch("app.find_run", return_value=run),
            patch(
                "app.compute_dataset_fingerprint",
                return_value={"hash": CURRENT_FP, "parts": {}},
            ),
            patch("app.save_selected_model"),
        ):
            response = self.client.post(
                "/tuning/use-config",
                data={
                    "run_id": "current",
                    "best": "1",
                    "legacy": "1",
                    "page": "2",
                    "next": "https://example.com/unsafe",
                },
            )

        query = parse_qs(urlparse(response.headers["Location"]).query)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(query["history_model"], ["random_forest"])
        self.assertEqual(query["best"], ["1"])
        self.assertEqual(query["page"], ["2"])
        self.assertNotIn("legacy", query)
        self.assertNotIn("next", query)

    def test_stale_run_is_rejected(self):
        run = history_row("stale", fingerprint=OLD_FP)
        with (
            patch("app.find_run", return_value=run),
            patch(
                "app.compute_dataset_fingerprint",
                return_value={"hash": CURRENT_FP, "parts": {}},
            ),
            patch("app._tuning_context", return_value={}),
            patch("app.render_template", return_value="error"),
            patch("app.save_selected_model") as save,
        ):
            response = self.client.post(
                "/tuning/use-config", data={"run_id": "stale"}
            )

        self.assertEqual(response.status_code, 400)
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
