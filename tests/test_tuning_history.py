"""Tests for the compact Tuning Lab history controls."""

import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from werkzeug.datastructures import MultiDict

from app import _build_history_page, _parse_history_query, app
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

        page = self.build(rows)

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

        current_page = self.build([legacy])
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
        self.assertNotIn("stale", html)
        self.assertNotIn("official-config-picker", html)
        self.assertIn("Chạy thử CV", html)
        self.assertIn("Dùng cấu hình này", html)
        self.assertNotIn("Xem lịch sử legacy", html)
        self.assertNotIn("Quay lại lịch sử cũ", html)


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
