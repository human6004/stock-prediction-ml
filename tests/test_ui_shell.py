"""Contract tests for the shared app shell (base.html).

Contract locked here:
- One base.html shell shared by all 8 templates.
- A skip-link targeting #main-content.
- Exactly one <main id="main-content"> and one <h1> per page.
- A primary sidebar nav (#primary-nav) whose current item carries
  aria-current="page"; exactly one item is current at a time.
- A server-rendered #theme-toggle mount point (no more .nav-links dependency).
- Each route passes the correct active_page (fetch status = tuning branch).
- Tuning AJAX swaps only #main-content, never the whole shell (regression, vá 2).
- <head> keeps the asset_ver cache-bust + both fonts (vá 3).
"""

from __future__ import annotations

import contextlib
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import app as web_app


ROOT_DIR = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT_DIR / "templates"

# active_page slug == data-nav attribute value (plan §5 stable contract).
NAV_ITEMS = ("prediction", "screener", "compare", "chat", "evaluation", "tuning")

# Every page template must inherit base.html. fetch_status is a child of tuning.
PAGE_TEMPLATES = {
    "index.html": "prediction",
    "screener.html": "screener",
    "compare.html": "compare",
    "chat.html": "chat",
    "evaluation.html": "evaluation",
    "tuning.html": "tuning",
    "fetch_status.html": "tuning",
}


def render_base(active_page, **extra):
    """Render base.html standalone (empty content block) with shell context."""
    context = {
        "active_page": active_page,
        "dataset_max_date": "2026-07-10",
        "symbol_count": 3,
        "symbols": ("FPT", "VNM", "TCD"),
        "selected_model": "Logistic Regression",
        "threshold_percent": 3,
        "horizon": 5,
    }
    context.update(extra)
    with web_app.app.test_request_context():
        return web_app.render_template("base.html", **context)


class ShellContractTests(unittest.TestCase):
    """The home route must render through the shared shell."""

    def setUp(self):
        web_app.app.config.update(TESTING=True)
        self.client = web_app.app.test_client()
        self.meta = patch.object(
            web_app,
            "get_dataset_meta",
            return_value={
                "dataset_max_date": "2026-07-10",
                "symbol_count": 3,
                "symbols": ("FPT", "VNM", "TCD"),
            },
        )
        self.sel = patch.object(
            web_app, "load_selected_model_from_report", return_value="Logistic Regression"
        )
        self.meta.start()
        self.sel.start()
        self.addCleanup(self.meta.stop)
        self.addCleanup(self.sel.stop)
        self.html = self.client.get("/").get_data(as_text=True)

    def test_home_renders_exactly_one_main_landmark(self):
        self.assertEqual(len(re.findall(r"<main\b", self.html)), 1)

    def test_main_landmark_carries_the_content_id(self):
        self.assertRegex(self.html, r'<main[^>]*\bid="main-content"')

    def test_page_has_exactly_one_h1(self):
        self.assertEqual(len(re.findall(r"<h1\b", self.html)), 1)

    def test_skip_link_targets_main_content(self):
        self.assertIn('href="#main-content"', self.html)

    def test_app_shell_and_primary_nav_present(self):
        self.assertIn('id="app-shell"', self.html)
        self.assertIn('id="primary-nav"', self.html)

    def test_theme_toggle_is_server_rendered(self):
        self.assertIn('id="theme-toggle"', self.html)

    def test_all_primary_nav_items_present(self):
        for item in NAV_ITEMS:
            self.assertRegex(self.html, rf'data-nav="{item}"')

    def test_home_marks_prediction_as_current(self):
        self.assertEqual(self.html.count('aria-current="page"'), 1)
        self.assertTrue(
            re.search(r'data-nav="prediction"[^>]*aria-current="page"', self.html)
            or re.search(r'aria-current="page"[^>]*data-nav="prediction"', self.html)
        )


class ActiveNavContractTests(unittest.TestCase):
    """base.html turns active_page into a single aria-current link."""

    def test_current_page_link_has_aria_current(self):
        for active in NAV_ITEMS:
            html = render_base(active)
            marked = re.search(
                rf'data-nav="{active}"[^>]*aria-current="page"', html
            ) or re.search(rf'aria-current="page"[^>]*data-nav="{active}"', html)
            self.assertTrue(marked, f"{active} nav link must carry aria-current=page")

    def test_exactly_one_link_is_current(self):
        for active in NAV_ITEMS:
            html = render_base(active)
            self.assertEqual(
                html.count('aria-current="page"'),
                1,
                f"exactly one current nav item expected for active_page={active}",
            )


class TemplateInheritanceTests(unittest.TestCase):
    """All 8 templates inherit the shared shell."""

    def test_every_page_extends_base(self):
        for name in PAGE_TEMPLATES:
            src = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
            self.assertRegex(
                src,
                r'{%\s*extends\s+"base\.html"\s*%}',
                f"{name} must extend base.html",
            )


class RouteActivePageTests(unittest.TestCase):
    """Each route hands the template the correct active_page slug."""

    def setUp(self):
        web_app.app.config.update(TESTING=True)
        self.client = web_app.app.test_client()
        self.meta = patch.object(
            web_app,
            "get_dataset_meta",
            return_value={
                "dataset_max_date": "2026-07-10",
                "symbol_count": 3,
                "symbols": ("FPT", "VNM", "TCD"),
            },
        )
        self.sel = patch.object(
            web_app, "load_selected_model_from_report", return_value="LR"
        )
        self.meta.start()
        self.sel.start()
        self.addCleanup(self.meta.stop)
        self.addCleanup(self.sel.stop)

        self.captured = {}

        def fake_render(template, **ctx):
            self.captured["template"] = template
            self.captured["active_page"] = ctx.get("active_page")
            return "ok"

        self.render = patch.object(
            web_app, "render_template", side_effect=fake_render
        )
        self.render.start()
        self.addCleanup(self.render.stop)

    def _assert_active(self, path, template, active, **extra_patches):
        with contextlib.ExitStack() as stack:
            for target, kwargs in extra_patches.items():
                stack.enter_context(patch.object(web_app, target, **kwargs))
            self.client.get(path)
        self.assertEqual(self.captured.get("template"), template)
        self.assertEqual(self.captured.get("active_page"), active)

    def test_index_active_prediction(self):
        self._assert_active("/", "index.html", "prediction")

    def test_compare_active_compare(self):
        self._assert_active("/compare", "compare.html", "compare")

    def test_chat_active_chat(self):
        self._assert_active("/chat", "chat.html", "chat")

    def test_screener_active_screener(self):
        self._assert_active(
            "/screener",
            "screener.html",
            "screener",
            predict_all_symbols={"return_value": []},
        )

    def test_evaluation_active_evaluation(self):
        self._assert_active(
            "/evaluation",
            "evaluation.html",
            "evaluation",
            load_evaluation_sections={"return_value": {"baseline_warning": None}},
        )

    def test_tuning_active_tuning(self):
        self._assert_active(
            "/tuning",
            "tuning.html",
            "tuning",
            _tuning_context={"return_value": {}},
        )

    def test_fetch_status_active_tuning_branch(self):
        self._assert_active(
            "/tuning/fetch-status",
            "fetch_status.html",
            "tuning",
            is_fetch_running={"return_value": False},
            read_fetch_log_tail={"return_value": ""},
            _load_fetch_report={"return_value": None},
        )


class TuningSwapRegressionTests(unittest.TestCase):
    """AJAX swap must replace #main-content only, keeping the sidebar shell."""

    def setUp(self):
        self.src = (TEMPLATES_DIR / "tuning.html").read_text(encoding="utf-8")

    def test_swap_targets_main_content(self):
        self.assertIn("#main-content", self.src)

    def test_swap_never_replaces_the_whole_shell(self):
        self.assertNotIn('querySelector("main.shell")', self.src)


class HeadContractTests(unittest.TestCase):
    """base.html <head> keeps cache-busting and both project fonts (vá 3)."""

    def setUp(self):
        self.html = render_base("prediction")

    def test_css_is_cache_busted(self):
        self.assertRegex(self.html, r"app\.css\?v=")

    def test_theme_script_present(self):
        self.assertIn("theme.js", self.html)

    def test_both_fonts_loaded(self):
        self.assertIn("Nunito", self.html)
        self.assertIn("IBM+Plex+Mono", self.html)


if __name__ == "__main__":
    unittest.main()
