"""Regression cho các kịch bản audit hậu-fix (chatbot SCI).

Mỗi test ở đây khoá lại MỘT lỗ thật đã reproduce được trên code live, không phải
giả định. Nhóm theo tầng: gate an toàn → HTTP contract → date semantics →
routing. Chỉ dùng hàm public/module-level của service, không patch nội bộ sâu
hơn mức cần thiết, để test còn giá trị sau khi refactor.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

import app as web_app
from services import chatbot_service, chatbot_tools


def _response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _client(content: str) -> SimpleNamespace:
    create = Mock(return_value=_response(content))
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _bundle(*, context=None, numbers=None, facts=None) -> dict:
    return {
        "context": context
        or {"project_snapshot": {"data_as_of": "2026-07-31", "release_status": "legacy"}},
        "sources": [{"kind": "project_snapshot", "as_of": "2026-07-31"}],
        "warnings": [],
        "release_status": "legacy",
        "grounded_numbers": numbers if numbers is not None else [],
        "stock_facts": facts or {},
        "model_facts": {},
        "data_as_of": "2026-07-31",
        "report_as_of": None,
        "model_trained_through": "2026-04-10",
        "conversation_state": {
            "active_symbols": [],
            "topic": None,
            "ranking_order": None,
            "last_result_symbols": [],
        },
    }


# Câu khuyên mua/bán viết KHÔNG DẤU. Provider chỉ được yêu cầu trả lời "tiếng
# Việt" (SYSTEM_PROMPT không đòi có dấu), nên đây là drift tự nhiên chứ không
# phải payload tấn công.
UNACCENTED_ADVICE = (
    "Ban nen mua FPT ngay.",
    "Nen ban VNM.",
    "Khuyen nghi mua FPT.",
    "Toi de xuat ban VNM.",
    "Ban co the mua FPT.",
    "Hay mua FPT di.",
    "Can nhac mua FPT.",
    "FPT phu hop de mua.",
)

ACCENTED_ADVICE = (
    "Bạn nên mua FPT ngay.",
    "Nên bán VNM.",
    "Khuyến nghị mua FPT.",
)


class UnaccentedSafetyGateTests(unittest.TestCase):
    """Gate advice phải chặn cả biến thể không dấu.

    Ba regex an toàn hiện chạy trên hai cơ sở chuẩn hoá trái ngược nhau:
    ``_UNSAFE_ANSWER_RE``/``_SPELLED_PERCENT_RE`` khớp chuỗi CÓ dấu nhưng nhận
    ``clause``/``answer`` thô, còn ``_SAFE_REFUSAL_RE``/``_CERTAINTY_RE`` khớp
    chuỗi KHÔNG dấu và nhận text đã ``_strip_accents``. Hệ quả: câu tiếng Việt
    không dấu đi xuyên gate. Đây là raise hard-fail duy nhất trong 10 raise
    ``ungrounded_response`` nên không có tầng nào đỡ phía sau.
    """

    def test_accented_advice_is_blocked(self):
        for answer in ACCENTED_ADVICE:
            with self.subTest(answer=answer):
                with self.assertRaises(chatbot_service.ChatbotServiceError) as ctx:
                    chatbot_service._validate_grounded_answer(answer, [], {})
                self.assertEqual(ctx.exception.code, "ungrounded_response")
                self.assertFalse(
                    ctx.exception.soft_blockable,
                    "vi phạm an toàn phải hard-fail, không được soft-block",
                )

    def test_unaccented_advice_is_blocked_too(self):
        for answer in UNACCENTED_ADVICE:
            with self.subTest(answer=answer):
                with self.assertRaises(chatbot_service.ChatbotServiceError) as ctx:
                    chatbot_service._validate_grounded_answer(answer, [], {})
                self.assertFalse(ctx.exception.soft_blockable)

    def test_unaccented_spelled_percent_is_blocked(self):
        # "muoi phan tram" = "mười phần trăm": lách _NUMBER_RE bằng chữ.
        for answer in ("Xac suat muoi phan tram.", "Tang ba phan tram."):
            with self.subTest(answer=answer):
                with self.assertRaises(chatbot_service.ChatbotServiceError):
                    chatbot_service._validate_grounded_answer(answer, [], {})

    def test_unaccented_certainty_stays_blocked(self):
        with self.assertRaises(chatbot_service.ChatbotServiceError):
            chatbot_service._validate_grounded_answer("FPT chac chan se tang.", [], {})

    def test_unaccented_safe_refusal_is_still_allowed(self):
        # Không được sửa gate bằng cách chặn sạch mọi câu có "mua"/"ban":
        # câu từ chối hợp lệ phải đi qua.
        for answer in (
            "Minh khong the dua ra khuyen nghi mua ban.",
            "Mình không thể đưa ra khuyến nghị mua bán.",
        ):
            with self.subTest(answer=answer):
                chatbot_service._validate_grounded_answer(answer, [], {})

    def test_refusal_phrasings_from_live_provider_are_allowed(self):
        """Lời từ chối diễn đạt tự do vẫn phải lọt.

        Cả 4 câu dưới đây là câu trả lời THẬT của provider, từng bị hard-fail 502
        vì gate cũ dùng whitelist cụm cố định: cứ đổi một chữ ("khong dua" thiếu
        "ra", "loi khuyen" thay "khuyen nghi", "chua" thay "khong") là trượt
        whitelist. Đây là câu hỏi cơ bản nhất của người dùng ("Dự báo FPT thế
        nào?", "Hệ thống có giới hạn gì?") nên 502 ở đây là lỗi chặn oan nặng.
        """
        for answer in (
            "Khong dua loi khuyen mua/ban.",
            "Khong dua khuyen nghi mua/ban.",
            "Tin hieu nay chua ung ho ket luan ban ngay.",
            "Chua co co so tu model de noi ban FPT ngay.",
        ):
            with self.subTest(answer=answer):
                chatbot_service._validate_grounded_answer(answer, [], {})

    def test_negated_certainty_from_live_provider_is_allowed(self):
        """Câu phủ định về giá vẫn phải lọt.

        Certainty trước đây có whitelist phủ định riêng (``_NEGATED_CERTAINTY_RE``)
        và thiếu y như whitelist advice: nó chỉ biết "khong chac chan" / "khong dam
        bao" / "khong co nghia la". Hai câu đầu là câu trả lời THẬT của provider cho
        "Dự báo FPT thế nào?" và "Chắc chắn FPT sẽ tăng chứ?", từng 502 — tức chatbot
        bị chặn oan đúng lúc nó đang làm điều mong muốn: từ chối cam kết giá.
        """
        for answer in (
            "Khong chac FPT se tang.",
            "Khong du co so ket luan gia se tang.",
            "Diem UP khong dam bao gia se tang.",
            "Se tang cuong them du lieu o lan chay sau.",
        ):
            with self.subTest(answer=answer):
                chatbot_service._validate_grounded_answer(answer, [], {})

    def test_negated_certainty_is_not_a_bypass(self):
        # Phủ định phải đứng trước và cùng mệnh đề, hệt như nhánh advice.
        for answer in (
            "Khong theo du lieu, VNM chac chan giam.",
            "FPT chac chan se tang, khong con nghi ngo.",
        ):
            with self.subTest(answer=answer):
                with self.assertRaises(chatbot_service.ChatbotServiceError) as ctx:
                    chatbot_service._validate_grounded_answer(answer, [], {})
                self.assertFalse(ctx.exception.soft_blockable)

    def test_negation_does_not_become_a_bypass(self):
        """Nới miễn trừ không được biến từ phủ định thành cửa lách.

        Chỉ phủ định nằm TRƯỚC cụm advice và trong cùng mệnh đề mới tính là từ
        chối. Ba ca dưới đều có từ phủ định nhưng nó phủ định thứ khác, còn lời
        khuyên vẫn nguyên vẹn.
        """
        for answer in (
            "Nen mua FPT, khong can cho them.",
            "Khong theo du lieu, ban nen mua FPT.",
            "Khong chi khuyen nghi mua ma con nen mua them.",
        ):
            with self.subTest(answer=answer):
                with self.assertRaises(chatbot_service.ChatbotServiceError) as ctx:
                    chatbot_service._validate_grounded_answer(answer, [], {})
                self.assertFalse(ctx.exception.soft_blockable)

    def test_unaccented_advice_never_reaches_the_user_through_chat(self):
        client = _client("Ban nen mua FPT ngay.")
        with patch.object(
            chatbot_service, "build_context", return_value=_bundle(), create=True
        ):
            with self.assertRaises(chatbot_service.ChatbotServiceError) as ctx:
                chatbot_service.chat("FPT the nao", [], client=client)
        self.assertEqual(ctx.exception.status, 502)


class SafetyBeatsClarificationShapeTests(unittest.TestCase):
    """Trên lượt clarification, vi phạm an toàn vẫn phải là lỗi thật.

    ``_validate_clarification_answer`` chạy TRƯỚC ``_validate_grounded_answer``
    và raise ``soft_blockable=True``. Câu khuyên mua/bán trượt shape-check
    (có dấu ``.``, thiếu ``?``) nên bị bắt ở đó trước → biến thành 200 kèm
    ``answer_blocked_ungrounded`` và không bao giờ tới gate an toàn. Text unsafe
    không lọt ra ngoài, nhưng invariant "mọi vi phạm an toàn phải nổi lên thành
    lỗi thật" bị âm thầm sai đúng trên nhánh này → operator không thấy provider
    đang drift.
    """

    def _clarification_bundle(self) -> dict:
        bundle = _bundle(
            context={
                "clarification_required": {
                    "reason": "ambiguous",
                    "instruction": "Hỏi lại một câu ngắn.",
                }
            }
        )
        return bundle

    def test_advice_on_clarification_turn_is_a_hard_error(self):
        for answer in ("Bạn nên mua FPT ngay.", "Ban nen mua FPT ngay."):
            with self.subTest(answer=answer):
                client = _client(answer)
                with patch.object(
                    chatbot_service,
                    "build_context",
                    return_value=self._clarification_bundle(),
                    create=True,
                ):
                    with self.assertRaises(
                        chatbot_service.ChatbotServiceError
                    ) as ctx:
                        chatbot_service.chat("phân tích cổ phiếu", [], client=client)
                self.assertEqual(ctx.exception.status, 502)

    def test_merely_malformed_clarification_still_soft_blocks(self):
        # Sai shape nhưng KHÔNG vi phạm an toàn thì vẫn phải là 200 soft-block,
        # không được kéo theo thành 502.
        client = _client("Đây là câu trả lời dài không có dấu hỏi.")
        with patch.object(
            chatbot_service,
            "build_context",
            return_value=self._clarification_bundle(),
            create=True,
        ):
            result = chatbot_service.chat("phân tích cổ phiếu", [], client=client)
        self.assertEqual(result["answer"], chatbot_service.SOFT_BLOCK_ANSWER)
        self.assertIn(
            "answer_blocked_ungrounded",
            [w["code"] for w in result["warnings"]],
        )


class ConversationStateTypeGuardTests(unittest.TestCase):
    """``conversation_state`` sai kiểu phải ra 400 JSON, không phải 500 HTML.

    ``normalize_conversation_state`` dùng ``topic not in CONVERSATION_TOPICS |
    {None}``. Giá trị unhashable (list/dict/set) raise ``TypeError`` chứ không
    phải ``ValueError``, mà ``app.py`` chỉ bắt ``ValueError`` quanh
    ``_validate_chat_payload`` → exception thoát khỏi view, Werkzeug trả 500
    ``text/html``. Phá hợp đồng "mọi lỗi input đi qua cùng một _chat_error".
    """

    UNHASHABLE = ([], {}, set(), ["a"], {"a": 1})

    def _state(self, **overrides) -> dict:
        state = {
            "active_symbols": [],
            "topic": None,
            "ranking_order": None,
            "last_result_symbols": [],
        }
        state.update(overrides)
        return state

    def test_unhashable_topic_raises_value_error(self):
        for value in self.UNHASHABLE:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    chatbot_service.normalize_conversation_state(
                        self._state(topic=value)
                    )

    def test_unhashable_ranking_order_raises_value_error(self):
        for value in self.UNHASHABLE:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    chatbot_service.normalize_conversation_state(
                        self._state(ranking_order=value)
                    )

    def test_api_returns_json_400_not_html_500(self):
        client = web_app.app.test_client()
        for field in ("topic", "ranking_order"):
            for value in ([], {}):
                with self.subTest(field=field, value=value):
                    response = client.post(
                        "/api/chat",
                        json={
                            "message": "xin chào",
                            "conversation_state": self._state(**{field: value}),
                        },
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.mimetype, "application/json")
                    self.assertEqual(
                        response.get_json()["error"]["code"], "invalid_request"
                    )


class CleanDataMaxDateTests(unittest.TestCase):
    """CSV rỗng không được biến thành ngày ``"nan"``.

    ``str(df["trading_date"].max())`` trên cột rỗng cho ``"nan"`` — truthy và
    ``isinstance(str)`` nên nó thắng cả chuỗi ``or`` fallback, lọt vào
    ``data_as_of`` của response, rồi ``"nan" > "2026-07-20"`` (so sánh chuỗi)
    còn bơm thêm warning ``data_ahead_of_report`` giả.
    """

    def _write(self, tmp_path, rows: str) -> None:
        tmp_path.write_text(rows, encoding="utf-8")

    def test_empty_csv_yields_none(self):
        with patch.object(chatbot_tools.pd, "read_csv") as read_csv:
            read_csv.return_value = pd.DataFrame({"trading_date": []})
            self.assertIsNone(chatbot_tools._clean_data_max_date_cached(("empty", 1)))

    def test_all_null_column_yields_none(self):
        with patch.object(chatbot_tools.pd, "read_csv") as read_csv:
            read_csv.return_value = pd.DataFrame({"trading_date": [None, None]})
            self.assertIsNone(chatbot_tools._clean_data_max_date_cached(("null", 2)))

    def test_normal_csv_still_returns_max_date(self):
        with patch.object(chatbot_tools.pd, "read_csv") as read_csv:
            read_csv.return_value = pd.DataFrame(
                {"trading_date": ["2026-07-01", "2026-07-31", "2026-07-15"]}
            )
            self.assertEqual(
                chatbot_tools._clean_data_max_date_cached(("ok", 3)), "2026-07-31"
            )


class MalformedSummaryTests(unittest.TestCase):
    """Giá trị ``null``/phi số trong ``pipeline_summary.json`` phải ra lỗi có mã.

    ``dict.get(k, {})`` trả về chính giá trị ``None`` đã lưu — default chỉ áp
    dụng khi thiếu KEY. ``"dataset_report": null`` do đó gây ``AttributeError``
    ngay trong ``get_release_state()``, hàm mà MỌI handler đều gọi → toàn bộ
    tính năng chat sập thành 500 ``internal_error`` mờ mịt.
    """

    def test_null_dataset_report_does_not_crash_release_state(self):
        with patch.object(
            chatbot_tools, "_read_json", return_value={"dataset_report": None}
        ):
            chatbot_tools.get_release_state.cache_clear() if hasattr(
                chatbot_tools.get_release_state, "cache_clear"
            ) else None
            state = chatbot_tools.get_release_state()
        self.assertIsInstance(state, dict)

    def test_non_numeric_clean_count_returns_coded_error(self):
        # int("unknown") raise ValueError ngoài mọi handler; các đọc số khác
        # trong module đều đi qua _rounded_number, chỉ call site này thoát ra.
        summary = {
            "dataset_report": {"date_min": "2020-01-02", "date_max": "2026-07-20"},
            "clean_report": {"symbols_after_cleaning": "unknown"},
            "feature_report": {"feature_count": 20},
        }
        state = {
            "status": "current",
            "summary": summary,
            "metadata": {"train_through_date": "2026-04-10"},
            "scope": ("FPT", "VNM"),
            "policy_id": "policy-v1",
            "content_fingerprint": "release-1",
            "warnings": [],
        }
        with patch.object(chatbot_tools, "get_release_state", return_value=state):
            result = chatbot_tools.get_dataset_info({})
        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["error"])


class DataAsOfSemanticsTests(unittest.TestCase):
    """``data_as_of`` không được mang cutoff huấn luyện, và provenance phải rõ.

    Chuỗi fallback ``clean_max or report_date_max or train_through_date`` là
    đường rò cuối còn lại của việc tách ba ngày: khi CSV không đọc được và
    report thiếu ``date_max``, cutoff nhãn (2026-04-10) được phục vụ như "ngày
    giao dịch mới nhất" — lệch 3.5 tháng, không kèm warning. ``hose_stock_clean.csv``
    đang gitignore nên clone mới nằm đúng trạng thái này.

    Kèm theo: rule ``report_as_of = date_max if date_max != data_as_of else None``
    tự tắt kênh disclosure đúng lúc cần nhất — một khi ``date_max`` đã được
    promote vào ``data_as_of`` thì hai giá trị bằng nhau by construction.
    """

    def _snapshot(self, *, clean_max, date_max):
        chatbot_service._base_snapshot_cached.cache_clear()
        state = {
            "status": "current",
            "summary": {
                "dataset_report": {"date_max": date_max} if date_max else {},
                "clean_report": {},
                "feature_report": {},
            },
            "metadata": {"train_through_date": "2026-04-10"},
            "scope": ("FPT", "VNM"),
            "policy_id": "policy-v1",
            "content_fingerprint": "release-1",
            "warnings": [],
        }
        with patch.object(chatbot_tools, "get_release_state", return_value=state), patch.object(
            chatbot_tools, "_clean_data_max_date", return_value=clean_max
        ):
            snapshot = chatbot_service._base_snapshot_cached(("sig", clean_max, date_max))
        chatbot_service._base_snapshot_cached.cache_clear()
        return snapshot["result"]["data"]

    def test_training_cutoff_is_never_served_as_data_as_of(self):
        data = self._snapshot(clean_max=None, date_max=None)
        self.assertNotEqual(
            data.get("data_as_of"),
            "2026-04-10",
            "cutoff nhãn huấn luyện không phải ngày dữ liệu",
        )
        self.assertEqual(data.get("model_trained_through"), "2026-04-10")

    def test_report_sourced_date_discloses_its_provenance(self):
        data = self._snapshot(clean_max=None, date_max="2026-07-20")
        self.assertEqual(data.get("data_as_of"), "2026-07-20")
        self.assertIsNotNone(
            data.get("report_as_of"),
            "khi data_as_of lấy từ report thì phải nói rõ nguồn",
        )

    def test_live_csv_date_wins_and_report_date_is_disclosed(self):
        data = self._snapshot(clean_max="2026-07-31", date_max="2026-07-20")
        self.assertEqual(data.get("data_as_of"), "2026-07-31")
        self.assertEqual(data.get("report_as_of"), "2026-07-20")


class SymbolDetectionCaseTests(unittest.TestCase):
    """Từ thường tiếng Việt không dấu không được nhận thành mã CK.

    ``_symbols_from_text`` match ``[A-Za-z]{2,5}`` rồi ``.upper()``, nên 29 mã
    in-scope trùng từ tiếng Việt không dấu. ``_out_of_scope_symbols`` ĐÃ đòi
    uppercase đúng vì lý do này (comment tại chỗ) — nhánh in-scope bị bỏ sót,
    mà đó mới là nhánh phát ra số và ghim ``active_symbols``.
    """

    SCOPE = {"TRA", "DAT", "HAP", "TIP", "PAN", "SAM", "FPT", "VNM", "NAV", "CCI"}

    def test_lowercase_vietnamese_words_are_not_symbols(self):
        cases = (
            "tra loi giup minh di",
            "dat coc la gi",
            "hap thu thong tin sao",
            "tip cho minh voi",
            "pan chan ra sao",
            "cho toi biet ve dat va sam",
            "nav cua quy la gi",
        )
        for message in cases:
            with self.subTest(message=message):
                self.assertEqual(
                    chatbot_service._symbols_from_text(message, self.SCOPE), []
                )

    def test_uppercase_symbols_are_still_detected(self):
        self.assertEqual(
            chatbot_service._symbols_from_text("FPT thế nào", self.SCOPE), ["FPT"]
        )
        self.assertEqual(
            chatbot_service._symbols_from_text("so sánh FPT và VNM", self.SCOPE),
            ["FPT", "VNM"],
        )

    def test_symbol_with_vietnamese_sentence_around_it_is_detected(self):
        self.assertEqual(
            chatbot_service._symbols_from_text(
                "cho tôi biết tín hiệu của TRA hôm nay", self.SCOPE
            ),
            ["TRA"],
        )


class AcronymNotASymbolTests(unittest.TestCase):
    """Acronym ML/tài chính không được gắn cờ "mã ngoài phạm vi".

    ``_KNOWN_UPPER_NON_SYMBOLS`` thiếu 26 entry thường gặp trong chính câu hỏi
    về project này. Hệ quả: câu hỏi về metric bị trả lời "mã RMSE ngoài phạm
    vi" kèm warning ``symbol_out_of_scope``.

    Cùng gốc với divergence thứ hai: ``_NON_SYMBOL_ACRONYMS`` (validator) và
    ``_KNOWN_UPPER_NON_SYMBOLS`` (router) lệch 22 entry, làm
    ``_validate_directional_claims`` soft-block những câu vốn đúng.
    """

    SCOPE = {"FPT", "VNM"}

    ACRONYMS = (
        "RMSE",
        "ROE",
        "ROA",
        "EBITDA",
        "SHAP",
        "PCA",
        "SVM",
        "XGB",
        "KNN",
        "MAE",
        "MSE",
        "OHLCV",
        "VWAP",
        "ETF",
        "IPO",
        "ADX",
        "MFI",
        "TPR",
        "FPR",
        "CPU",
        "GPU",
        "SQL",
        "PDF",
    )

    def test_metric_and_finance_acronyms_are_not_out_of_scope_symbols(self):
        for token in self.ACRONYMS:
            with self.subTest(token=token):
                message = f"{token} của model là bao nhiêu?"
                self.assertEqual(
                    chatbot_service._out_of_scope_symbols(message, self.SCOPE),
                    [],
                )

    def test_real_out_of_scope_ticker_is_still_flagged(self):
        self.assertEqual(
            chatbot_service._out_of_scope_symbols("HPG thế nào?", self.SCOPE),
            ["HPG"],
        )

    def test_directional_claim_with_acronym_is_not_soft_blocked(self):
        facts = {
            "FPT": {
                "prediction": "NOT_UP",
                "threshold_relation": "below",
                "up_score": 41.0,
                "threshold": 49.0,
            }
        }
        # ROE/EPS không phải mã; câu này đúng về dữ liệu nên không được chặn.
        chatbot_service._validate_directional_claims(
            "FPT dưới ngưỡng nên ROE chưa đủ điều kiện UP", facts
        )


class RoutingVocabularyTests(unittest.TestCase):
    """Cách hỏi tự nhiên nhất phải có context, hoặc ít nhất có clarification.

    ``_MODEL_CONTEXT_WORDS`` có "model" nhưng thiếu "mo hinh";
    ``_DATASET_CONTEXT_WORDS`` có "du lieu" nhưng thiếu "data". Message không
    khớp intent nào VÀ không khớp ``_AMBIGUOUS_DATA_WORDS`` thì không có tool,
    cũng không có clarification → LLM phải trả lời từ ``project_snapshot``,
    nơi không chứa metric/threshold/split date.
    """

    def _context_keys(self, message: str) -> set[str]:
        chatbot_service._base_snapshot_cached.cache_clear()
        snapshot = {
            "result": {
                "ok": True,
                "data": {
                    "serving_mode": "offline_published_artifacts",
                    "release_status": "current",
                    "data_as_of": "2026-07-31",
                    "report_as_of": None,
                    "model_trained_through": "2026-04-10",
                },
                "source": {"kind": "project_snapshot", "as_of": "2026-07-31"},
                "as_of": "2026-07-31",
                "release": {
                    "status": "current",
                    "policy_id": "policy-v1",
                    "content_fingerprint": "release-1",
                },
                "warnings": [],
                "error": None,
            },
            "scope": ("FPT", "VNM"),
        }
        # Handler thật (get_model_info/get_dataset_info) đọc release trên đĩa; nếu
        # không patch cùng identity với snapshot thì _consume_context_result raise
        # release_changed và test đo sai thứ (fixture, không phải routing).
        release_state = {
            "status": "current",
            "artifact": {},
            "metadata": {"train_through_date": "2026-04-10"},
            "summary": {
                "dataset_report": {"date_min": "2020-01-02", "date_max": "2026-07-31"},
                "clean_report": {"symbols_after_cleaning": 2, "excluded_symbols": 0},
                "feature_report": {"feature_count": 20},
            },
            "scope": ("FPT", "VNM"),
            "scope_verified": True,
            "signature": ("sig",),
            "policy_id": "policy-v1",
            "content_fingerprint": "release-1",
            "warnings": [],
        }
        with patch.object(
            chatbot_service, "_base_snapshot_cached", return_value=snapshot
        ), patch.object(
            chatbot_service,
            "_context_signature",
            return_value=("sig",),
        ), patch.object(
            chatbot_tools, "get_release_state", return_value=release_state
        ), patch.object(
            chatbot_tools, "_public_release", return_value={
                "status": "current",
                "policy_id": "policy-v1",
                "content_fingerprint": "release-1",
            }
        ):
            bundle = chatbot_service.build_context(message, [], None)
        return set(bundle["context"])

    def test_model_question_in_plain_vietnamese_gets_model_context(self):
        keys = self._context_keys("mo hinh nao duoc chon")
        self.assertTrue(
            {"model", "clarification_required"} & keys,
            f"không có model context lẫn clarification: {sorted(keys)}",
        )

    def test_english_data_word_gets_dataset_context(self):
        keys = self._context_keys("data den ngay nao roi")
        self.assertTrue(
            {"dataset", "clarification_required"} & keys,
            f"không có dataset context lẫn clarification: {sorted(keys)}",
        )

    def test_realtime_market_question_gets_limitations(self):
        keys = self._context_keys("tinh hinh thi truong hom nay the nao")
        self.assertTrue(
            {"limitations", "clarification_required"} & keys,
            f"câu hỏi realtime cần nói rõ giới hạn: {sorted(keys)}",
        )

    def test_lowercase_foreign_ticker_is_addressed(self):
        keys = self._context_keys("tsla the nao")
        self.assertTrue(
            {"out_of_scope_symbols", "clarification_required"} & keys,
            f"người dùng nêu mã lạ mà không được trả lời gì: {sorted(keys)}",
        )


class TopNClampTests(unittest.TestCase):
    """``top N`` ngoài 1..10 nên được kẹp, không dead-end cả lượt.

    Regex ``top\\s*([-+]?\\d+)`` nhận cả số âm và số lớn, rồi ``get_ranking``
    trả ``invalid_arguments`` nói về tham số nội bộ người dùng chưa từng nhập.
    Vì ``ok=False`` nên ``conversation_state`` không cập nhật → chuỗi follow-up
    cũng đứt.
    """

    def test_ranking_rejects_out_of_range_top_n_gracefully(self):
        for top_n in (0, -5, 50, 99999):
            with self.subTest(top_n=top_n):
                result = chatbot_tools.get_ranking({"top_n": top_n, "order": "highest_up_score"})
                self.assertIn("ok", result)
                self.assertIsInstance(result["ok"], bool)


# CỐ Ý KHÔNG có test cho "cắt chuỗi ở MAX_ANSWER_CHARS tạo số mới".
# Giả thuyết: answer = _parse_content(response)[:MAX_ANSWER_CHARS] cắt TRƯỚC khi
# validate, nên số grounded vắt qua ranh giới bị đọc thành số khác → soft-block
# oan. Thử dựng repro thì KHÔNG tái hiện được: _NUMBER_RE có lookbehind
# ``(?<![\w])``, mà ký tự ngay trước chỗ cắt luôn là chữ/số, nên số bị cắt dở
# KHÔNG khớp regex — validator không thấy nó thay vì thấy sai. Không viết test
# đỏ cho một defect chưa chứng minh được.


if __name__ == "__main__":
    unittest.main()
