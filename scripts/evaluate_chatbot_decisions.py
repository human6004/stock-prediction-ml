"""Chạy bộ scenario live để kiểm tra quyết định action của chatbot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from services import chatbot_service  # noqa: E402


def _history(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


SCENARIOS = (
    ("01 greeting", "Xin chào", [], "GENERAL_CHAT", {"kind": "greeting"}),
    (
        "02 capability",
        "Bạn có thể làm gì?",
        [],
        "GENERAL_CHAT",
        {"kind": "capabilities"},
    ),
    ("03 single stock", "FPT thế nào?", [], "STOCK_SIGNAL", {"symbols": ["FPT"]}),
    (
        "04 multiple stocks",
        "FPT với HPG thế nào?",
        [],
        "STOCK_SIGNAL",
        {"symbols": ["FPT", "HPG"]},
    ),
    (
        "05 three symbols",
        "FPT, HPG và VNM thì sao?",
        [],
        "STOCK_SIGNAL",
        {"symbols": ["FPT", "HPG", "VNM"]},
    ),
    (
        "06 comparison",
        "So sánh FPT và HPG",
        [],
        "STOCK_SIGNAL",
        {"symbols": ["FPT", "HPG"]},
    ),
    (
        "07 stock follow-up",
        "Còn HPG?",
        _history("FPT thế nào?", "Đã trả tín hiệu FPT từ dữ liệu offline."),
        "STOCK_SIGNAL",
        {"symbols": ["HPG"]},
    ),
    (
        "08 comparison follow-up",
        "Còn VNM?",
        _history("So sánh FPT và HPG", "Đã so sánh tín hiệu FPT và HPG."),
        "STOCK_SIGNAL",
        {"symbols": ["VNM"]},
    ),
    (
        "09 top ranking",
        "Top 5 cổ phiếu",
        [],
        "STOCK_RANKING",
        {"order": "highest", "top_n": 5},
    ),
    (
        "10 bottom ranking",
        "5 mã score thấp nhất",
        [],
        "STOCK_RANKING",
        {"order": "lowest", "top_n": 5},
    ),
    (
        "11 best stock",
        "Mã nào tốt nhất?",
        [],
        "STOCK_RANKING",
        {"order": "highest", "top_n": 1},
    ),
    (
        "12 worst stock",
        "Mã nào tệ nhất?",
        [],
        "STOCK_RANKING",
        {"order": "lowest", "top_n": 1},
    ),
    (
        "13 ranking order",
        "Xếp từ cao xuống thấp",
        [],
        "STOCK_RANKING",
        {"order": "highest", "top_n": 5},
    ),
    (
        "14 ranking limit",
        "Top 10 hiện tại",
        [],
        "STOCK_RANKING",
        {"order": "highest", "top_n": 10},
    ),
    (
        "15 project overview",
        "Project này làm gì?",
        [],
        "PROJECT_INFO",
        {"topic": "overview"},
    ),
    (
        "16 project data",
        "Dataset lấy ở đâu và có bao nhiêu dữ liệu?",
        [],
        "PROJECT_INFO",
        {"topic": "dataset"},
    ),
    (
        "17 project features",
        "Model dùng feature gì?",
        [],
        "PROJECT_INFO",
        {"topic": "features"},
    ),
    (
        "18 project model",
        "Dùng model gì và vì sao?",
        [],
        "PROJECT_INFO",
        {"topic": "model"},
    ),
    (
        "19 project evaluation",
        "Model dùng metric gì và kết quả evaluation ra sao?",
        [],
        "PROJECT_INFO",
        {"topic": "evaluation"},
    ),
    (
        "20 project inference",
        "Khi yêu cầu dự đoán thì hệ thống chạy thế nào?",
        [],
        "PROJECT_INFO",
        {"topic": "inference"},
    ),
    (
        "21 project limitations",
        "Hạn chế của model và dữ liệu là gì?",
        [],
        "PROJECT_INFO",
        {"topic": "limitations"},
    ),
    (
        "22 project follow-up",
        "Vậy train thế nào?",
        _history("Model dùng gì?", "Đã giải thích model của project."),
        "PROJECT_INFO",
        {"topic": "model"},
    ),
    (
        "23 ambiguous prediction",
        "Dự đoán giúp tôi",
        [],
        "GENERAL_CHAT",
        {"kind": "clarify_symbol"},
    ),
    (
        "24 out of scope",
        "Viết giúp tôi game rắn săn mồi bằng Java",
        [],
        "OUT_OF_SCOPE",
        {"reason": "other"},
    ),
    (
        "25 investment advice",
        "Tôi có nên mua FPT ngay không?",
        [],
        "OUT_OF_SCOPE",
        {"reason": "trading_advice"},
    ),
    ("26 thanks", "Cảm ơn bạn", [], "GENERAL_CHAT", {"kind": "thanks"}),
    (
        "27 ambiguous request",
        "Giúp tôi với",
        [],
        "GENERAL_CHAT",
        {"kind": "clarify_request"},
    ),
    (
        "28 realtime",
        "Giá FPT realtime hiện là bao nhiêu?",
        [],
        "OUT_OF_SCOPE",
        {"reason": "realtime"},
    ),
    (
        "29 market news",
        "Tin tức mới nhất về HPG là gì?",
        [],
        "OUT_OF_SCOPE",
        {"reason": "news"},
    ),
    (
        "30 fundamentals",
        "Phân tích báo cáo tài chính của VNM",
        [],
        "OUT_OF_SCOPE",
        {"reason": "fundamentals"},
    ),
    (
        "31 explicit prediction",
        "Dự đoán FPT",
        [],
        "STOCK_SIGNAL",
        {"symbols": ["FPT"]},
    ),
    (
        "32 ranking cap",
        "Top 20 cổ phiếu",
        [],
        "STOCK_RANKING",
        {"order": "highest", "top_n": 10},
    ),
    (
        "33 five-symbol boundary",
        "So sánh FPT, HPG, VNM, MWG và VCB",
        [],
        "STOCK_SIGNAL",
        {"symbols": ["FPT", "HPG", "VNM", "MWG", "VCB"]},
    ),
)


def _validate_scenarios() -> None:
    for name, message, history, _action, _arguments in SCENARIOS:
        assert 1 <= len(message) <= chatbot_service.MAX_MESSAGE_CHARS, name
        assert len(history) <= chatbot_service.MAX_HISTORY_MESSAGES, name
        assert len(history) % 2 == 0, name
        assert sum(len(item["content"]) for item in history) <= chatbot_service.MAX_HISTORY_CHARS, name
        for index, item in enumerate(history):
            assert item["role"] == ("user" if index % 2 == 0 else "assistant"), name


def _show(action: str, arguments: dict) -> str:
    return f"{action} {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"


def main() -> int:
    _validate_scenarios()
    try:
        client = chatbot_service._create_client()
    except chatbot_service.ChatbotServiceError as exc:
        print(f"ERROR setup: {exc.code}")
        return 1
    except Exception as exc:
        print(f"ERROR setup: {type(exc).__name__}")
        return 1

    passed = 0
    for name, message, history, expected_action, expected_arguments in SCENARIOS:
        expected = _show(expected_action, expected_arguments)
        try:
            decision = chatbot_service._decide(message, history, client)
            actual = _show(decision["action"], decision["arguments"])
        except chatbot_service.ChatbotServiceError as exc:
            print(f"FAIL {name}: expected {expected}; error {exc.code}", flush=True)
            continue
        except Exception as exc:
            print(
                f"FAIL {name}: expected {expected}; error {type(exc).__name__}",
                flush=True,
            )
            continue

        if decision["action"] == expected_action and decision["arguments"] == expected_arguments:
            passed += 1
            print(f"PASS {name}: {actual}", flush=True)
        else:
            print(f"FAIL {name}: expected {expected}; actual {actual}", flush=True)

    failed = len(SCENARIOS) - passed
    print(f"SUMMARY: {passed}/{len(SCENARIOS)} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
