"""CLI dự báo một symbol bằng final_model.pkl đã train sẵn; không retrain model."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from services.prediction_service import predict_symbol  # noqa: E402


def main() -> None:
    """Đọc ``--symbol`` từ dòng lệnh rồi in kết quả dự báo của model đã publish."""
    parser = argparse.ArgumentParser(description="Predict HOSE stock direction")
    parser.add_argument("--symbol", required=True, help="Stock symbol, e.g. FPT")
    args = parser.parse_args()

    # predict_symbol dùng dữ liệu offline mới nhất, tính feature rồi áp threshold.
    # Lưu ý: nhãn UP/NOT_UP KHÔNG lấy từ ngưỡng 0.5 mặc định mà từ
    # decision_threshold đã chọn bằng OOF lúc train (lưu trong artifact), nên
    # probability_up có thể < 50% mà vẫn ra UP, hoặc ngược lại.
    result = predict_symbol(args.symbol)

    print(f"Symbol: {result['symbol']}")
    print(f"Reference date: {result['reference_date']}")
    print(f"Close: {result['close_at_reference']:.2f}")
    print(f"Prediction: {result['prediction']}")
    if result["probability_up"] is not None:
        print(f"Model UP score: {result['probability_up'] * 100:.1f}%")
    print(f"Model: {result['model_name']}")
    print(result["summary_sentence"])


if __name__ == "__main__":
    main()
