import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from services.database_service import log_prediction  # noqa: E402
from services.prediction_service import predict_symbol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict HOSE stock direction")
    parser.add_argument("--symbol", required=True, help="Stock symbol, e.g. FPT")
    parser.add_argument("--log-db", action="store_true", help="Log prediction to SQLite")
    args = parser.parse_args()

    result = predict_symbol(args.symbol)
    try:
        if args.log_db:
            log_prediction(result)
    except Exception:
        pass

    print(f"Symbol: {result['symbol']}")
    print(f"Reference date: {result['reference_date']}")
    print(f"Close: {result['close_at_reference']:.2f}")
    print(f"Prediction: {result['prediction']}")
    if result["probability_up"] is not None:
        print(f"Probability UP: {result['probability_up'] * 100:.1f}%")
    print(f"Model: {result['model_name']}")
    print(result["summary_sentence"])


if __name__ == "__main__":
    main()
