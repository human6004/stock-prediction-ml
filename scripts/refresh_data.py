"""Background job: fetch raw OHLCV, preprocess, and rebuild ml_dataset.csv."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    print("[1/3] Fetch OHLCV tu vnstock...")
    from scripts.fetch_hose_data import main as fetch_main

    fetch_main()

    print("\n[2/3] Preprocess raw data...")
    from scripts.preprocess_data import main as preprocess_main

    preprocess_main()

    print("\n[3/3] Build features + ml_dataset.csv...")
    from scripts.build_features import main as build_main

    build_main()

    print("\nRefresh data completed.")


if __name__ == "__main__":
    main()
