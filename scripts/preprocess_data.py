"""CLI mỏng cho bước làm sạch: đọc raw, clean, rồi ghi CSV/report chất lượng."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from services.preprocessing import clean_data, dataset_check, write_clean_outputs  # noqa: E402


def main() -> None:
    # Logic thật nằm trong services.preprocessing; script chỉ nối các bước IO.
    raw_df, report = dataset_check()
    cleaned_all, cleaned_for_training, symbol_stats, clean_report = clean_data(raw_df)
    write_clean_outputs(cleaned_all, symbol_stats)
    print("Preprocess done.")
    print(f"Rows cleaned: {clean_report['rows_after_cleaning']}")
    print(f"Eligible symbols: {clean_report['eligible_symbols']}")
    print(f"Rows for training: {clean_report['rows_after_symbol_filter']}")


if __name__ == "__main__":
    main()
