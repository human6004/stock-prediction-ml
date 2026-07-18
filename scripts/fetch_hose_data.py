"""Fetch OHLCV tăng dần từ vnstock rồi merge vào raw CSV.

Mỗi mã được chuẩn hóa về cùng schema OHLCV. `new_rows` trong fetch report đếm
record tải về trước dedupe; số dòng raw tăng ròng có thể nhỏ hơn.
"""

import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config.settings import (  # noqa: E402
    FETCH_END_DATE,
    FETCH_MAX_RETRIES,
    FETCH_REPORT_PATH,
    FETCH_SLEEP_SECONDS,
    FETCH_SOURCE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    REQUIRED_COLUMNS,
)

COLUMN_ALIASES = {
    "time": "trading_date",
    "date": "trading_date",
    "ticker": "symbol",
    "open_price": "open",
    "high_price": "high",
    "low_price": "low",
    "close_price": "close",
    "match_volume": "volume",
    "vol": "volume",
}

RATE_LIMIT_MARKERS = ("rate limit", "giới hạn", "gioi han", "20/20")


def normalize_vnstock_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên cột từ nhiều biến thể vnstock về schema chuẩn của project."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = frame.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    if "symbol" not in df.columns:
        df["symbol"] = symbol.upper()

    if "trading_date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        first_col = df.columns[0]
        if first_col not in REQUIRED_COLUMNS:
            df = df.rename(columns={first_col: "trading_date"})

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Thieu cot sau chuan hoa: {missing}")

    out = df[REQUIRED_COLUMNS].copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["trading_date"] = pd.to_datetime(out["trading_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=REQUIRED_COLUMNS)
    return out


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in RATE_LIMIT_MARKERS)


def fetch_symbol_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    from vnstock.api.quote import Quote

    last_error = None
    for attempt in range(1, FETCH_MAX_RETRIES + 1):
        try:
            quote = Quote(symbol=symbol, source=FETCH_SOURCE)
            raw = quote.history(start=start, end=end, interval="1D")
            return normalize_vnstock_frame(symbol, raw)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if is_rate_limit_error(exc):
                wait_seconds = 65
                print(f"  Rate limit {symbol}, cho {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue
            if attempt < FETCH_MAX_RETRIES:
                time.sleep(FETCH_SLEEP_SECONDS * attempt)
    raise RuntimeError(str(last_error))


def resolve_fetch_window(existing: pd.DataFrame) -> tuple[str, str, bool]:
    dates = pd.to_datetime(existing["trading_date"], errors="coerce")
    max_date = dates.max().date()
    fetch_start = (max_date + timedelta(days=1)).isoformat()
    fetch_end = FETCH_END_DATE or date.today().isoformat()
    if fetch_start > fetch_end:
        return fetch_start, fetch_end, False
    return fetch_start, fetch_end, True


def merge_and_save(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Merge an toàn theo (symbol, trading_date), bản tải mới thắng khi trùng."""
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined["symbol"] = combined["symbol"].astype(str).str.strip().str.upper()
    combined["trading_date"] = pd.to_datetime(
        combined["trading_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    combined = combined.dropna(subset=REQUIRED_COLUMNS)
    combined = combined.drop_duplicates(subset=["symbol", "trading_date"], keep="last")
    combined = combined.sort_values(["symbol", "trading_date"]).reset_index(drop=True)
    combined.to_csv(RAW_DATA_PATH, index=False)
    return combined


def main() -> None:
    raw_path = Path(RAW_DATA_PATH)
    if not raw_path.exists():
        raise FileNotFoundError(f"Khong tim thay dataset: {raw_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(raw_path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in existing.columns]
    if missing_cols:
        raise ValueError(f"Dataset thieu cot: {missing_cols}")

    symbols = sorted(existing["symbol"].astype(str).str.strip().str.upper().unique())
    fetch_start, fetch_end, should_fetch = resolve_fetch_window(existing)

    report = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "raw_path": str(raw_path),
        "symbols_in_csv": int(len(symbols)),
        "rows_before": int(len(existing)),
        "fetch_start": fetch_start,
        "fetch_end": fetch_end,
        "fetch_source": FETCH_SOURCE,
        "should_fetch": should_fetch,
        "success_symbols": [],
        "failed_symbols": [],
        "new_rows": 0,
        "rows_after": int(len(existing)),
        "date_min_after": str(pd.to_datetime(existing["trading_date"]).min().date()),
        "date_max_after": str(pd.to_datetime(existing["trading_date"]).max().date()),
    }

    if not should_fetch:
        print(f"Dataset da cap nhat den {report['date_max_after']}. Khong can fetch.")
        FETCH_REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return

    print(
        f"Fetch OHLCV: {fetch_start} -> {fetch_end} "
        f"({len(symbols)} ma, sleep={FETCH_SLEEP_SECONDS}s)"
    )
    frames = []
    combined = existing
    for index, symbol in enumerate(symbols, start=1):
        try:
            chunk = fetch_symbol_history(symbol, fetch_start, fetch_end)
            if not chunk.empty:
                frames.append(chunk)
                combined = merge_and_save(combined, chunk)
                report["rows_after"] = int(len(combined))
                report["date_max_after"] = str(
                    pd.to_datetime(combined["trading_date"]).max().date()
                )
            report["success_symbols"].append(symbol)
            print(f"[{index}/{len(symbols)}] {symbol}: {len(chunk)} dong")
        except Exception as exc:  # noqa: BLE001
            report["failed_symbols"].append({"symbol": symbol, "error": str(exc)})
            print(f"[{index}/{len(symbols)}] {symbol}: LOI - {exc}")
        time.sleep(FETCH_SLEEP_SECONDS)

    report["new_rows"] = int(sum(len(frame) for frame in frames))
    report["date_min_after"] = str(
        pd.to_datetime(combined["trading_date"]).min().date()
    )

    FETCH_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Fetch report: {FETCH_REPORT_PATH}")
    print(
        f"Hoan tat: +{report['new_rows']} dong | "
        f"OK {len(report['success_symbols'])} | "
        f"Loi {len(report['failed_symbols'])} | "
        f"date_max={report['date_max_after']}"
    )


if __name__ == "__main__":
    main()
