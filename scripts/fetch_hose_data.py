"""Fetch OHLCV tăng dần từ vnstock rồi merge vào raw CSV.

Mỗi mã được chuẩn hóa về cùng schema OHLCV. `new_rows` trong fetch report đếm
record tải về trước dedupe; số dòng raw tăng ròng có thể nhỏ hơn.
"""

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
from services.pipeline_utils import write_json  # noqa: E402

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
EMPTY_DATA_MARKERS = ("dữ liệu trống", "du lieu trong")


def normalize_vnstock_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên cột từ nhiều biến thể vnstock về schema chuẩn của project.

    vnstock đổi tên cột theo source/phiên bản (``time`` vs ``date``, ``vol`` vs
    ``match_volume``, có/không có ``ticker``), nên hàm này là lớp adapter: sau khi
    chạy xong, phần còn lại của pipeline chỉ thấy đúng ``REQUIRED_COLUMNS``.

    Các bước dễ gây nhầm:
    - Hạ toàn bộ tên cột về lowercase TRƯỚC khi map alias, vì source có thể trả
      "Time"/"Close".
    - Nếu không có cột ``trading_date`` mà index là ``DatetimeIndex`` → ngày đang
      nằm ở index, phải ``reset_index()`` rồi đổi tên cột đầu. Điều kiện
      ``first_col not in REQUIRED_COLUMNS`` tránh đổi tên nhầm một cột hợp lệ.
    - Thiếu cột sau chuẩn hóa → raise ngay, KHÔNG tự bù giá trị: bù sẽ tạo dữ liệu
      giá giả trong dataset raw.
    - ``errors="coerce"`` biến giá trị lỗi thành NaN rồi ``dropna`` loại cả dòng,
      nên một ô rác không kéo theo cả phiên sai kiểu vào raw CSV.
    """
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


def is_empty_data_error(exc: Exception) -> bool:
    """Nhận diện lỗi rỗng kể cả khi vnstock bọc nó trong RetryError."""
    while exc:
        if any(marker in str(exc).lower() for marker in EMPTY_DATA_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def fetch_symbol_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Tải lịch sử 1 mã và xử lý theo loại phản hồi.

    - vnstock báo ``Dữ liệu trống``: coi là 0 dòng mới, không phải lỗi fetch.
    - Lỗi rate limit (nhận diện qua ``RATE_LIMIT_MARKERS`` trong message, kể cả
      ``SystemExit`` do vnai phát ra): chờ cố định 65s rồi ``continue`` —
      ``continue`` KHÔNG tăng
      ``attempt`` một cách hữu ích theo backoff, vì nghỉ đủ lâu là đủ, và 65s
      chọn > 60s để chắc chắn vượt qua cửa sổ đếm 1 phút của API.
    - Lỗi khác: backoff tuyến tính ``FETCH_SLEEP_SECONDS * attempt``.

    Nhận diện rate limit bằng chuỗi trong message là do vnstock không phơi ra mã
    lỗi riêng; đây là điểm dễ hỏng khi library đổi wording.

    Hết ``FETCH_MAX_RETRIES`` mà vẫn lỗi → raise ``RuntimeError`` với lỗi cuối,
    để caller ghi mã đó vào ``failed_symbols`` chứ không dừng cả phiên fetch.
    """
    from vnstock.api.quote import Quote

    last_error = None
    for attempt in range(1, FETCH_MAX_RETRIES + 1):
        try:
            quote = Quote(symbol=symbol, source=FETCH_SOURCE)
            raw = quote.history(start=start, end=end, interval="1D")
            return normalize_vnstock_frame(symbol, raw)
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            if isinstance(exc, SystemExit) and not is_rate_limit_error(exc):
                raise
            last_error = exc
            if is_empty_data_error(exc):
                return pd.DataFrame(columns=REQUIRED_COLUMNS)
            if is_rate_limit_error(exc):
                wait_seconds = 65
                print(f"  Rate limit {symbol}, cho {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue
            if attempt < FETCH_MAX_RETRIES:
                time.sleep(FETCH_SLEEP_SECONDS * attempt)
    raise RuntimeError(str(last_error))


def resolve_fetch_windows(
    existing: pd.DataFrame, *, end: str | None = None
) -> dict[str, tuple[str, str, bool]]:
    """Resolve an incremental window from each symbol's own latest row.

    Mỗi mã có cửa sổ RIÊNG, không dùng một ``fetch_start`` chung, vì các mã có thể
    lệch ngày cuối (mã mới lên sàn, mã bị lỗi ở lần fetch trước). Dùng mốc chung sẽ
    hoặc tải lại rất nhiều dữ liệu đã có, hoặc bỏ sót phiên của mã tụt lại.

    Trả về ``{symbol: (start, end, should_fetch)}`` với ``start`` = ngày cuối đã có
    + 1 ngày (nên không tải trùng phiên cuối), và ``should_fetch = start <= end``
    → mã đã cập nhật tới ``end`` sẽ bị bỏ qua, tiết kiệm cả rate limit lẫn thời gian.
    """
    frame = existing[["symbol", "trading_date"]].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"], errors="raise")
    fetch_end = end or FETCH_END_DATE or date.today().isoformat()
    windows = {}
    for symbol, latest in frame.groupby("symbol")["trading_date"].max().items():
        fetch_start = (latest.date() + timedelta(days=1)).isoformat()
        windows[symbol] = (fetch_start, fetch_end, fetch_start <= fetch_end)
    return windows


def merge_and_save(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Merge an toàn theo (symbol, trading_date), bản tải mới thắng khi trùng.

    ``keep="last"`` sau khi concat [cũ, mới] nghĩa là bản vừa tải GHI ĐÈ bản cũ
    cùng (mã, ngày) — đúng ý muốn khi sàn điều chỉnh lại giá/khối lượng của phiên
    đã đóng.

    Ghi file theo kiểu atomic (ghi ``.tmp`` cạnh đích rồi ``replace``) để tiến
    trình khác đang đọc raw CSV không bao giờ thấy file ghi dở. ``unlink`` trong
    ``finally`` dọn tmp nếu ghi lỗi giữa đường; nếu ``replace`` đã thành công thì
    tmp không còn tồn tại nên ``missing_ok=True``.
    """
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined["symbol"] = combined["symbol"].astype(str).str.strip().str.upper()
    combined["trading_date"] = pd.to_datetime(
        combined["trading_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    combined = combined.dropna(subset=REQUIRED_COLUMNS)
    combined = combined.drop_duplicates(subset=["symbol", "trading_date"], keep="last")
    combined = combined.sort_values(["symbol", "trading_date"]).reset_index(drop=True)
    target = Path(RAW_DATA_PATH)
    temp_path = target.with_suffix(target.suffix + ".tmp")
    try:
        combined.to_csv(temp_path, index=False)
        temp_path.replace(target)
    finally:
        temp_path.unlink(missing_ok=True)
    return combined


def main() -> None:
    """Cập nhật tăng dần ``hose_stock.csv``: chỉ fetch phần dữ liệu còn thiếu.

    Thiết kế "incremental", không tải lại từ đầu:
    - Danh sách mã lấy TỪ CHÍNH file raw hiện có, không từ config. File raw là
      nguồn sự thật duy nhất về phạm vi mã đang theo dõi; muốn thêm mã mới thì
      thêm dòng vào CSV, script sẽ tự bù lịch sử cho mã đó.
    - ``resolve_fetch_windows`` trả ``(start, end, should_fetch)`` RIÊNG cho từng
      mã, vì các mã không nhất thiết cùng ngày cuối (mã mới thêm còn trống hẳn,
      mã bị lỗi lần trước bị thiếu vài phiên). Không mã nào cần fetch →
      ghi report rồi ``return`` sớm, tiết kiệm hoàn toàn lời gọi mạng.
    - ``time.sleep(FETCH_SLEEP_SECONDS)`` sau MỖI mã, kể cả mã lỗi: rate limit
      của vnstock tính theo số request, không theo số request thành công.
    - ``except Exception`` mỗi mã để một mã lỗi không giết cả vòng lặp; lỗi được
      gom vào ``failed_symbols``.

    Điểm quan trọng ở cuối: nếu có bất kỳ mã lỗi thì ``raise RuntimeError`` DÙ đã
    ghi file thành công. Dữ liệu mới vẫn được giữ (không mất công fetch), nhưng
    exit code khác 0 báo cho ``refresh_data.py`` biết dataset đang KHUYẾT phiên —
    tránh việc train trên dữ liệu thiếu mà tưởng là đủ.
    """
    raw_path = Path(RAW_DATA_PATH)
    if not raw_path.exists():
        raise FileNotFoundError(f"Khong tim thay dataset: {raw_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(raw_path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in existing.columns]
    if missing_cols:
        raise ValueError(f"Dataset thieu cot: {missing_cols}")

    symbols = sorted(existing["symbol"].astype(str).str.strip().str.upper().unique())
    windows = resolve_fetch_windows(existing)
    should_fetch = any(window[2] for window in windows.values())
    fetch_end = FETCH_END_DATE or date.today().isoformat()
    fetch_start = min((window[0] for window in windows.values()), default=fetch_end)

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
        write_json(FETCH_REPORT_PATH, report)
        return

    print(
        f"Fetch OHLCV: {fetch_start} -> {fetch_end} "
        f"({len(symbols)} ma, sleep={FETCH_SLEEP_SECONDS}s)"
    )
    frames = []
    combined = existing
    for index, symbol in enumerate(symbols, start=1):
        symbol_start, symbol_end, symbol_should_fetch = windows[symbol]
        if not symbol_should_fetch:
            report["success_symbols"].append(symbol)
            print(f"[{index}/{len(symbols)}] {symbol}: da cap nhat")
            continue
        try:
            chunk = fetch_symbol_history(symbol, symbol_start, symbol_end)
            if not chunk.empty:
                frames.append(chunk)
            report["success_symbols"].append(symbol)
            print(f"[{index}/{len(symbols)}] {symbol}: {len(chunk)} dong")
        except Exception as exc:  # noqa: BLE001
            report["failed_symbols"].append({"symbol": symbol, "error": str(exc)})
            print(f"[{index}/{len(symbols)}] {symbol}: LOI - {exc}")
        time.sleep(FETCH_SLEEP_SECONDS)

    if frames:
        combined = merge_and_save(existing, pd.concat(frames, ignore_index=True))
    report["rows_after"] = int(len(combined))
    report["date_max_after"] = str(pd.to_datetime(combined["trading_date"]).max().date())
    report["new_rows"] = int(sum(len(frame) for frame in frames))
    report["date_min_after"] = str(
        pd.to_datetime(combined["trading_date"]).min().date()
    )

    write_json(FETCH_REPORT_PATH, report)
    print(f"Fetch report: {FETCH_REPORT_PATH}")
    print(
        f"Hoan tat: +{report['new_rows']} dong | "
        f"OK {len(report['success_symbols'])} | "
        f"Loi {len(report['failed_symbols'])} | "
        f"date_max={report['date_max_after']}"
    )
    if report["failed_symbols"]:
        raise RuntimeError(
            f"Fetch partial: {len(report['failed_symbols'])} symbol(s) failed; retry required."
        )


if __name__ == "__main__":
    main()
