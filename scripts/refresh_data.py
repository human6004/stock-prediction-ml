"""Background job: fetch raw OHLCV, preprocess, and rebuild ml_dataset.csv.

Script này được ``app.py`` chạy dưới dạng tiến trình con nền (``Popen``), log ghi
vào ``experiments/last_fetch_run.log``. Hai chi tiết KHÔNG được đổi vì UI phụ thuộc:

1. Các marker ``[1/3]``, ``[2/3]``, ``[3/3]`` và dòng cuối ``Refresh data
   completed.`` chính là thứ ``app._infer_refresh_progress`` bắt để suy ra bước
   hiện tại và phần trăm tiến độ. Đổi chữ = mất thanh tiến độ.
2. ``import`` được đặt BÊN TRONG ``main()`` theo từng bước, không phải ở đầu file.
   Lý do: mỗi bước import module nặng (vnstock, pandas pipeline) và đọc file
   config; nạp lười giúp một bước lỗi vẫn in được marker của bước trước đó, và
   tránh trả giá import cho bước chưa tới.

Không bọc try/except: lỗi phải thoát với exit code khác 0 để lock/log phản ánh
đúng là job thất bại, thay vì âm thầm "completed".
"""

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
    """Chạy tuần tự 3 bước làm mới dữ liệu; mỗi bước in marker cho UI đọc tiến độ.

    Ba bước phụ thuộc nhau theo dây: fetch ghi ``hose_stock_raw.csv`` → preprocess
    đọc file đó để ghi bản sạch → build_features đọc bản sạch để ghi
    ``ml_dataset.csv``. Vì vậy không thể song song hoá, và một bước lỗi thì các
    bước sau vô nghĩa (không try/except: exception thoát ngay, exit code khác 0).

    Job này KHÔNG train model. Sau khi xong, ``ml_dataset.csv`` mới sẽ có
    fingerprint khác → cấu hình Tuning Lab cũ hết hiệu lực và pipeline sẽ đòi
    chọn lại tham số cho snapshot mới.
    """
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
