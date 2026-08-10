import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import joblib

@contextmanager
def atomic_output_path(path: Path):
    """Yield a sibling temp path and replace the target only after success.

    Vì sao mọi output của pipeline đều đi qua kiểu ghi này: web app đọc cùng lúc
    pipeline ghi. Ghi trực tiếp vào file đích khiến UI/chatbot có thể đọc được
    một file ghi DỞ (CSV thiếu dòng cuối, JSON không đóng ngoặc). ``os.replace``
    là atomic ở tầng filesystem, nên người đọc chỉ thấy bản cũ hoàn chỉnh hoặc
    bản mới hoàn chỉnh, không bao giờ thấy trạng thái trung gian.

    Temp file được tạo Ở CÙNG THƯ MỤC với đích (``dir=path.parent``) là bắt buộc:
    ``os.replace`` chỉ atomic khi nguồn và đích cùng một filesystem; đặt temp ở
    /tmp có thể rơi vào ổ khác và biến thành copy + delete (mất tính atomic).

    Nếu body ``with`` raise, ``os.replace`` không chạy → file đích giữ nguyên bản
    cũ, và ``finally`` dọn temp. ``missing_ok=True`` cần thiết vì khi thành công
    thì temp đã bị ``os.replace`` chuyển đi, không còn tồn tại.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=f".tmp{path.suffix}", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        yield temp_path
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_json(path: Path, payload: dict) -> None:
    atomic_write_text(
        path,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
    )


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Ghi text kiểu "hoặc bản cũ, hoặc bản mới", không bao giờ ra file nửa vời.

    Cách làm: tạo file tạm CÙNG thư mục đích rồi ``os.replace``. Cùng thư mục là
    bắt buộc — ``os.replace`` chỉ nguyên tử khi nguồn và đích nằm trên cùng ổ
    đĩa/volume; nếu để file tạm ở ``%TEMP%`` thì thao tác biến thành copy + delete
    và mất tính nguyên tử.

    ``mkstemp`` trả file descriptor mở nên phải ``os.close(fd)`` ngay, sau đó mới
    ghi bằng ``write_text``; trên Windows một handle còn mở sẽ khiến ``os.replace``
    fail với PermissionError.

    ``finally: unlink(missing_ok=True)`` dọn rác nếu ``write_text`` lỗi giữa
    đường; khi ``os.replace`` thành công thì file tạm đã đổi tên nên ``missing_ok``
    làm no-op.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(text, encoding=encoding)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_dataframe_to_csv(frame, path: Path, **kwargs) -> None:
    """Bản CSV của [[atomic_write_text]] — pandas ghi thẳng vào file tạm.

    Dùng cho mọi CSV báo cáo (model_comparison, confusion_matrix, ...) vì UI đọc
    các file này bất cứ lúc nào: nếu ``to_csv`` ghi trực tiếp lên đích mà pipeline
    bị kill giữa dòng, trang web sẽ đọc được CSV thiếu dòng cuối và vỡ parse.

    ``**kwargs`` chuyển nguyên cho ``frame.to_csv`` (thường là ``index=False``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        frame.to_csv(temp_path, **kwargs)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _json_safe(value):
    """Đổi NaN/Infinity thành None đệ quy trước khi dump JSON.

    ``json.dumps`` mặc định ghi ``NaN``/``Infinity`` — hợp lệ với Python nhưng KHÔNG
    hợp lệ theo chuẩn JSON, nên ``JSON.parse`` ở phía JS sẽ vỡ. Ta ghi với
    ``allow_nan=False`` (fail loud) và làm sạch trước bằng hàm này, thay vì để
    một metric NaN làm hỏng cả file report.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def atomic_joblib_dump(value, path: Path) -> None:
    """Write one joblib artifact beside its target, then replace atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        joblib.dump(value, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_model_release(
    artifact: dict,
    metadata: dict,
    *,
    model_path: Path,
    metadata_path: Path,
) -> None:
    """Promote model and metadata together; restore both if either replace fails.

    Bài toán khó ở đây: HAI file phải cùng release. ``os.replace`` chỉ atomic cho
    MỘT file, nên nếu ghi lần lượt và crash ở giữa, ta sẽ có model mới + metadata
    cũ — đúng trường hợp mà chatbot/prediction gọi là ``inconsistent``.

    Cách xử lý (transaction thủ công, 3 giai đoạn):
    1. **Stage**: ghi cả hai bản mới ra file tạm cạnh đích. Lỗi ở đây chưa ảnh
       hưởng gì tới release đang chạy.
    2. **Backup**: ``shutil.copy2`` bản đang publish (nếu có) sang ``.backup``.
       Dùng copy chứ không move để giữa hai bước file đích vẫn tồn tại cho reader.
    3. **Commit**: hai ``os.replace`` liên tiếp. Nếu cái thứ hai lỗi, khối
       ``except`` KHÔI PHỤC cả hai từ backup (hoặc xóa hẳn nếu trước đó chưa có
       file) rồi re-raise → thà không có release còn hơn release lệch nhau.

    Tên file tạm/backup nhúng ``os.getpid()`` để hai process cùng publish không
    giẫm lên file tạm của nhau. ``finally`` dọn sạch mọi tạm/backup ở mọi nhánh.

    Cửa sổ còn lại (không xóa được bằng POSIX API): giữa hai ``os.replace``, reader
    có thể bắt đúng khoảnh khắc model mới + metadata cũ. Vì vậy tầng đọc vẫn phải
    tự kiểm identity 3 nguồn thay vì tin tưởng hàm này là hoàn hảo.
    """
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    staged_model = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
    staged_metadata = metadata_path.with_name(
        f".{metadata_path.name}.{os.getpid()}.tmp"
    )
    backups: dict[Path, Path] = {}
    try:
        joblib.dump(artifact, staged_model)
        staged_metadata.write_text(
            json.dumps(
                _json_safe(metadata),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        for target in (model_path, metadata_path):
            if not target.exists():
                continue
            backup = target.with_name(f".{target.name}.{os.getpid()}.backup")
            shutil.copy2(target, backup)
            backups[target] = backup

        try:
            os.replace(staged_model, model_path)
            os.replace(staged_metadata, metadata_path)
        except Exception:
            for target in (model_path, metadata_path):
                backup = backups.get(target)
                if backup and backup.exists():
                    os.replace(backup, target)
                elif target.exists():
                    target.unlink()
            raise
    finally:
        staged_model.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
