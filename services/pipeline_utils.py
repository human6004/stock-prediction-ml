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
    """Yield a sibling temp path and replace the target only after success."""
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
    """Promote model and metadata together; restore both if either replace fails."""
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
