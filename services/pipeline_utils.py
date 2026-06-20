import json
import shutil
from pathlib import Path

from config.settings import (
    BASE_DIR,
    BEST_PARAMS_PATH,
    FETCH_REPORT_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    PRESERVED_REPORT_JSON,
    PROCESSED_DIR,
    REPORTS_DIR,
    ROADMAP_PATH,
)

OUTPUT_DIRS = [PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]


def ensure_output_dirs() -> None:
    for directory in OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def cleanup_outputs() -> dict:
    """Remove generated artifacts; preserve fetch_report, best_params, model_metadata."""
    ensure_output_dirs()
    removed = []
    missing = []

    cleanup_targets = [
        PROCESSED_DIR,
        BASE_DIR / "data" / "cleaned",
        BASE_DIR / "data" / "features",
    ]
    cleanup_patterns = [
        (MODELS_DIR, "*.pkl"),
        (MODELS_DIR, "*.joblib"),
        (REPORTS_DIR, "*.csv"),
        (REPORTS_DIR, "*.png"),
        (REPORTS_DIR, "*.txt"),
        (REPORTS_DIR, "*.md"),
    ]

    for target in cleanup_targets:
        if target.exists():
            for item in target.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                removed.append(str(item.relative_to(BASE_DIR)))
        else:
            missing.append(str(target.relative_to(BASE_DIR)))

    for directory, pattern in cleanup_patterns:
        if directory.exists():
            for item in directory.glob(pattern):
                if item.is_file():
                    item.unlink()
                    removed.append(str(item.relative_to(BASE_DIR)))
        else:
            missing.append(str(directory.relative_to(BASE_DIR)))

    if REPORTS_DIR.exists():
        for item in REPORTS_DIR.glob("*.json"):
            if item.name in PRESERVED_REPORT_JSON:
                continue
            if item.is_file():
                item.unlink()
                removed.append(str(item.relative_to(BASE_DIR)))

    for cache_dir in BASE_DIR.rglob("__pycache__"):
        if ".venv" in cache_dir.parts or ".cursor" in cache_dir.parts:
            continue
        shutil.rmtree(cache_dir)
        removed.append(str(cache_dir.relative_to(BASE_DIR)))

    ensure_output_dirs()
    return {"removed": removed, "missing": sorted(set(missing))}


def read_roadmap() -> dict:
    roadmap = Path(ROADMAP_PATH)
    exists = roadmap.exists()
    text = ""
    if exists:
        text = roadmap.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(roadmap),
        "exists": exists,
        "read_success": exists,
        "chars": len(text),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
