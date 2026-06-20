"""Deprecated: tuning is integrated in train_tune_models.py / run_pipeline.py."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("finetune_model.py da duoc thay the boi train_tune_models.py.")
print("Hay chay: python scripts/train_tune_models.py")
print("Hoac pipeline day du: python scripts/run_pipeline.py")
