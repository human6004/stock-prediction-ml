import argparse
import sys, time, json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from services.tuning_lab import load_train, evaluate_single_config
from config.settings import TUNING_HISTORY_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--budget-seconds", type=int, default=7200)
args = parser.parse_args()

BUDGET_S = args.budget_seconds
train = load_train()


def norm(p):
    return (int(p["n_estimators"]), round(float(p["learning_rate"]), 4),
            int(p["max_depth"]), round(float(p["subsample"]), 3))


def load_seen():
    s = set()
    with open(TUNING_HISTORY_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model_key"] == "gradient_boosting" and r["status"] == "ok" and r["cv_f1_up_mean"]:
                try:
                    s.add(norm(json.loads(r["params_json"])))
                except Exception:
                    pass
    return s


seen = load_seen()
print(f"already-tested GB combos: {len(seen)}", flush=True)

grid = {
    "n_estimators": [40, 60, 80, 100, 120, 150],
    "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    "max_depth": [2, 3, 4, 5],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}
configs = [{"n_estimators": int(n), "learning_rate": float(lr), "max_depth": int(d), "subsample": float(s)}
           for n in grid["n_estimators"] for lr in grid["learning_rate"]
           for d in grid["max_depth"] for s in grid["subsample"]]
rng = np.random.default_rng(2026)
rng.shuffle(configs)
configs = [p for p in configs if norm(p) not in seen]
print(f"remaining untested combos to try: {len(configs)}", flush=True)

start = time.perf_counter()
done = 0
best = (-1.0, None, None)
for p in configs:
    if time.perf_counter() - start >= BUDGET_S:
        print(f"[stop] budget {BUDGET_S}s reached after {done} configs", flush=True)
        break
    if norm(p) in seen:
        continue
    try:
        r = evaluate_single_config("gradient_boosting", p, train,
                                   note="auto GB search 3h ext (threshold-tuned)")
        seen.add(norm(p))
        f1 = r["cv_f1_up_mean"]
        if f1 > best[0]:
            best = (f1, r["run_id"], p)
        done += 1
        print(f"[{done}] {r['run_id']} f1_up={f1:.4f} | n={p['n_estimators']} "
              f"lr={p['learning_rate']} depth={p['max_depth']} sub={p['subsample']} "
              f"| elapsed={time.perf_counter()-start:.0f}s", flush=True)
    except Exception as exc:
        print(f"[err] {p}: {exc}", flush=True)

print(f"\nDONE: {done} configs in {time.perf_counter()-start:.0f}s", flush=True)
print(f"BEST(this run): f1_up={best[0]:.4f} run_id={best[1]} params={best[2]}", flush=True)
