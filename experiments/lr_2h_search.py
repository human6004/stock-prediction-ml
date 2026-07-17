import sys, time, json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from services.tuning_lab import load_train, evaluate_single_config
from config.settings import TUNING_HISTORY_PATH

BUDGET_S = 7200  # 2 hours
train = load_train()


def norm(p):
    # dedup key: C to 6 sig-figs + solver
    return (float(f"{float(p['C']):.6g}"), str(p["solver"]))


def load_seen():
    s = set()
    with open(TUNING_HISTORY_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model_key"] == "logistic_regression" and r["status"] == "ok" and r["cv_f1_up_mean"]:
                try:
                    s.add(norm(json.loads(r["params_json"])))
                except Exception:
                    pass
    return s


seen = load_seen()
print(f"already-tested LR combos: {len(seen)}", flush=True)

# only tunable levers are C and solver -> dense log grid on C x both solvers
c_grid = np.unique(np.round(np.logspace(-8, 4, 800), 12))
configs = [{"C": float(c), "solver": sv}
           for c in c_grid for sv in ("lbfgs", "liblinear")]
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
        r = evaluate_single_config("logistic_regression", p, train,
                                   note="auto LR search 2h (threshold-tuned)")
        seen.add(norm(p))
        f1 = r["cv_f1_up_mean"]
        if f1 > best[0]:
            best = (f1, r["run_id"], p)
        done += 1
        print(f"[{done}] {r['run_id']} f1_up={f1:.4f} | C={p['C']:.3e} "
              f"solver={p['solver']} | elapsed={time.perf_counter()-start:.0f}s", flush=True)
    except Exception as exc:
        print(f"[err] {p}: {exc}", flush=True)

print(f"\nDONE: {done} configs in {time.perf_counter()-start:.0f}s", flush=True)
print(f"BEST(this run): f1_up={best[0]:.4f} run_id={best[1]} params={best[2]}", flush=True)
