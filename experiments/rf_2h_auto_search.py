import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import TUNING_HISTORY_PATH
from services.experiment_state import compute_dataset_fingerprint
from services.tuning_lab import evaluate_single_config, load_train


def norm(params: dict) -> tuple:
    max_features = params["max_features"]
    if isinstance(max_features, float):
        max_features = round(max_features, 4)
    return (
        int(params["n_estimators"]),
        int(params["max_depth"]),
        int(params["min_samples_leaf"]),
        max_features,
    )


def candidate_configs() -> list[dict]:
    base = {
        "n_estimators": 130,
        "max_depth": 6,
        "min_samples_leaf": 25,
        "max_features": 0.3,
    }
    baseline = {
        "n_estimators": 120,
        "max_depth": 10,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
    }

    configs = [base, baseline]
    configs += [{**base, "max_depth": d} for d in [3, 4, 5, 7, 8, 9, 10, 11, 12]]
    configs += [{**base, "min_samples_leaf": l} for l in [5, 10, 15, 20, 30, 35, 40, 50, 60, 80]]
    configs += [{**base, "max_features": f} for f in [0.2, 0.24, 0.25, 0.28, 0.32, 0.35, 0.4, 0.5, "sqrt", "log2"]]
    configs += [{**base, "n_estimators": n} for n in [50, 70, 90, 110, 150, 180, 220, 260]]

    for depth in [4, 5, 6, 7, 8]:
        for leaf in [15, 20, 25, 30, 40]:
            for features in [0.25, 0.3, 0.35, "sqrt"]:
                configs.append(
                    {
                        "n_estimators": 130,
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "max_features": features,
                    }
                )

    for depth in [3, 4, 5, 6, 7]:
        for leaf in [18, 22, 25, 28, 32]:
            for features in [0.33, 0.34, 0.36, 0.37]:
                configs.append(
                    {
                        "n_estimators": 130,
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "max_features": features,
                    }
                )

    seen = set()
    unique = []
    for params in configs:
        key = norm(params)
        if key not in seen:
            seen.add(key)
            unique.append(params)
    return unique


def load_seen(fingerprint: str) -> tuple[set[tuple], dict | None]:
    seen = set()
    best = None
    path = Path(TUNING_HISTORY_PATH)
    if not path.exists():
        return seen, best

    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("model_key") != "random_forest"
                or row.get("dataset_fingerprint") != fingerprint
                or row.get("status") != "ok"
                or not row.get("cv_f1_up_mean")
            ):
                continue
            try:
                params = json.loads(row["params_json"])
                score = float(row["cv_f1_up_mean"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            seen.add(norm(params))
            if best is None or score > best["cv_f1_up_mean"]:
                best = {
                    "run_id": row.get("run_id"),
                    "params": params,
                    "cv_f1_up_mean": score,
                    "source": "history",
                }
    return seen, best


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-seconds", type=int, default=7200)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fingerprint = compute_dataset_fingerprint()["hash"]
    run_id = datetime.now().strftime("RF_2H_%Y%m%d_%H%M%S")
    log_path = ROOT / "experiments" / f"random_forest_2h_auto_{run_id}.log"
    latest_path = ROOT / "experiments" / "random_forest_2h_auto_latest.json"
    configs = candidate_configs()
    seen, best = load_seen(fingerprint)
    pending = [params for params in configs if norm(params) not in seen]

    if args.dry_run:
        assert configs
        assert len(configs) == len({norm(params) for params in configs})
        print(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "candidate_count": len(configs),
                    "already_seen": len(seen),
                    "pending": len(pending),
                },
                ensure_ascii=False,
            )
        )
        return 0

    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.perf_counter()
    completed = 0
    errors = 0

    def status(state: str) -> dict:
        return {
            "status": state,
            "run_id": run_id,
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "budget_seconds": args.budget_seconds,
            "elapsed_seconds": round(time.perf_counter() - started, 1),
            "dataset_fingerprint": fingerprint,
            "candidate_count": len(configs),
            "completed_new_runs": completed,
            "reused_history_runs": len(seen),
            "pending_runs": max(0, len(pending) - completed),
            "errors": errors,
            "best": best,
            "log_path": str(log_path.relative_to(ROOT)),
        }

    train = load_train()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{started_at} | START {run_id} fingerprint={fingerprint} pending={len(pending)}\n")
        write_json(latest_path, status("running"))

        for params in pending:
            if args.max_runs is not None and completed >= args.max_runs:
                break
            if time.perf_counter() - started >= args.budget_seconds:
                break
            try:
                result = evaluate_single_config(
                    "random_forest",
                    params,
                    train,
                    note=f"auto RF 2h search {run_id}",
                )
                completed += 1
                score = result["cv_f1_up_mean"]
                if best is None or score > best["cv_f1_up_mean"]:
                    best = {
                        "run_id": result["run_id"],
                        "params": params,
                        "cv_f1_up_mean": score,
                        "cv_f1_up_std": result["cv_f1_up_std"],
                        "source": "new",
                    }
                    log.write(f"{datetime.now().isoformat(timespec='seconds')} | BEST f1={score:.9f} params={params}\n")
                log.write(
                    f"{datetime.now().isoformat(timespec='seconds')} | TRAIN f1={score:.9f} "
                    f"run_id={result['run_id']} params={params}\n"
                )
            except Exception as exc:  # noqa: BLE001 - keep long run alive after one bad config
                errors += 1
                log.write(f"{datetime.now().isoformat(timespec='seconds')} | ERROR params={params} error={exc}\n")
            log.flush()
            write_json(latest_path, status("running"))

        final_state = (
            "max_runs_reached"
            if args.max_runs is not None and completed >= args.max_runs
            else "budget_reached"
            if time.perf_counter() - started >= args.budget_seconds
            else "done"
        )
        payload = status(final_state)
        write_json(latest_path, payload)
        log.write(f"{datetime.now().isoformat(timespec='seconds')} | DONE {json.dumps(payload, ensure_ascii=False)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
