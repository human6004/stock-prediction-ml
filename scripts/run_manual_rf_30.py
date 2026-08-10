"""Run exactly 30 manual Random Forest configs on current TRAIN."""

import json

from services.experiment_state import compute_dataset_fingerprint, read_history
from services.tuning_lab import evaluate_single_config, load_train


CONFIGS = [
    {"n_estimators": 60, "max_depth": 5, "min_samples_leaf": 50, "max_features": 0.2},
    {"n_estimators": 90, "max_depth": 5, "min_samples_leaf": 75, "max_features": 0.2},
    {"n_estimators": 120, "max_depth": 5, "min_samples_leaf": 100, "max_features": 0.2},
    {"n_estimators": 150, "max_depth": 5, "min_samples_leaf": 125, "max_features": 0.2},
    {"n_estimators": 180, "max_depth": 5, "min_samples_leaf": 75, "max_features": 0.3},
    {"n_estimators": 60, "max_depth": 5, "min_samples_leaf": 100, "max_features": 0.3},
    {"n_estimators": 90, "max_depth": 5, "min_samples_leaf": 125, "max_features": 0.3},
    {"n_estimators": 120, "max_depth": 5, "min_samples_leaf": 50, "max_features": 0.3},
    {"n_estimators": 150, "max_depth": 5, "min_samples_leaf": 75, "max_features": 0.4},
    {"n_estimators": 180, "max_depth": 5, "min_samples_leaf": 100, "max_features": 0.4},
    {"n_estimators": 60, "max_depth": 6, "min_samples_leaf": 50, "max_features": 0.2},
    {"n_estimators": 90, "max_depth": 6, "min_samples_leaf": 75, "max_features": 0.2},
    {"n_estimators": 120, "max_depth": 6, "min_samples_leaf": 100, "max_features": 0.2},
    {"n_estimators": 150, "max_depth": 6, "min_samples_leaf": 125, "max_features": 0.2},
    {"n_estimators": 180, "max_depth": 6, "min_samples_leaf": 75, "max_features": 0.3},
    {"n_estimators": 60, "max_depth": 6, "min_samples_leaf": 100, "max_features": 0.3},
    {"n_estimators": 90, "max_depth": 6, "min_samples_leaf": 125, "max_features": 0.3},
    {"n_estimators": 120, "max_depth": 6, "min_samples_leaf": 50, "max_features": 0.3},
    {"n_estimators": 150, "max_depth": 6, "min_samples_leaf": 75, "max_features": 0.4},
    {"n_estimators": 180, "max_depth": 6, "min_samples_leaf": 100, "max_features": 0.4},
    {"n_estimators": 60, "max_depth": 7, "min_samples_leaf": 50, "max_features": 0.2},
    {"n_estimators": 90, "max_depth": 7, "min_samples_leaf": 75, "max_features": 0.2},
    {"n_estimators": 120, "max_depth": 7, "min_samples_leaf": 100, "max_features": 0.2},
    {"n_estimators": 150, "max_depth": 7, "min_samples_leaf": 125, "max_features": 0.2},
    {"n_estimators": 180, "max_depth": 7, "min_samples_leaf": 75, "max_features": 0.3},
    {"n_estimators": 60, "max_depth": 7, "min_samples_leaf": 100, "max_features": 0.3},
    {"n_estimators": 90, "max_depth": 7, "min_samples_leaf": 125, "max_features": 0.3},
    {"n_estimators": 120, "max_depth": 8, "min_samples_leaf": 50, "max_features": 0.3},
    {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 75, "max_features": 0.2},
    {"n_estimators": 180, "max_depth": 8, "min_samples_leaf": 100, "max_features": 0.2},
]
RUN_TAG = "manual-rf-30-20260731"


def main() -> None:
    assert len(CONFIGS) == 30
    train = load_train()
    fingerprint = compute_dataset_fingerprint()
    prior = read_history()
    done = {
        (row.get("dataset_fingerprint"), row.get("params_json"))
        for row in prior
        if row.get("model_key") == "random_forest"
        and row.get("note", "").startswith(RUN_TAG)
    }
    results = []
    for index, params in enumerate(CONFIGS, start=1):
        params_json = json.dumps(params, ensure_ascii=False)
        if (fingerprint["hash"], params_json) in done:
            print(f"[{index:02d}/30] skipped existing {params}", flush=True)
            continue
        try:
            result = evaluate_single_config(
                "random_forest",
                params,
                train,
                note=f"{RUN_TAG} index={index:02d} manual-grid",
                fingerprint=fingerprint,
            )
        except Exception as exc:
            print(f"[{index:02d}/30] error {params}: {exc}", flush=True)
            continue
        results.append(result)
        print(
            f"[{index:02d}/30] done params={params} "
            f"cv_f1={result['cv_f1_up_mean']:.6f} "
            f"threshold={result['decision_threshold']:.2f} "
            f"seconds={result['train_seconds']:.3f}",
            flush=True,
        )

    if not results:
        raise RuntimeError("Không có kết quả RF mới để chọn.")
    best = max(results, key=lambda r: (r["cv_f1_up_mean"], -r["cv_f1_up_std"]))
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    print(f"HISTORY_TAG={RUN_TAG}", flush=True)
    print(f"FINGERPRINT={fingerprint['hash']}", flush=True)


if __name__ == "__main__":
    main()
