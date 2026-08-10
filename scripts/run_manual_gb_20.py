"""Run exactly 20 manual Gradient Boosting configs on current TRAIN."""

import json

from services.experiment_state import compute_dataset_fingerprint, read_history
from services.tuning_lab import evaluate_single_config, load_train


CONFIGS = [
    {"n_estimators": 70, "learning_rate": 0.15, "max_depth": 2, "subsample": 0.5},
    {"n_estimators": 90, "learning_rate": 0.15, "max_depth": 2, "subsample": 0.5},
    {"n_estimators": 110, "learning_rate": 0.15, "max_depth": 2, "subsample": 0.5},
    {"n_estimators": 120, "learning_rate": 0.15, "max_depth": 2, "subsample": 0.5},
    {"n_estimators": 70, "learning_rate": 0.20, "max_depth": 2, "subsample": 0.5},
    {"n_estimators": 90, "learning_rate": 0.20, "max_depth": 2, "subsample": 0.5},
    {"n_estimators": 110, "learning_rate": 0.20, "max_depth": 2, "subsample": 0.5},
    {"n_estimators": 120, "learning_rate": 0.20, "max_depth": 2, "subsample": 0.5},
    {"n_estimators": 70, "learning_rate": 0.25, "max_depth": 2, "subsample": 0.5},
    {"n_estimators": 90, "learning_rate": 0.25, "max_depth": 2, "subsample": 0.5},
    {"n_estimators": 110, "learning_rate": 0.25, "max_depth": 2, "subsample": 0.5},
    {"n_estimators": 120, "learning_rate": 0.25, "max_depth": 2, "subsample": 0.5},
    {"n_estimators": 90, "learning_rate": 0.20, "max_depth": 2, "subsample": 0.6},
    {"n_estimators": 110, "learning_rate": 0.25, "max_depth": 2, "subsample": 0.6},
    {"n_estimators": 120, "learning_rate": 0.20, "max_depth": 2, "subsample": 0.7},
    {"n_estimators": 110, "learning_rate": 0.25, "max_depth": 2, "subsample": 0.7},
    {"n_estimators": 90, "learning_rate": 0.30, "max_depth": 2, "subsample": 0.8},
    {"n_estimators": 120, "learning_rate": 0.30, "max_depth": 2, "subsample": 0.8},
    {"n_estimators": 90, "learning_rate": 0.10, "max_depth": 3, "subsample": 0.5},
    {"n_estimators": 110, "learning_rate": 0.15, "max_depth": 3, "subsample": 0.7},
]
RUN_TAG = "manual-gb-20-20260731"


def main() -> None:
    assert len(CONFIGS) == 20
    train = load_train()
    fingerprint = compute_dataset_fingerprint()
    prior = read_history()
    done = {
        (row.get("dataset_fingerprint"), row.get("params_json"))
        for row in prior
        if row.get("model_key") == "gradient_boosting"
        and row.get("note", "").startswith(RUN_TAG)
    }
    results = []
    for index, params in enumerate(CONFIGS, start=1):
        params_json = json.dumps(params, ensure_ascii=False)
        if (fingerprint["hash"], params_json) in done:
            print(f"[{index:02d}/20] skipped existing {params}", flush=True)
            continue
        try:
            result = evaluate_single_config(
                "gradient_boosting",
                params,
                train,
                note=f"{RUN_TAG} index={index:02d} manual-grid",
                fingerprint=fingerprint,
            )
        except Exception as exc:
            print(f"[{index:02d}/20] error {params}: {exc}", flush=True)
            continue
        results.append(result)
        print(
            f"[{index:02d}/20] done params={params} "
            f"cv_f1={result['cv_f1_up_mean']:.6f} "
            f"threshold={result['decision_threshold']:.2f} "
            f"seconds={result['train_seconds']:.3f}",
            flush=True,
        )

    if not results:
        raise RuntimeError("Không có kết quả GB mới để chọn.")
    best = max(results, key=lambda r: (r["cv_f1_up_mean"], -r["cv_f1_up_std"]))
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    print(f"HISTORY_TAG={RUN_TAG}", flush=True)
    print(f"FINGERPRINT={fingerprint['hash']}", flush=True)


if __name__ == "__main__":
    main()
