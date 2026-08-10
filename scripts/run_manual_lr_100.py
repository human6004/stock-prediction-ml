"""Run exactly 100 manual Logistic Regression configs on current TRAIN."""

import json
from pathlib import Path

from services.experiment_state import compute_dataset_fingerprint, read_history
from services.tuning_lab import evaluate_single_config, load_train


CS = [
    1e-8, 1.5e-8, 2.2e-8, 3.2e-8, 4.6e-8, 6.8e-8,
    1e-7, 1.5e-7, 2.2e-7, 3.2e-7, 4.6e-7, 6.8e-7,
    1e-6, 1.5e-6, 2.2e-6, 3.2e-6, 4.6e-6, 6.8e-6,
    1e-5, 1.2e-5, 1.4e-5, 1.6e-5, 1.8e-5, 2e-5,
    2.2e-5, 2.4e-5, 2.6e-5, 2.8e-5, 3e-5, 3.2e-5,
    3.4e-5, 3.6e-5, 3.8e-5, 4e-5, 4.5e-5, 5e-5,
    6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 1.2e-4,
    1.5e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3, 5e-3,
]
SOLVERS = ("liblinear", "lbfgs")
RUN_TAG = "manual-lr-100-20260731"


def configs():
    return [
        {"C": c, "solver": solver}
        for c in CS
        for solver in SOLVERS
    ]


def main() -> None:
    params_list = configs()
    assert len(CS) == 50
    assert len(params_list) == 100

    train = load_train()
    fingerprint = compute_dataset_fingerprint()
    prior = read_history()
    done = {
        (row.get("dataset_fingerprint"), row.get("params_json"))
        for row in prior
        if row.get("model_key") == "logistic_regression"
        and row.get("note", "").startswith(RUN_TAG)
    }
    results = []
    for index, params in enumerate(params_list, start=1):
        params_json = json.dumps(params, ensure_ascii=False)
        if (fingerprint["hash"], params_json) in done:
            print(f"[{index:03d}/100] skipped existing {params}", flush=True)
            continue
        note = f"{RUN_TAG} index={index:03d} manual-C-grid"
        result = evaluate_single_config(
            "logistic_regression",
            params,
            train,
            note=note,
            fingerprint=fingerprint,
        )
        results.append(result)
        print(
            f"[{index:03d}/100] done C={params['C']:.10g} solver={params['solver']} "
            f"cv_f1={result['cv_f1_up_mean']:.6f} "
            f"threshold={result['decision_threshold']:.2f} "
            f"seconds={result['train_seconds']:.3f}",
            flush=True,
        )

    eligible = [r for r in results if r.get("threshold_constraint_passed")]
    if not eligible:
        raise RuntimeError("100 config hoàn tất nhưng không có config hợp lệ.")
    best = max(eligible, key=lambda r: (r["cv_f1_up_mean"], -r["cv_f1_up_std"]))
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    print(f"HISTORY_TAG={RUN_TAG}", flush=True)
    print(f"FINGERPRINT={fingerprint['hash']}", flush=True)


if __name__ == "__main__":
    main()
