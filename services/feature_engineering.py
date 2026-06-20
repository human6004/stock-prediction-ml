import numpy as np
import pandas as pd

from config.settings import (
    FEATURE_COLUMNS,
    FEATURE_DATA_PATH,
    ML_DATA_PATH,
    PREDICTION_HORIZON,
    SPLIT_DATE,
    TRAIN_TEST_SUMMARY_PATH,
    UP_THRESHOLD,
)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(loss != 0, 100)
    rsi = rsi.where(~((gain == 0) & (loss == 0)), 50)
    return rsi


def build_features(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frames = []
    for _, group in clean_df.groupby("symbol", sort=False):
        g = group.sort_values("trading_date").copy()
        close = g["close"]
        volume = g["volume"]

        g["return_1d"] = close.pct_change(1)
        g["return_3d"] = close.pct_change(3)
        g["return_5d"] = close.pct_change(5)
        g["close_open_return"] = (g["close"] / g["open"]) - 1
        g["sma5"] = close.rolling(window=5, min_periods=5).mean()
        g["sma20"] = close.rolling(window=20, min_periods=20).mean()
        g["sma50"] = close.rolling(window=50, min_periods=50).mean()
        g["close_vs_sma20"] = (g["close"] / g["sma20"]) - 1
        g["sma20_vs_sma50"] = (g["sma20"] / g["sma50"]) - 1
        g["rsi14"] = compute_rsi(close, 14)
        g["volatility_5d"] = g["return_1d"].rolling(window=5, min_periods=5).std()
        g["volatility_20d"] = g["return_1d"].rolling(window=20, min_periods=20).std()
        g["price_range"] = (g["high"] - g["low"]) / g["close"]
        g["volume_change_1d"] = volume.pct_change(1)
        volume_ma20 = volume.rolling(window=20, min_periods=20).mean()
        g["volume_ratio_20"] = volume / volume_ma20
        frames.append(g)

    features = pd.concat(frames, ignore_index=True)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(subset=FEATURE_COLUMNS).copy()
    features = features.sort_values(["symbol", "trading_date"]).reset_index(drop=True)
    report = {
        "rows_after_features": int(len(features)),
        "symbols_after_features": int(features["symbol"].nunique()),
        "feature_count": len(FEATURE_COLUMNS),
    }
    return features, report


def _label_end_date_for_row(group: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(group["trading_date"])
    indexed = pd.Series(dates.values, index=group.index)
    shifted = indexed.shift(-PREDICTION_HORIZON)
    return shifted.dt.strftime("%Y-%m-%d")


def create_labels(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frames = []
    for _, group in features.groupby("symbol", sort=False):
        g = group.sort_values("trading_date").copy()
        g["future_close_5d"] = g["close"].shift(-PREDICTION_HORIZON)
        g["future_return_5d"] = (g["future_close_5d"] / g["close"]) - 1
        g["label_end_date"] = _label_end_date_for_row(g)
        frames.append(g)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.dropna(subset=["future_return_5d", "label_end_date"]).copy()
    dataset["target"] = (dataset["future_return_5d"] > UP_THRESHOLD).astype(int)
    dataset["target_label"] = np.where(dataset["target"] == 1, "UP", "NOT_UP")
    dataset = dataset.sort_values(["trading_date", "symbol"]).reset_index(drop=True)

    up_count = int((dataset["target"] == 1).sum())
    not_up_count = int((dataset["target"] == 0).sum())
    report = {
        "horizon": PREDICTION_HORIZON,
        "threshold": UP_THRESHOLD,
        "rows_after_labeling": int(len(dataset)),
        "up_count": up_count,
        "not_up_count": not_up_count,
        "up_ratio": round(up_count / len(dataset), 6),
        "not_up_ratio": round(not_up_count / len(dataset), 6),
    }
    return dataset, report


def time_based_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split by label_end_date to avoid leakage from future labels in train."""
    split_date = SPLIT_DATE
    train = dataset[dataset["label_end_date"] <= split_date].copy()
    test = dataset[dataset["label_end_date"] > split_date].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Time-based split produced empty set (train={len(train)}, test={len(test)}, "
            f"split_date={split_date})."
        )

    overlap = set(train.index).intersection(set(test.index))
    report = {
        "split_method": "label_end_date",
        "split_date": split_date,
        "shuffle": False,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_date_min": str(train["trading_date"].min()),
        "train_date_max": str(train["trading_date"].max()),
        "train_label_end_max": str(train["label_end_date"].max()),
        "test_date_min": str(test["trading_date"].min()),
        "test_date_max": str(test["trading_date"].max()),
        "test_label_end_min": str(test["label_end_date"].min()),
        "overlap_rows": int(len(overlap)),
    }
    return train, test, report


def write_train_test_summary(train: pd.DataFrame, test: pd.DataFrame) -> None:
    rows = [
        {
            "split": "train",
            "rows": len(train),
            "symbols": train["symbol"].nunique(),
            "date_min": train["trading_date"].min(),
            "date_max": train["trading_date"].max(),
            "label_end_max": train["label_end_date"].max(),
            "up_count": int((train["target"] == 1).sum()),
            "not_up_count": int((train["target"] == 0).sum()),
        },
        {
            "split": "test",
            "rows": len(test),
            "symbols": test["symbol"].nunique(),
            "date_min": test["trading_date"].min(),
            "date_max": test["trading_date"].max(),
            "label_end_min": test["label_end_date"].min(),
            "up_count": int((test["target"] == 1).sum()),
            "not_up_count": int((test["target"] == 0).sum()),
        },
    ]
    TRAIN_TEST_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TRAIN_TEST_SUMMARY_PATH, index=False)


def write_feature_outputs(features: pd.DataFrame, dataset: pd.DataFrame) -> None:
    FEATURE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_DATA_PATH, index=False)
    dataset.to_csv(ML_DATA_PATH, index=False)


def verify_data_pipeline(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if train["label_end_date"].max() > SPLIT_DATE:
        raise ValueError(
            f"Train max label_end_date {train['label_end_date'].max()} > SPLIT_DATE {SPLIT_DATE}"
        )
    if test["label_end_date"].min() <= SPLIT_DATE:
        raise ValueError(
            f"Test min label_end_date {test['label_end_date'].min()} <= SPLIT_DATE {SPLIT_DATE}"
        )
    leaked = {"future_close_5d", "future_return_5d", "target", "label_end_date"} & set(
        FEATURE_COLUMNS
    )
    if leaked:
        raise ValueError(f"Label columns leaked into FEATURE_COLUMNS: {leaked}")
