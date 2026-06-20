CREATE TABLE IF NOT EXISTS raw_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    trading_date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    UNIQUE(symbol, trading_date)
);

CREATE TABLE IF NOT EXISTS clean_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    trading_date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    UNIQUE(symbol, trading_date)
);

CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    trading_date TEXT NOT NULL,
    close REAL,
    feature_json TEXT,
    UNIQUE(symbol, trading_date)
);

CREATE TABLE IF NOT EXISTS tuning_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    cv_f1_up REAL,
    best_params_json TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS model_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    accuracy REAL,
    f1_up REAL,
    recall_up REAL,
    selected INTEGER DEFAULT 0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    reference_date TEXT,
    prediction TEXT,
    probability_up REAL,
    model_name TEXT,
    created_at TEXT
);
