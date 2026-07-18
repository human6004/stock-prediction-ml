-- Schema SQLite khởi tạo cho dữ liệu, báo cáo model và lịch sử dự báo.
-- Lưu ý: database_service dùng pandas to_sql(if_exists='replace') cho ba bảng
-- snapshot bên dưới, nên schema runtime của chúng có thể không giữ constraint này.

-- Snapshot dữ liệu: được dựng lại từ CSV sau mỗi official pipeline.
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

-- Lịch sử train/evaluate: mỗi official run append thêm dòng.
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

-- Lịch sử prediction từ web/CLI: mỗi lần dự báo INSERT một dòng.
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    reference_date TEXT,
    prediction TEXT,
    probability_up REAL,
    model_name TEXT,
    created_at TEXT
);
