import sqlite3
from pathlib import Path

from config.settings import DATABASE_DIR, DATABASE_PATH

INIT_SQL_PATH = Path(__file__).resolve().parent / "init_db.sql"


def get_connection() -> sqlite3.Connection:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    sql = INIT_SQL_PATH.read_text(encoding="utf-8")
    with get_connection() as conn:
        conn.executescript(sql)
        conn.commit()
