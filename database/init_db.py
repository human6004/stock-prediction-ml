import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from database.db_connection import init_database  # noqa: E402
from services.database_service import sync_all  # noqa: E402


def main() -> None:
    init_database()
    report = sync_all()
    print("Database initialized and synced.")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
