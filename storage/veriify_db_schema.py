"""
SQLite Schema Verification Utility
----------------------------------
Verifies the integrity of the anomaly detection database schema:
- Confirms table existence
- Checks foreign key constraints
- Lists indexes for performance validation
- Logs all results both to console and 'logs/schema_check.log'
- Prints a [DB-READY] message if all checks pass
"""

import sqlite3
import os
import logging
from utils.config_loader import load_config

# -------------------- Logging Setup --------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "schema_check.log")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),              # Console output
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")  # Persistent file log
    ]
)

def verify_schema(db_path: str):
    if not os.path.exists(db_path):
        logging.error(f"Database file not found: {db_path}")
        return False

    logging.info(f"Connecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1️⃣ List all tables
    cur.execute("PRAGMA table_list;")
    tables = [row[1] for row in cur.fetchall()]
    logging.info(f"Found tables: {tables}")

    expected_tables = {"raw_data", "labels", "anomalies", "detectors"}
    missing = expected_tables - set(tables)
    if missing:
        logging.error(f"Missing tables: {missing}")
        return False

    # 2️⃣ Check foreign keys (labels/anomalies → raw_data)
    logging.info("\nChecking foreign key constraints:")
    for table in ("labels", "anomalies"):
        cur.execute(f"PRAGMA foreign_key_list({table});")
        fks = cur.fetchall()
        for fk in fks:
            logging.info(f"  {table}: {fk[2]}({fk[3]} → {fk[4]})")
        if not fks:
            logging.warning(f"  No foreign keys found in {table}")

    # 3️⃣ Check indexes
    logging.info("\nVerifying indexes:")
    for table in ("raw_data", "anomalies"):
        cur.execute(f"PRAGMA index_list({table});")
        indexes = [r[1] for r in cur.fetchall()]
        logging.info(f"  {table}: {indexes}")

    conn.close()

    logging.info("\n✅ [DB-READY] Schema validated successfully.")
    return True


if __name__ == "__main__":
    try:
        config = load_config()
        db_folder = config["database"]["folder"]
        db_name = config["database"]["name"]
        db_path = os.path.join(os.path.dirname(__file__), "..", db_folder, db_name)
    except Exception as e:
        logging.warning(f"Could not load config.yaml: {e}")
        db_path = os.path.join(os.path.dirname(__file__), "project_adapt.db")

    db_path = os.path.abspath(db_path)
    verify_schema(db_path)
    logging.info(f"Logs saved at: {LOG_FILE}")
