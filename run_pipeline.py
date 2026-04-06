# === run_pipeline.py ===

import os
import sqlite3
import yaml
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path


# --------------------------
# YAML Loader
# --------------------------
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------
# Logger Setup
# --------------------------
def setup_logger(config):
    os.makedirs(os.path.dirname(config["logging"]["file"]), exist_ok=True)
    logging.basicConfig(
        filename=config["logging"]["file"],
        level=getattr(logging, config["logging"]["level"]),
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8"
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # also print to console


# --------------------------
# SQLite DB Setup
# --------------------------
def init_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables if not exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            feature1 REAL,
            feature2 REAL,
            feature3 REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_id INTEGER,
            label TEXT,
            timestamp TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_id INTEGER,
            anomaly_score REAL,
            detector_name TEXT,
            timestamp TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            type TEXT,
            path TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    logging.info(f"✅ Database initialized at '{db_path}'")
    return conn


# --------------------------
# Ingestion
# --------------------------
def run_ingestion(config, conn):
    logging.info("[Ingestion] Starting data ingestion...")

    data_source = config["pipeline"]["data_source"]
    try:
        df = pd.read_csv(data_source)
    except Exception as e:
        logging.error(f"[Ingestion] Failed to read CSV: {e}")
        return

    if df.empty:
        logging.warning("[Ingestion] No data found in the source CSV.")
        return

    # Dynamically get columns from CSV
    columns = df.columns.tolist()
    logging.info(f"[Ingestion] Detected columns: {columns}")

    # Drop old raw_data table if schema mismatched
    conn.execute("DROP TABLE IF EXISTS raw_data")

    # Create new table based on CSV header
    cols_def = ", ".join([f"{col} TEXT" for col in columns])
    conn.execute(f"CREATE TABLE raw_data (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols_def})")

    # Insert all rows dynamically
    placeholders = ", ".join(["?"] * len(columns))
    insert_query = f"INSERT INTO raw_data ({', '.join(columns)}) VALUES ({placeholders})"
    conn.executemany(insert_query, df.values.tolist())
    conn.commit()

    logging.info(f"[Ingestion] Loaded {len(df)} rows from {data_source}")
    logging.info("[Ingestion] Completed successfully.")


# --------------------------
# Preprocessing
# --------------------------
def run_preprocessing(config, conn):
    logging.info("[Preprocessing] Cleaning and transforming data...")
    # Example: here we could normalize or filter data if needed
    logging.info("[Preprocessing] Completed successfully.")


# --------------------------
# Anomaly Detection
# --------------------------
def run_detection(config, conn):
    print("[Detection] Running anomaly detection...")
    try:
        df = pd.read_sql_query("SELECT * FROM raw_data", conn)
    except Exception as e:
        print(f"⚠️ [Detection] Could not load raw_data: {e}")
        return

    # --- Ensure required columns exist (failsafe against crashes) ---
    required_cols = ["src_bytes", "dst_bytes"]
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ [Detection] Missing expected column '{col}'. Skipping detection to prevent errors.")
            return

    # --- Convert data types to numeric ---
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # convert to float, set invalid as NaN
    df = df.dropna(subset=required_cols)  # remove rows with NaN values

    # --- Simple heuristic anomaly scoring ---
    for _, row in df.iterrows():
        try:
            score = (row["src_bytes"] + row["dst_bytes"]) / 2  # now both numeric

            # FIXED: Column names now perfectly match the database schema
            conn.execute(
                "INSERT INTO anomalies (row_id, detector_name, anomaly_score, timestamp) VALUES (?, ?, ?, ?)",
                (row["id"], "heuristic_detector", score, datetime.now().isoformat()),
            )
        except Exception as e:
             # Failsafe: If one row fails, don't crash the whole script
             print(f"⚠️ [Detection] Error processing row {row.get('id', 'unknown')}: {e}")

    conn.commit()
    print("[Detection] Completed successfully.")


# --------------------------
# Register Detector
# --------------------------
def register_detector(conn, name, type_, path):
    try:
        conn.execute(
            "INSERT INTO detectors (name, type, path, created_at) VALUES (?, ?, ?, ?)",
            (name, type_, path, datetime.now().isoformat())
        )
        conn.commit()
        logging.info(f"[Detector] Registered '{name}' of type '{type_}'")
    except Exception as e:
        logging.error(f"[Detector] Failed to register detector: {e}")


# --------------------------
# Pipeline Runner
# --------------------------
def run_pipeline():
    try:
        config = load_config()
    except Exception as e:
        print(f"❌ Failed to load config.yaml: {e}")
        return

    setup_logger(config)
    logging.info("=== Pipeline Started ===")
    logging.info(f"Mode: {config['pipeline'].get('mode', 'unknown')}")
    logging.info(f"Database: {os.path.join(config['database']['folder'], config['database']['name'])}")
    logging.info(f"Data Source: {config['pipeline']['data_source']}")

    try:
        conn = init_db(os.path.join(config["database"]["folder"], config["database"]["name"]))
    except Exception as e:
        logging.critical(f"❌ Failed to initialize database: {e}")
        return

    # Pipeline steps
    run_ingestion(config, conn)
    run_preprocessing(config, conn)
    run_detection(config, conn)
    register_detector(conn, "SimpleDetector", "Threshold", "models/simple_detector.joblib")

    logging.info("=== Pipeline Ended ===")
    conn.close()


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    run_pipeline()