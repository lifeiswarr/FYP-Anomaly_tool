"""
SQLite Database Manager for the ML-Powered Anomaly Detection System
"""

from __future__ import annotations
import sqlite3
import json
import threading
import time
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ============================================================
#                  Configuration Loader Integration
# ============================================================
from utils.config_loader import load_config

try:
    config = load_config()
    db_folder = config["database"]["folder"]
    db_name = config["database"]["name"]
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", db_folder, db_name)
except Exception as e:
    print(f"[WARN] Could not load config.yaml: {e}")
    print("[INFO] Falling back to default database path in ./storage/")
    db_folder = "storage"
    db_name = "project_adapt.db"
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", db_folder, db_name)

# ensure the folder exists
DB_PATH = os.path.normpath(DB_PATH)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
print(f"Using database path: {DB_PATH}")

# ============================================================
#                        Logger Setup
# ============================================================
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ============================================================
#                        SQLite Optimizations
# ============================================================
_DEFAULT_PRAGMAS = (
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("temp_store", "MEMORY"),
    ("foreign_keys", "ON"),
    ("mmap_size", "268435456"),  # 256 MB
)

# ============================================================
#                        Database Schema
# ============================================================
_SCHEMA_SQL = r"""
BEGIN;

CREATE TABLE IF NOT EXISTS raw_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    sensor_id TEXT NOT NULL,
    value REAL,
    payload TEXT,
    created_at INTEGER DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id INTEGER,
    label TEXT NOT NULL,
    labeler TEXT,
    comment TEXT,
    created_at INTEGER DEFAULT (strftime('%s','now')),
    FOREIGN KEY(raw_id) REFERENCES raw_data(id) ON DELETE SET NULL
);

-- ✅ UPDATED anomalies table to match pipeline structure
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id INTEGER,
    detector_name TEXT,
    score REAL,
    detected_at TEXT
);

CREATE TABLE IF NOT EXISTS detectors (
    name TEXT PRIMARY KEY,
    version TEXT,
    description TEXT,
    created_at INTEGER DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_sensor ON raw_data(sensor_id);
CREATE INDEX IF NOT EXISTS idx_labels_raw ON labels(raw_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_raw ON anomalies(record_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_score ON anomalies(score);

COMMIT;
"""

# ============================================================
#                     Database Manager Class
# ============================================================
class DatabaseManager:
    """Thread-safe SQLite manager for time-series and anomaly storage."""

    DB_NAME = db_name
    DB_PATH = DB_PATH

    def __init__(self, path: str = DB_PATH, pragmas: Optional[Iterable[Tuple[str, str]]] = None):
        self.path = path
        self._pragmas = list(pragmas or _DEFAULT_PRAGMAS)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._configure()
        self._ensure_schema()
        logger.info(f"DatabaseManager initialized at {self.path}")

    def _configure(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            for k, v in self._pragmas:
                try:
                    cur.execute(f"PRAGMA {k}={v};")
                except sqlite3.DatabaseError:
                    logger.debug(f"PRAGMA {k}={v} not supported")
            cur.execute("PRAGMA locking_mode=EXCLUSIVE;")
            cur.close()
            self._conn.commit()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()

    # Utility methods
    def _now(self) -> int:
        return int(time.time())

    def _to_json(self, obj: Optional[Dict[str, Any]]) -> Optional[str]:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False) if obj else None

    def _from_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for c in ("payload", "metadata"):
            if c in d and d[c]:
                try:
                    d[c] = json.loads(d[c])
                except Exception:
                    pass
        return d

    # ============================================================
    # Data Insertion
    # ============================================================
    def insert_raw(self, timestamp: int, sensor_id: str, value: Optional[float] = None,
                   payload: Optional[Dict[str, Any]] = None) -> int:
        payload_txt = self._to_json(payload)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO raw_data (timestamp, sensor_id, value, payload) VALUES (?, ?, ?, ?)",
                (timestamp, sensor_id, value, payload_txt)
            )
            rid = cur.lastrowid
            self._conn.commit()
        logger.debug(f"Inserted raw_data id={rid}")
        return rid

    def insert_label(self, raw_id: int, label: str, labeler: Optional[str] = None, comment: Optional[str] = None) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("INSERT OR REPLACE INTO labels (raw_id, label, labeler, comment) VALUES (?, ?, ?, ?)",
                        (raw_id, label, labeler, comment))
            lid = cur.lastrowid
            self._conn.commit()
        logger.debug(f"Inserted label id={lid} for raw_id={raw_id}")
        return lid

    def insert_anomaly(self, record_id: int, detector_name: str, score: float, detected_at: Optional[str] = None) -> int:
        detected_at = detected_at or datetime.now().isoformat()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("INSERT INTO anomalies (record_id, detector_name, score, detected_at) VALUES (?, ?, ?, ?)",
                        (record_id, detector_name, score, detected_at))
            aid = cur.lastrowid
            self._conn.commit()
        logger.debug(f"Inserted anomaly id={aid}")
        return aid

    # ============================================================
    # Maintenance
    # ============================================================
    def vacuum(self) -> None:
        with self._lock:
            self._conn.execute("VACUUM")
            self._conn.commit()
        logger.info("Performed VACUUM on database")

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            finally:
                self._conn.close()
        logger.debug("Closed DB connection")

# ============================================================
#                      Entry Point
# ============================================================
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(f"Using database path: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print("Creating new empty database...")
        db = DatabaseManager(path=DB_PATH)
        db.close()
        print("✅ Empty database created.")
    else:
        print("Database already exists. Ready to use.")
