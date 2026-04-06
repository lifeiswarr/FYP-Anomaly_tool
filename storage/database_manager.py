"""
SQLite Database Manager for A-DAPT (ML-Powered Anomaly Detection System)
Optimized for Network Flow Features (Groups A-I)
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

# Ensure the folder exists
DB_PATH = os.path.normpath(DB_PATH)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

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
#               Network-Aware Database Schema
# ============================================================
_SCHEMA_SQL = r"""
BEGIN;

-- Stores the core network features (Supports Groups A, B, C, D, E, H)
CREATE TABLE IF NOT EXISTS network_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Identification & IP Features (Group E)
    src_ip TEXT,
    dst_ip TEXT,
    src_port INTEGER,
    dst_port INTEGER,
    protocol TEXT,

    -- Volumetric Features (Group A)
    total_bytes INTEGER,
    total_packets INTEGER,
    avg_packet_size REAL,

    -- Temporal & Protocol Features (Group B & C)
    pkt_rate REAL,
    syn_count INTEGER,
    ack_count INTEGER,
    syn_ack_ratio REAL,
    
    -- Port/Service Features (Group D)
    port_entropy REAL,
    
    -- Catch-all for extra metadata (Group F, G, I)
    metadata_json TEXT 
);

-- Stores anomaly detection results
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id INTEGER,           -- Corrected: Now explicitly exists for your pipeline
    detector_name TEXT,
    score REAL,
    explanation TEXT,            -- Added: To store output from your Explainer module
    detected_at TEXT,
    FOREIGN KEY(record_id) REFERENCES network_features(id) ON DELETE CASCADE
);

-- Standard labels for supervised training
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_id INTEGER,
    label TEXT NOT NULL,         -- 'normal' or 'attack'
    comment TEXT,
    created_at INTEGER DEFAULT (strftime('%s','now')),
    FOREIGN KEY(feature_id) REFERENCES network_features(id) ON DELETE CASCADE
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_feat_ts ON network_features(timestamp);
CREATE INDEX IF NOT EXISTS idx_feat_ips ON network_features(src_ip, dst_ip);
CREATE INDEX IF NOT EXISTS idx_anomalies_record ON anomalies(record_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_score ON anomalies(score);

COMMIT;
"""

# ============================================================
#                     Database Manager Class
# ============================================================
class DatabaseManager:
    """Thread-safe SQLite manager for Network Anomaly storage."""

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

    # ============================================================
    # Data Insertion
    # ============================================================

    def insert_network_features(self, feature_dict: Dict[str, Any]) -> int:
        """
        Inserts a dictionary of network features into the database.
        Expected keys: src_ip, dst_ip, src_port, dst_port, protocol,
        total_bytes, total_packets, avg_packet_size, pkt_rate,
        syn_count, ack_count, syn_ack_ratio, port_entropy, metadata
        """
        metadata_json = json.dumps(feature_dict.get('metadata', {}))

        sql = """
            INSERT INTO network_features (
                src_ip, dst_ip, src_port, dst_port, protocol,
                total_bytes, total_packets, avg_packet_size,
                pkt_rate, syn_count, ack_count, syn_ack_ratio,
                port_entropy, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            feature_dict.get('src_ip'), feature_dict.get('dst_ip'),
            feature_dict.get('src_port'), feature_dict.get('dst_port'),
            feature_dict.get('protocol'), feature_dict.get('total_bytes'),
            feature_dict.get('total_packets'), feature_dict.get('avg_packet_size'),
            feature_dict.get('pkt_rate'), feature_dict.get('syn_count'),
            feature_dict.get('ack_count'), feature_dict.get('syn_ack_ratio'),
            feature_dict.get('port_entropy'), metadata_json
        )

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            rid = cur.lastrowid
            self._conn.commit()
        return rid

    def insert_anomaly(self, record_id: int, detector_name: str, score: float,
                       explanation: str = "", detected_at: Optional[str] = None) -> int:
        """Saves a detected anomaly linked to a specific network record."""
        detected_at = detected_at or datetime.now().isoformat()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO anomalies (record_id, detector_name, score, explanation, detected_at) VALUES (?, ?, ?, ?, ?)",
                (record_id, detector_name, score, explanation, detected_at)
            )
            aid = cur.lastrowid
            self._conn.commit()
        return aid

    def insert_label(self, feature_id: int, label: str, comment: str = "") -> int:
        """Saves a ground-truth label for a specific feature record."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO labels (feature_id, label, comment) VALUES (?, ?, ?)",
                (feature_id, label, comment)
            )
            lid = cur.lastrowid
            self._conn.commit()
        return lid

    # ============================================================
    # Maintenance & Close
    # ============================================================
    def vacuum(self) -> None:
        with self._lock:
            self._conn.execute("VACUUM")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            finally:
                self._conn.close()

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(f"Initializing A-DAPT Database at: {DB_PATH}")
    db = DatabaseManager(path=DB_PATH)
    print("✅ Database schema verified and ready.")
    db.close()