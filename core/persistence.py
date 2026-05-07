"""
Persistence Layer (Task 19: Logging Upgrade).

SQLite logging + frame archival with trend, repeat count, and time delta tracking.
"""

import json
import sqlite3
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import DB_PATH, FRAME_ARCHIVE_DIR
from core.event_logic import InspectionEvent

logger = logging.getLogger(__name__)


class PersistenceStore:
    """SQLite-backed event logger with frame archival and enhanced logging."""

    def __init__(self, db_path: Path = DB_PATH, archive_dir: Path = FRAME_ARCHIVE_DIR):
        self.db_path = db_path
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    defect TEXT,
                    severity TEXT,
                    priority TEXT,
                    action TEXT,
                    rationale TEXT,
                    visual_evidence TEXT,
                    confidence TEXT,
                    trend TEXT,
                    repeat_count INTEGER DEFAULT 0,
                    time_delta REAL,
                    frame_path TEXT
                )
            """)
            conn.commit()
        logger.info(f"Database initialized: {self.db_path}")

    def save_frame(self, frame: np.ndarray, event: InspectionEvent) -> str:
        """Save anomaly frame to disk. Returns the file path."""
        filename = f"{int(event.timestamp)}_{event.defect.replace(' ', '_')}.jpg"
        filepath = self.archive_dir / filename
        cv2.imwrite(str(filepath), frame)
        return str(filepath)

    def log_event(self, event: InspectionEvent, frame: Optional[np.ndarray] = None):
        """Persist an event to SQLite. Optionally archive the frame."""
        frame_path = None
        if frame is not None and event.defect.lower() != "none":
            frame_path = self.save_frame(frame, event)
            event.frame_path = frame_path

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO events
                   (timestamp, defect, severity, priority, action,
                    rationale, visual_evidence, confidence,
                    trend, repeat_count, time_delta, frame_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event.timestamp, event.defect, event.severity, event.priority,
                 event.action, event.rationale, event.visual_evidence,
                 event.confidence, event.trend, event.repeat_count,
                 event.time_delta, event.frame_path),
            )
            conn.commit()

    def get_recent(self, n: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_defect_stats(self) -> dict:
        """Task 19: Get aggregate defect statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT defect, COUNT(*) as count,
                       MAX(severity) as max_severity,
                       MAX(timestamp) as last_seen
                FROM events
                WHERE defect != 'none'
                GROUP BY defect
                ORDER BY count DESC
            """).fetchall()
        return {r["defect"]: dict(r) for r in rows}
