"""Tiny SQLite storage for prediction history (standard library only)."""
import json
import logging
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger(__name__)
RISK_LEVELS = ("Low", "Moderate", "High")


def _resolve_path():
    """DATABASE_PATH env var > instance/ folder > system temp folder (read-only hosts)."""
    env_path = os.environ.get("DATABASE_PATH")
    if env_path:
        return Path(env_path)
    folder = BASE_DIR / "instance"
    try:
        folder.mkdir(exist_ok=True)
        if os.access(folder, os.W_OK):
            return folder / "predictions.db"
    except OSError:
        pass
    return Path(tempfile.gettempdir()) / "predictions.db"


DB_PATH = _resolve_path()


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(_connect()) as conn, conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   created_at TEXT NOT NULL,
                   source TEXT NOT NULL,
                   inputs TEXT NOT NULL,
                   prediction INTEGER NOT NULL,
                   probability REAL NOT NULL,
                   risk TEXT NOT NULL)"""
        )
    log.info("History database ready at %s", DB_PATH)


def save_prediction(source, values, result):
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO predictions (created_at, source, inputs, prediction, probability, risk) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (created, source, json.dumps(values), result["prediction"],
             result["probability"], result["risk"]),
        )


def get_history(limit=50):
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    history = []
    for row in rows:
        item = dict(row)
        item["inputs"] = json.loads(item["inputs"])
        item["percent"] = round(item["probability"] * 100, 1)
        history.append(item)
    return history


def get_stats(recent=30):
    with closing(_connect()) as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        average = conn.execute("SELECT AVG(probability) FROM predictions").fetchone()[0]
        counts = dict(conn.execute("SELECT risk, COUNT(*) FROM predictions GROUP BY risk").fetchall())
        latest = conn.execute(
            "SELECT probability FROM predictions ORDER BY id DESC LIMIT ?", (recent,)
        ).fetchall()
    return {
        "total": total,
        "average_percent": round((average or 0) * 100, 1),
        "counts": {level: counts.get(level, 0) for level in RISK_LEVELS},
        "recent": [round(row[0] * 100, 1) for row in reversed(latest)],
    }


def clear_history():
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM predictions")
