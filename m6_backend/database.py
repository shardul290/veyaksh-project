"""SQLite schema and initializer for the M6 nodes and events database."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "veyaksh.db"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    cam_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    track_id TEXT,
    class TEXT NOT NULL,
    conf REAL,
    zone_id TEXT,
    plate_text TEXT,
    plate_conf REAL,
    plate_img TEXT,
    clip TEXT NOT NULL,
    snapshot TEXT,
    received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (node_id) REFERENCES nodes(node_id)
);
"""

LOCATIONS_FILE = Path(__file__).resolve().parents[1] / "configs" / "locations.json"


def load_camera_nodes() -> tuple[tuple[str, str, float, float], ...]:
    """Load map locations from the editable project configuration."""
    raw = json.loads(LOCATIONS_FILE.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of locations in {LOCATIONS_FILE}")
    result = []
    seen = set()
    for location in raw:
        if not isinstance(location, dict):
            raise ValueError("Each location must be a JSON object")
        node_id, name = location.get("node_id"), location.get("name")
        if not isinstance(node_id, str) or not node_id or node_id in seen:
            raise ValueError(f"Location node_id must be a unique non-empty string: {node_id!r}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Location {node_id} must have a name")
        lat, lon = float(location["lat"]), float(location["lon"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"Location {node_id} has invalid coordinates")
        seen.add(node_id)
        result.append((node_id, name.strip(), lat, lon))
    return tuple(result)

_initialize_lock = threading.Lock()


def initialize_database(db_path: Path | str = DB_PATH) -> Path:
    # Camera playback can encounter several events on the same frame. Keep
    # first-use migrations serialized while concurrent requests start up.
    with _initialize_lock:
        return _initialize_database(db_path)


def _initialize_database(db_path: Path | str = DB_PATH) -> Path:
    """Create the schema and seed placeholder nodes without replacing existing data."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript(SCHEMA)
        event_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(events)").fetchall()
        }
        if "snapshot" not in event_columns:
            connection.execute("ALTER TABLE events ADD COLUMN snapshot TEXT")
        if "archived" not in event_columns:
            connection.execute("ALTER TABLE events ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
            # Footage from before the surveyed locations was declared invalid.
            # Retain its records for audit without attaching it to the new pins.
            connection.execute("UPDATE events SET archived = 1")
        for name, definition in {
            "status": "TEXT NOT NULL DEFAULT 'pending'",
            "status_at": "TEXT",
            "video_ts_s": "REAL",
            "source_recording": "TEXT",
            "event_key": "TEXT",
        }.items():
            if name not in event_columns:
                connection.execute(f"ALTER TABLE events ADD COLUMN {name} {definition}")
        connection.executescript("""
            CREATE UNIQUE INDEX IF NOT EXISTS events_event_key ON events(event_key) WHERE event_key IS NOT NULL;
            CREATE TABLE IF NOT EXISTS event_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL REFERENCES events(id),
                status TEXT NOT NULL,
                ts TEXT NOT NULL
            );
        """)
        connection.executemany(
            """INSERT INTO nodes (node_id, name, lat, lon) VALUES (?, ?, ?, ?)
               ON CONFLICT(node_id) DO UPDATE SET name = excluded.name,
               lat = excluded.lat, lon = excluded.lon""",
            load_camera_nodes(),
        )

    return path


if __name__ == "__main__":
    print(f"Initialized SQLite database: {initialize_database()}")
