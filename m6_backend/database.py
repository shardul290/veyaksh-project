"""SQLite schema and initializer for the M6 nodes and events database."""

from __future__ import annotations

import sqlite3
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
    received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (node_id) REFERENCES nodes(node_id)
);
"""

# Placeholder locations near IIT Roorkee; replace these when camera locations
# are confirmed by the team.
PLACEHOLDER_NODES = (
    ("N01", "Solanipuram", 29.8640, 77.8960),
    ("N02", "Adarsh nagar", 29.8660, 77.8990),
    ("N03", "Civil lines", 29.8610, 77.9020),
)


def initialize_database(db_path: Path | str = DB_PATH) -> Path:
    """Create the schema and seed placeholder nodes without replacing existing data."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA)
        connection.executemany(
            """INSERT INTO nodes (node_id, name, lat, lon) VALUES (?, ?, ?, ?)
               ON CONFLICT(node_id) DO UPDATE SET name = excluded.name""",
            PLACEHOLDER_NODES,
        )

    return path


if __name__ == "__main__":
    print(f"Initialized SQLite database: {initialize_database()}")
