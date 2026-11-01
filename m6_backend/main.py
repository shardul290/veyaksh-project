from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("EVENTS_DB_PATH", ROOT / "events.sqlite3"))
app = FastAPI(title="Video Event Capture API", version="1.0.0")


class EventIn(BaseModel):
    type: str = Field(min_length=1, max_length=100)
    details: dict[str, Any] = Field(default_factory=dict)


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL,
        details TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    db.commit()
    return db


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/events", status_code=201)
def create_event(event: EventIn) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        cur = db.execute(
            "INSERT INTO events (type, details, created_at) VALUES (?, ?, ?)",
            (event.type, json.dumps(event.details), created_at),
        )
        event_id = cur.lastrowid
    return {"id": event_id, "type": event.type, "details": event.details, "created_at": created_at}


@app.get("/events")
def list_events() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("SELECT id, type, details, created_at FROM events ORDER BY id").fetchall()
    return [
        {"id": row["id"], "type": row["type"], "details": json.loads(row["details"]), "created_at": row["created_at"]}
        for row in rows
    ]
