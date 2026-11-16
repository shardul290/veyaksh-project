from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .database import DB_PATH as DEFAULT_DB_PATH, initialize_database

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DASHBOARD_FILE = ROOT / "m6_dashboard" / "index.html"
PRIVACY_FILE = ROOT / "m6_dashboard" / "privacy.html"
TERMS_FILE = ROOT / "m6_dashboard" / "terms.html"
FAVICON_FILE = ROOT / "m6_dashboard" / "favicon.svg"
DB_PATH = Path(os.environ.get("EVENTS_DB_PATH", DEFAULT_DB_PATH))
MEDIA_DIR = BACKEND_DIR / "media"

app = FastAPI(title="VEYAKSH Traffic Events API", version="2.0.0")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")


class PlateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    conf: float = Field(ge=0, le=1)
    img: str = Field(min_length=1)


class EventIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(pattern=r"^N[0-9]{2}$")
    cam_id: str = Field(pattern=r"^C[0-9]+$")
    ts: str
    type: Literal["wrong_way", "encroachment"]
    track_id: int = Field(ge=0)
    vehicle_class: Literal[
        "car", "two_wheeler", "auto", "e_rickshaw", "cart", "pedestrian", "other"
    ] = Field(alias="class")
    conf: float = Field(ge=0, le=1)
    zone_id: str = Field(min_length=1)
    plate: PlateIn | None
    clip: str = Field(min_length=1)
    snapshot: str | None = None

    @field_validator("ts")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("ts must be an ISO 8601 date-time") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("ts must include a timezone offset")
        return value


def connect() -> sqlite3.Connection:
    initialize_database(DB_PATH)
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD_FILE)


@app.get("/privacy", include_in_schema=False)
def privacy_policy() -> FileResponse:
    return FileResponse(PRIVACY_FILE)


@app.get("/terms", include_in_schema=False)
def terms_and_conditions() -> FileResponse:
    return FileResponse(TERMS_FILE)


@app.get("/favicon.svg", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(FAVICON_FILE, media_type="image/svg+xml")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/events", status_code=201)
def create_event(event: EventIn) -> dict[str, int]:
    with connect() as db:
        node = db.execute("SELECT 1 FROM nodes WHERE node_id = ?", (event.node_id,)).fetchone()
        if node is None:
            raise HTTPException(status_code=422, detail=f"Unknown node_id: {event.node_id}")

        plate = event.plate
        cursor = db.execute(
            """INSERT INTO events
               (node_id, cam_id, ts, type, track_id, class, conf, zone_id,
                plate_text, plate_conf, plate_img, clip, snapshot)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.node_id, event.cam_id, event.ts, event.type, event.track_id,
                event.vehicle_class, event.conf, event.zone_id,
                plate.text if plate else None,
                plate.conf if plate else None,
                plate.img if plate else None,
                event.clip,
                event.snapshot,
            ),
        )
        event_id = cursor.lastrowid
    return {"id": event_id}


@app.get("/events")
def list_events(
    since_id: int | None = Query(default=None, ge=0),
    type: Literal["wrong_way", "encroachment"] | None = None,
    node_id: str | None = Query(default=None, pattern=r"^N[0-9]{2}$"),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, object]]:
    clauses: list[str] = []
    values: list[object] = []
    if since_id is not None:
        clauses.append("id > ?")
        values.append(since_id)
    if type is not None:
        clauses.append("type = ?")
        values.append(type)
    if node_id is not None:
        clauses.append("node_id = ?")
        values.append(node_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    values.append(limit)

    with connect() as db:
        rows = db.execute(
            f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ?", values
        ).fetchall()
    result: list[dict[str, object]] = []
    for row in rows:
        plate = None
        if row["plate_text"] is not None:
            plate = {
                "text": row["plate_text"],
                "conf": row["plate_conf"],
                "img": row["plate_img"],
            }
        result.append({
            "id": row["id"], "node_id": row["node_id"], "cam_id": row["cam_id"],
            "ts": row["ts"], "type": row["type"], "track_id": row["track_id"],
            "class": row["class"], "conf": row["conf"], "zone_id": row["zone_id"],
            "plate": plate, "clip": row["clip"], "snapshot": row["snapshot"],
            "received_at": row["received_at"],
        })
    return result


@app.get("/nodes")
def list_nodes() -> list[dict[str, object]]:
    with connect() as db:
        rows = db.execute(
            "SELECT node_id, name, lat, lon FROM nodes ORDER BY node_id"
        ).fetchall()
    return [dict(row) for row in rows]
