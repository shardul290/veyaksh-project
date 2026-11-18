from __future__ import annotations

import os
import json
import sqlite3
from datetime import datetime, timezone
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
MAP_DASHBOARD_FILE = ROOT / "m6_dashboard" / "map.html"
PRIVACY_FILE = ROOT / "m6_dashboard" / "privacy.html"
TERMS_FILE = ROOT / "m6_dashboard" / "terms.html"
FAVICON_FILE = ROOT / "m6_dashboard" / "favicon.svg"
BRAND_LOGO_FILE = ROOT / "m6_dashboard" / "veyaksh-logo.jpg"
DB_PATH = Path(os.environ.get("EVENTS_DB_PATH", DEFAULT_DB_PATH))
MEDIA_DIR = BACKEND_DIR / "media"

app = FastAPI(title="VEYAKSH Traffic Events API", version="2.0.0")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
app.mount("/dashboard-assets", StaticFiles(directory=ROOT / "m6_dashboard"), name="dashboard-assets")


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
    video_ts_s: float | None = Field(default=None, ge=0)

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


@app.get("/map", include_in_schema=False)
def map_dashboard() -> FileResponse:
    return FileResponse(DASHBOARD_FILE)


@app.get("/history", include_in_schema=False)
def history_page() -> FileResponse:
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


@app.get("/branding/veyaksh-logo.jpg", include_in_schema=False)
def brand_logo() -> FileResponse:
    return FileResponse(BRAND_LOGO_FILE, media_type="image/jpeg")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/events", status_code=201)
def create_event(event: EventIn) -> dict[str, int]:
    if {"N01": "C01", "N02": "C02"}.get(event.node_id) != event.cam_id:
        raise HTTPException(422, "Use the surveyed camera pair N01/C01 or N02/C02")
    with connect() as db:
        node = db.execute("SELECT 1 FROM nodes WHERE node_id = ?", (event.node_id,)).fetchone()
        if node is None:
            raise HTTPException(status_code=422, detail=f"Unknown node_id: {event.node_id}")

        plate = event.plate
        cursor = db.execute(
            """INSERT INTO events
               (node_id, cam_id, ts, type, track_id, class, conf, zone_id,
                plate_text, plate_conf, plate_img, clip, snapshot, video_ts_s)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.node_id, event.cam_id, event.ts, event.type, event.track_id,
                event.vehicle_class, event.conf, event.zone_id,
                plate.text if plate else None,
                plate.conf if plate else None,
                plate.img if plate else None,
                event.clip,
                event.snapshot,
                event.video_ts_s,
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
    before_id: int | None = Query(default=None, ge=1),
    status: Literal["pending", "false_alert", "action_taken", "authority_requested"] | None = None,
    active: bool = False,
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
    if before_id is not None:
        clauses.append("id < ?")
        values.append(before_id)
    if status is not None:
        clauses.append("status = ?")
        values.append(status)
    if active:
        clauses.append("archived = 0")
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
            "status": row["status"], "status_at": row["status_at"],
            "video_ts_s": row["video_ts_s"],
            "archived": bool(row["archived"]),
        })
    return result


@app.get("/nodes")
def list_nodes() -> list[dict[str, object]]:
    with connect() as db:
        rows = db.execute(
            "SELECT node_id, name, lat, lon FROM nodes WHERE node_id IN ('N01', 'N02') ORDER BY node_id"
        ).fetchall()
    return [dict(row) for row in rows]


class ReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["false_alert", "action_taken", "authority_requested"]


@app.get("/events-summary")
def event_summary() -> dict[str, int]:
    result = dict.fromkeys(("pending", "false_alert", "action_taken", "authority_requested"), 0)
    with connect() as db:
        for row in db.execute("SELECT status, COUNT(*) AS total FROM events WHERE archived = 0 GROUP BY status"):
            result[row["status"]] = row["total"]
    return result


@app.patch("/events/{event_id}/status")
def review_event(event_id: int, review: ReviewIn) -> dict[str, object]:
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        event = db.execute("SELECT status, status_at, archived FROM events WHERE id = ?", (event_id,)).fetchone()
        if event is None:
            raise HTTPException(404, "Event not found")
        if event["archived"]:
            raise HTTPException(409, "Legacy evidence is archived and cannot be actioned")
        if event["status"] == review.status:
            return {"id": event_id, "status": event["status"], "status_at": event["status_at"]}
        allowed = {"pending": {"false_alert", "action_taken"}, "action_taken": {"authority_requested"}}
        if review.status not in allowed.get(event["status"], set()):
            raise HTTPException(409, "This event has already been reviewed. Refresh its status.")
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        db.execute("UPDATE events SET status = ?, status_at = ? WHERE id = ?", (review.status, now, event_id))
        db.execute("INSERT INTO event_actions (event_id, status, ts) VALUES (?, ?, ?)", (event_id, review.status, now))
    return {"id": event_id, "status": review.status, "status_at": now}


@app.get("/events/{event_id}/actions")
def event_actions(event_id: int) -> list[dict[str, object]]:
    with connect() as db:
        if db.execute("SELECT 1 FROM events WHERE id = ?", (event_id,)).fetchone() is None:
            raise HTTPException(404, "Event not found")
        return [dict(row) for row in db.execute(
            "SELECT status, ts FROM event_actions WHERE event_id = ? ORDER BY id", (event_id,)
        ).fetchall()]


@app.get("/cameras")
def camera_recordings() -> list[dict[str, object]]:
    config = json.loads((ROOT / "configs" / "cameras.json").read_text(encoding="utf-8"))
    for camera in config:
        available = []
        for recording in camera.get("recordings", []):
            path = (MEDIA_DIR / recording["path"]).resolve()
            if path.is_relative_to(MEDIA_DIR.resolve()) and path.is_file():
                available.append(recording)
        camera["recordings"] = available
    return config
