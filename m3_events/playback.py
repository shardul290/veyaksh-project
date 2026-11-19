"""Versioned, camera-bound playback analysis and consistent compact overlays."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MEDIA_ROOT = ROOT / "m6_backend" / "media"
PLAYBACK_ROOT = ROOT / "runs" / "playback"


def source_identity(source: Path) -> dict[str, Any]:
    source = source.resolve()
    stat = source.stat()
    return {"path": source.relative_to(MEDIA_ROOT.resolve()).as_posix(),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def event_key(identity: dict[str, Any], event: dict[str, Any], node_id: str, cam_id: str) -> str:
    value = [identity, node_id, cam_id, event["type"], int(event["track_id"]),
             round(float(event["video_ts_s"]), 6), event["zone_id"]]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def register_analysis(video: Path, tracks: Path, events: Path, config: dict[str, Any],
                      directory: Path | None = None, metadata: Path | None = None) -> Path | None:
    """Publish only analyses tied to a source inside the dashboard media directory."""
    try:
        identity = source_identity(video)
    except ValueError:
        return None
    payload = {"version": 1, "source": identity, "node_id": config["node_id"],
               "cam_id": config["cam_id"], "frame_size": config["frame_size"],
               "tracks": str(tracks.resolve()), "events": str(events.resolve())}
    if metadata and metadata.is_file():
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if info.get("source_size") == identity["size"] and info.get("source_mtime_ns") == identity["mtime_ns"]:
            payload["processing"] = info
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    for path in (tracks, events):
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                fingerprint.update(chunk)
    payload["analysis_id"] = fingerprint.hexdigest()
    directory = directory or PLAYBACK_ROOT
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{config['cam_id']}.json"
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def merge_display_tracks(raw: list[dict[str, Any]], classified: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep visible detections while restricting path decisions to confirmed IDs.

    ByteTrack can return a valid detection before assigning an identity. These
    boxes must remain visible, even though they cannot generate temporal alerts.
    """
    def key(row):
        return row["frame"], row["ts"], row.get("track_id"), row["class"]
    decisions = {key(row): row.get("path") for row in classified}
    rows = []
    for detection in raw:
        row = dict(detection)
        if row["class"] == "pedestrian":
            row["path"] = decisions.get(key(row)) or {
                "status": "unknown", "zone_id": None,
                "reason": "untracked_detection" if row.get("track_id") is None else "track_not_confirmed",
            }
        rows.append(row)
    return sorted(rows, key=lambda row: (row["ts"], row["frame"]))


def draw_detection(frame, detection: dict[str, Any], alert: bool = False, text: str | None = None) -> None:
    """Thin boxes, small regular text, and a translucent label; no filled boxes."""
    import cv2

    h, w = frame.shape[:2]
    scale = max(.4, w / 1920 * .42)
    thickness = max(1, round(w / 1920))
    x1, y1, x2, y2 = [int(round(v)) for v in detection["bbox"]]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    status = detection.get("path", {}).get("status")
    unsafe = alert or status == "wrong_path"
    pedestrian = detection["class"] == "pedestrian"
    color = (65, 65, 235) if unsafe else (90, 210, 70) if pedestrian else (220, 180, 80)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    label = f"#{detection.get('track_id') if detection.get('track_id') is not None else '?'} {detection['class']}"
    if text:
        label = text
    elif unsafe:
        label += " | WRONG PATH" if pedestrian else " | ALERT"
    elif status == "right_path":
        label += " | right path"
    elif pedestrian and status:
        label += " | " + status
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(label, font, scale, thickness)
    left = max(0, min(x1, w - tw - 6))
    top = max(0, y1 - th - base - 6)
    right, bottom = min(w, left + tw + 6), min(h, top + th + base + 6)
    region = frame[top:bottom, left:right]
    region[:] = (region.astype("float32") * .72).astype("uint8")
    cv2.putText(frame, label, (left + 3, top + th + 2), font, scale,
                color, thickness, cv2.LINE_AA)


def render_video(video: Path, tracks: Path, events: Path, output: Path) -> None:
    """Render final rule decisions over every source frame, at source timing."""
    import cv2
    from bisect import bisect_right
    from collections import defaultdict

    cv2.setNumThreads(2)

    grouped = defaultdict(list)
    for row in read_jsonl(tracks):
        grouped[row["ts"]].append(row)
    times = sorted(grouped)
    alerts = read_jsonl(events)
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    fps = capture.get(cv2.CAP_PROP_FPS)
    w, h = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not capture.isOpened() or fps <= 0:
        capture.release()
        raise ValueError(f"Cannot read video {video}")
    output.parent.mkdir(parents=True, exist_ok=True)
    ratio = min(1, 1920 / w)
    size = (max(2, round(w * ratio) // 2 * 2), max(2, round(h * ratio) // 2 * 2))
    temporary = output.with_name(output.stem + ".rendering" + output.suffix)
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Cannot write video {output}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            ts = index / fps
            if (w, h) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            i = bisect_right(times, ts + 1e-5) - 1
            if i >= 0 and ts - times[i] <= .25:
                for row in grouped[times[i]]:
                    alert = any(e["track_id"] == row["track_id"] and 0 <= ts - e["video_ts_s"] <= 2 for e in alerts)
                    shown = {**row, "bbox": [value * (size[0] / w if i % 2 == 0 else size[1] / h)
                                            for i, value in enumerate(row["bbox"])]}
                    draw_detection(frame, shown, alert=alert)
            writer.write(frame)
            index += 1
            if index % 1000 == 0:
                print(f"Rendered {index} frames: {output.name}", flush=True)
    finally:
        capture.release()
        writer.release()
    temporary.replace(output)
