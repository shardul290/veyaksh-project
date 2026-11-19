"""Load source-bound analyses; invalidate them when recordings or results change."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from m3_events.playback import MEDIA_ROOT, PLAYBACK_ROOT, read_jsonl, source_identity

_cache: dict[str, tuple[tuple, dict[str, Any]]] = {}


def load_analysis(cam_id: str, cameras: list[dict], directory: Path | None = None) -> dict[str, Any]:
    camera = next((c for c in cameras if c["cam_id"] == cam_id), None)
    if camera is None:
        raise ValueError("Camera not found")
    manifest_path = (directory or PLAYBACK_ROOT) / f"{cam_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recording = next((r for r in camera["recordings"] if r["path"] == manifest["source"]["path"]), None)
    if recording is None or manifest["cam_id"] != cam_id or manifest["node_id"] != camera["node_id"]:
        raise ValueError("Analysis does not match this camera recording")
    source = (MEDIA_ROOT / recording["path"]).resolve()
    if not source.is_relative_to(MEDIA_ROOT.resolve()) or source_identity(source) != manifest["source"]:
        raise ValueError("Recording changed; prepare its analysis again")
    tracks, events = Path(manifest["tracks"]), Path(manifest["events"])
    key = (str(manifest_path), manifest_path.stat().st_mtime_ns,
           tracks.stat().st_mtime_ns, events.stat().st_mtime_ns, manifest["analysis_id"])
    if cam_id not in _cache or _cache[cam_id][0] != key:
        import cv2
        capture = cv2.VideoCapture(str(source))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
        finally:
            capture.release()
        detections = read_jsonl(tracks)
        processing = manifest.get("processing", {})
        payload = {"analysis_id": manifest["analysis_id"], "source": manifest["source"],
                   "recording": recording["path"], "start_ts": recording.get("start_ts"),
                   "node_id": camera["node_id"], "cam_id": cam_id,
                   "frame_size": manifest["frame_size"], "detections": detections,
                   "events": read_jsonl(events),
                   "coverage": {"complete": processing.get("complete", False),
                                "last_detection_s": max((r["ts"] for r in detections), default=0),
                                "source_duration_s": duration}}
        _cache[cam_id] = key, payload
    return _cache[cam_id][1]
