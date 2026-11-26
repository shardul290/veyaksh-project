"""Own live inference processes independently of browser viewers."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import struct
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import psutil
from .live_preview import preview_path

ACTIVE_STATES = {"starting", "running", "stopping"}


def safe_source(source: str) -> str:
    if source.lower().startswith(("rtsp://", "rtsps://")):
        parts = urlsplit(source)
        return f"{parts.scheme}://{parts.hostname}:{parts.port or 554}{parts.path}"
    return Path(source).name


def public_error(message: str, source: str) -> str:
    return re.sub(r"rtsps?://[^\s]+", "RTSP source", message.replace(source, safe_source(source)))[:500]


class LiveManager:
    def __init__(self, root: Path, media: Path, persist_event: Callable):
        self.root = root
        self.media = media
        self.directory = root / "runs" / "live_sessions"
        self.persist_event = persist_event
        self.lock = threading.RLock()
        self.sessions: dict[str, dict] = {}

    def start(self, config: dict) -> dict:
        with self.lock:
            if any(s["state"] in ACTIVE_STATES for s in self.sessions.values()):
                raise ValueError("A live session is already active. Stop it before starting another source.")
            session_id = uuid.uuid4().hex
            directory = self.directory / session_id
            directory.mkdir(parents=True)
            config = {**config, "session_id": session_id, "directory": str(directory),
                      "media_dir": str(self.media),
                      "started_at": datetime.now(timezone.utc).isoformat()}
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            state = {"session_id": session_id, "state": "starting", "source": safe_source(config["source"]),
                     "source_kind": config["source_kind"], "cam_id": config["zones"]["cam_id"],
                     "node_id": config["zones"]["node_id"], "model": Path(config["model"]).name,
                     "decoder": "GStreamer", "event_count": 0, "recent_events": [], "error": None,
                     "browser_input": config.get("browser_input", False),
                     "input_token": uuid.uuid4().hex, "last_input_ts": -1.0,
                     "directory": directory, "config": config, "process": None}
            self.sessions[session_id] = state
            try:
                env = {**os.environ, "PYTHONUTF8": "1", "YOLO_OFFLINE": "true"}
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "m6_backend.live_worker",
                     "--config", str(directory / "config.json")],
                    cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                )
                state["process"] = process
                watcher = threading.Thread(target=self._watch, args=(session_id,), daemon=True,
                                           name=f"live-{session_id[:8]}")
                state["watcher"] = watcher
                watcher.start()
            except Exception:
                state["state"] = "failed"
                raise
            return {**self.snapshot(session_id), "input_token": state["input_token"]}

    def _watch(self, session_id: str) -> None:
        session = self.sessions[session_id]
        process = session["process"]
        directory = session["directory"]
        try:
            with (directory / "worker.log").open("w", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if not line.startswith("@event "):
                        continue
                    payload = json.loads(line[7:])
                    saved = self.persist_event(payload)
                    with (directory / "events.jsonl").open("a", encoding="utf-8") as events:
                        events.write(json.dumps({**payload, **saved}) + "\n")
                    with self.lock:
                        session["event_count"] += 1
                        session["recent_events"] = [{**payload, **saved}, *session["recent_events"]][:12]
            code = process.wait()
            with self.lock:
                status = self._read_status(session)
                stopped = session["state"] == "stopping"
                session["state"] = "stopped" if stopped else "completed" if code == 0 else "failed"
                if code != 0 and not stopped:
                    session["error"] = public_error(status.get("error", "The live worker stopped. See its log."),
                                                   session["config"]["source"])
        except Exception as exc:
            self._terminate(process)
            with self.lock:
                session["state"] = "failed"
                session["error"] = public_error(str(exc), session["config"]["source"])
        finally:
            process.stdout.close()
            with self.lock:
                (directory / "result.json").write_text(json.dumps(self.snapshot(session_id)), encoding="utf-8")

    @staticmethod
    def _read_status(session: dict) -> dict:
        try:
            return json.loads((session["directory"] / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def snapshot(self, session_id: str) -> dict:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            result = {key: value for key, value in session.items()
                      if key not in {"process", "watcher", "directory", "config", "input_token", "last_input_ts"}}
            status = self._read_status(session)
            for field in ("source_fps", "display_fps", "detection_fps", "video_ts_s", "duration_s",
                          "decoded_frames", "inferred_frames", "skipped_frames", "box_age_s",
                          "detections", "backend", "width", "height", "decoder", "decoder_warning",
                          "boxes", "detection_ts_s", "input_ready", "inference_ms", "presentation_ts_s", "preparation"):
                if field in status:
                    result[field] = status[field]
            if result["state"] == "starting" and status.get("state") == "running":
                result["state"] = "running"
            if status.get("error"):
                result["error"] = public_error(status["error"], session["config"]["source"])
            result["frame_url"] = f"/live-sessions/{session_id}/frames"
            result["has_frame"] = (session["directory"] / "frame.jpg").is_file()
            return result

    def list(self) -> list[dict]:
        with self.lock:
            return [self.snapshot(key) for key in reversed(self.sessions)]

    def busy(self) -> bool:
        with self.lock:
            return any(s["state"] in ACTIVE_STATES for s in self.sessions.values())

    def submit_frame(self, session_id: str, token: str, ts: float, image: bytes) -> dict:
        with self.lock:
            session = self.sessions[session_id]
            if token != session["input_token"] or not session["browser_input"]:
                raise ValueError("This tab does not own the playback session.")
            if session["state"] not in ACTIVE_STATES:
                raise ValueError("The live session has ended.")
            if ts < session["last_input_ts"]:
                raise ValueError("Playback timestamps must move forward. Start a new session to replay.")
            sequence = int(time.monotonic_ns())
            temporary = session["directory"] / "input.tmp"
            temporary.write_bytes(struct.pack("<dQ", ts, sequence) + image)
            try:
                temporary.replace(session["directory"] / "input.bin")
            except PermissionError:
                temporary.unlink(missing_ok=True)
                return self.snapshot(session_id)
            session["last_input_ts"] = ts
            return self.snapshot(session_id)

    def end_input(self, session_id: str, token: str) -> None:
        with self.lock:
            session = self.sessions[session_id]
            if token != session["input_token"]:
                raise ValueError("This tab does not own the playback session.")
            (session["directory"] / "input_end").touch()

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in reversed(children):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            process.terminate()
            process.wait(timeout=5)
        except (psutil.NoSuchProcess, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=5)

    def stop(self, session_id: str) -> dict:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            if session["state"] in ACTIVE_STATES:
                session["state"] = "stopping"
                (session["directory"] / "stop").touch()
            process = session["process"]
            watcher = session.get("watcher")
        try:
            if process is not None:
                process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self._terminate(process)
        if watcher is not None and watcher is not threading.current_thread():
            watcher.join(timeout=8)
        return self.snapshot(session_id)

    def close(self) -> None:
        for session in self.list():
            self.stop(session["session_id"])

    def frames(self, session_id: str):
        # Read-only viewers share one inference worker. Slow clients receive the newest JPEG.
        with self.lock:
            session = self.sessions[session_id]
        last = None
        while True:
            try:
                frame = session["directory"] / "frame.jpg"
                identity = frame.stat().st_mtime_ns
                if identity != last:
                    content = frame.read_bytes()
                    last = identity
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(content)).encode() + b"\r\n\r\n" + content + b"\r\n")
            except (FileNotFoundError, PermissionError):
                pass
            with self.lock:
                ended = session["state"] not in ACTIVE_STATES
            if ended:
                break
            time.sleep(0.04)
        yield b"--frame--\r\n"

    def source_path(self, session_id: str) -> Path:
        # Historical evidence can still resolve its source after a backend restart.
        if not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise FileNotFoundError("Session not found")
        config = json.loads((self.directory / session_id / "config.json").read_text(encoding="utf-8"))
        if config["source_kind"] != "file":
            raise FileNotFoundError("This session uses a network stream")
        path = Path(config["source"])
        if not path.is_file():
            raise FileNotFoundError("Source recording is unavailable")
        identity = config.get("source_identity")
        stat = path.stat()
        if identity and (stat.st_size, stat.st_mtime_ns) != (identity["size"], identity["mtime_ns"]):
            raise FileNotFoundError("Source recording changed after this session")
        return path

    def playback_path(self, session_id: str) -> Path:
        source = self.source_path(session_id)
        preview = preview_path(self.root, source)
        return preview if preview.is_file() else source
