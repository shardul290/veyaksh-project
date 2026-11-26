"""GStreamer source clock, latest-frame YOLO26 inference and live event rules."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import threading
import time
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from m2_detect_track.export_tracks_jsonl import iter_gstreamer_frames, normalize_names, TARGET_NAMES
from m3_events.event_engine import load_zones, VEHICLE_CLASSES
from m3_events.playback import draw_detection
from m3_events.rules.wrong_way import WrongWayRule
from m3_events.rules.encroachment import EncroachmentRule
from m3_events.rules.pedestrian_path import PedestrianPathRule
from m6_backend.live_preview import preview_path


def atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    try:
        temporary.replace(path)
    except PermissionError:
        # A Windows HTTP reader can briefly hold the previous JPEG open.
        temporary.unlink(missing_ok=True)


class OnlineRules:
    def __init__(self, config: dict):
        self.width, self.height, self.lanes, self.no_park = load_zones(config)
        self.max_gap = float(config.get("max_track_gap_s", 1.0))
        self.consistency = float(config.get("min_track_class_consistency", 0.0))
        self.wrong_way = WrongWayRule(self.width, self.height, self.max_gap)
        self.encroachment = EncroachmentRule(self.width, self.height, self.max_gap)
        self.pedestrian = PedestrianPathRule(
            self.width, self.height, config.get("pedestrian_zones", []),
            {"max_gap_s": self.max_gap, **config.get("pedestrian_path", {})})
        self.labels = {}
        self.seen = {}

    def update(self, detections: list[dict], ts: float) -> list[dict]:
        expired = {track for track, seen in self.seen.items() if ts - seen > self.max_gap}
        for track in expired:
            self.seen.pop(track, None)
            self.labels.pop(track, None)
            self.pedestrian.states.pop(track, None)
        for rule in (self.wrong_way, self.encroachment):
            for key in list(rule.states):
                if key[0] in expired:
                    rule.states.pop(key)
        events = []
        for detection in detections:
            track = detection["track_id"]
            if track is None:
                continue
            self.seen[track] = ts
            history = self.labels.setdefault(track, deque(maxlen=30))
            history.append(detection["class"])
            majority, count = Counter(history).most_common(1)[0]
            if majority != detection["class"] or count / len(history) < self.consistency:
                continue
            if detection["class"] == "pedestrian":
                decision, event = self.pedestrian.update(detection)
                detection["path"] = decision
                if event:
                    events.append(event)
            elif detection["class"] in VEHICLE_CLASSES:
                for lane in self.lanes:
                    event = self.wrong_way.update(detection, lane)
                    if event:
                        events.append(event)
                for zone in self.no_park:
                    event = self.encroachment.update(detection, zone)
                    if event:
                        events.append(event)
        return events


class LivePipeline:
    def __init__(self, config: dict):
        self.config = config
        self.directory = Path(config["directory"])
        self.stop = threading.Event()
        self.finished = threading.Event()
        self.condition = threading.Condition()
        self.latest = None
        self.detections = []
        self.detection_ts = None
        self.inferred = 0
        self.decoded = 0
        self.displayed = 0
        self.inference_times = deque(maxlen=20)
        self.display_times = deque(maxlen=60)
        self.error = None
        self.decoder = None
        self.decoder_name = "GStreamer"
        self.decoder_warning = None
        self.rules = OnlineRules(config["zones"])
        self.last_event = {}
        self.started = time.monotonic()
        self.status = {"state": "starting"}
        self.source_fps = 25.0
        self.duration = None
        self.backend = "Loading YOLO26"
        self.inference_ms = None
        self.tracker = None

    def track_detections(self, result, index: int, ts: float) -> list[dict]:
        """Keep raw detector positions, including detections awaiting a track ID."""
        from ultralytics.trackers.byte_tracker import BYTETracker

        threshold = float(self.config["confidence"])
        if self.tracker is None:
            high = min(.25, max(.1, threshold))
            self.tracker = BYTETracker(SimpleNamespace(
                track_high_thresh=high, track_low_thresh=min(.08, high / 2),
                new_track_thresh=high, track_buffer=10, match_thresh=.8, fuse_score=True))
        boxes = result.boxes.cpu().numpy()
        tracked = self.tracker.update(boxes, result.orig_img)
        identities = {int(row[-1]): int(row[4]) for row in tracked}
        names = normalize_names(result.names)
        height, width = result.orig_img.shape[:2]
        return [{"frame": index, "ts": ts, "track_id": identities.get(i), "class": names[int(cid)],
                 "conf": float(conf), "bbox": [float(xyxy[0]) * self.rules.width / width,
                  float(xyxy[1]) * self.rules.height / height,
                  float(xyxy[2]) * self.rules.width / width,
                  float(xyxy[3]) * self.rules.height / height]}
                for i, (xyxy, cid, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf))
                if conf >= threshold]

    def probe(self) -> tuple[int, int]:
        source = self.config["source"]
        if self.config["source_kind"] == "rtsp":
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000]
            capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
        else:
            capture = cv2.VideoCapture(source)
        try:
            if not capture.isOpened():
                raise ValueError("Could not open the video source. Check its path or RTSP connection.")
            width, height = int(capture.get(3)), int(capture.get(4))
            fps = capture.get(cv2.CAP_PROP_FPS)
            if math.isfinite(fps) and 0 < fps <= 240:
                self.source_fps = fps
            if width <= 0 or height <= 0:
                raise ValueError("The source did not provide valid video dimensions.")
            if self.config["source_kind"] == "file":
                self.duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / self.source_fps
            if abs((width / height) / (self.rules.width / self.rules.height) - 1) > 0.02:
                raise ValueError("Source aspect ratio differs from this camera's zone calibration. Select matching camera zones.")
            preview_width = min(1280, width)
            preview_height = max(2, round(height * preview_width / width / 2) * 2)
            return preview_width, preview_height
        finally:
            capture.release()

    def _decoder_started(self, process):
        self.decoder = process

    def prepare_playback(self, size: tuple[int, int]) -> None:
        """Avoid concurrent 4K decode and inference on a shared integrated GPU."""
        import imageio_ffmpeg
        source = Path(self.config["source"])
        root = Path(__file__).resolve().parents[1]
        target = preview_path(root, source)
        if target.is_file():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.stem + "." + self.config["session_id"] + ".tmp.mp4")
        width = min(960, size[0]) // 2 * 2
        self.status["preparation"] = "Preparing a lightweight playback copy (first use only)…"
        self.write_status()
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
                   "-y", "-threads", "2", "-filter_threads", "1", "-i", str(source),
                   "-map", "0:v:0", "-an", "-vf", f"scale={width}:-2:flags=area,fps=25",
                   "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
                   "-threads", "2", "-movflags", "+faststart", str(temporary)]
        try:
            with target.with_suffix(".log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.decoder = process
                while process.poll() is None:
                    if (self.directory / "stop").exists():
                        self.stop.set()
                        process.terminate()
                        process.wait(timeout=5)
                        return
                    self.stop.wait(.1)
                if process.returncode:
                    raise RuntimeError("Could not prepare the local playback copy. See runs/live_previews/*.log.")
            temporary.replace(target)
        finally:
            self.decoder = None
            temporary.unlink(missing_ok=True)

    def frames(self, size: tuple[int, int]):
        if self.config.get("browser_input"):
            self.decoder_name = "Browser video"
            previous = None
            while not self.stop.is_set():
                try:
                    payload = (self.directory / "input.bin").read_bytes()
                    ts, sequence = struct.unpack("<dQ", payload[:16])
                    if sequence != previous:
                        frame = cv2.imdecode(np.frombuffer(payload[16:], dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            raise ValueError("Playback sent an invalid image.")
                        previous = sequence
                        yield sequence, ts, frame
                except (FileNotFoundError, PermissionError):
                    pass
                if (self.directory / "input_end").exists():
                    break
                self.stop.wait(0.02)
            return
        source = self.config["source"]
        if self.config["source_kind"] == "file":
            source = Path(source)
        try:
            yield from iter_gstreamer_frames(
                source, Path(self.config["gst_python"]), size, realtime=True,
                output_size=size, on_process=self._decoder_started)
            return
        except Exception:
            if self.decoded or self.stop.is_set():
                raise
            # Preserve the live workflow when the installed native plugins cannot load.
            self.decoder_name = "FFmpeg"
            self.decoder_warning = "GStreamer could not decode this source. Using FFmpeg for this session."
            print(self.decoder_warning, flush=True)
        params = [cv2.CAP_PROP_N_THREADS, 2]
        if self.config["source_kind"] == "rtsp":
            params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
        capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, params)
        try:
            if not capture.isOpened():
                raise ValueError("Both decoders could not open this source. Check the video path or RTSP connection.")
            clock = time.monotonic()
            index = 0
            while not self.stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    if self.config["source_kind"] == "rtsp":
                        raise RuntimeError("The RTSP connection stopped delivering video. Restart to reconnect.")
                    break
                ts = index / self.source_fps
                if self.config["source_kind"] == "file":
                    pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    if math.isfinite(pts) and pts >= 0:
                        ts = pts
                    if self.stop.wait(max(0, clock + ts - time.monotonic())):
                        break
                else:
                    ts = time.monotonic() - clock
                yield index, ts, cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                index += 1
        finally:
            capture.release()

    def decode(self, size: tuple[int, int]) -> None:
        first_pts = None
        last_ts = -1.0
        browser_index = 0
        try:
            frames = self.frames(size)
            try:
                for index, pts, frame in frames:
                    if self.stop.is_set():
                        break
                    if self.config.get("browser_input"):
                        ts = pts
                        index = browser_index
                        browser_index += 1
                    elif pts is not None:
                        if first_pts is None:
                            first_pts = pts
                        ts = max(0.0, pts - first_pts)
                    else:
                        ts = index / self.source_fps
                    if ts <= last_ts:
                        ts = last_ts + 1 / self.source_fps
                    last_ts = ts
                    with self.condition:
                        self.latest = (index, ts, frame, time.monotonic())
                        self.decoded += 1
                        self.condition.notify_all()
            finally:
                frames.close()
        except Exception as exc:
            if not self.stop.is_set():
                self.error = str(exc)
        finally:
            self.finished.set()
            with self.condition:
                self.condition.notify_all()

    def save_event(self, event: dict, frame) -> None:
        # One event per confirmed episode and track, independent of viewer count.
        identity = [self.config["session_id"], event["type"], event["track_id"],
                    event["zone_id"], round(event["video_ts_s"], 6)]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        evidence = Path(self.config["media_dir"]) / "event_evidence" / "live"
        evidence.mkdir(parents=True, exist_ok=True)
        snapshot = evidence / f"{key}.jpg"
        shot = frame.copy()
        h, w = shot.shape[:2]
        box = {**event, "bbox": [event["bbox"][0] * w / self.rules.width,
                                event["bbox"][1] * h / self.rules.height,
                                event["bbox"][2] * w / self.rules.width,
                                event["bbox"][3] * h / self.rules.height]}
        draw_detection(shot, box, alert=True, text=f"#{event['track_id']} {event['class']} | {event['type']}")
        if not cv2.imwrite(str(snapshot), shot):
            raise RuntimeError("Could not save event evidence.")
        relative = snapshot.relative_to(Path(self.config["media_dir"])).as_posix()
        start = datetime.fromisoformat(self.config["started_at"])
        clip = (f"/live-sessions/{self.config['session_id']}/source#t={event['video_ts_s']:.3f}"
                if self.config["source_kind"] == "file" else relative)
        payload = {
            "node_id": self.config["zones"]["node_id"], "cam_id": self.config["zones"]["cam_id"],
            "ts": (start + timedelta(seconds=event["video_ts_s"])).isoformat(timespec="milliseconds"),
            "type": event["type"], "track_id": event["track_id"], "class": event["class"],
            "conf": event["conf"], "zone_id": event["zone_id"], "plate": None,
            "snapshot": relative, "clip": clip, "video_ts_s": event["video_ts_s"],
            "source_recording": self.config["source_label"], "event_key": key,
        }
        print("@event " + json.dumps(payload), flush=True)

    def infer(self, model) -> None:
        last_index = -1
        try:
            with (self.directory / "tracks.jsonl").open("w", encoding="utf-8") as tracks:
                while not self.stop.is_set():
                    with self.condition:
                        self.condition.wait_for(
                            lambda: self.stop.is_set() or self.finished.is_set()
                            or self.latest is not None and self.latest[0] != last_index, timeout=1)
                        if self.stop.is_set():
                            break
                        item = self.latest
                        if item is None or item[0] == last_index:
                            if self.finished.is_set():
                                break
                            continue
                    index, ts, frame, _ = item
                    last_index = index
                    started = time.monotonic()
                    result = model.predict(frame, device="cpu", imgsz=640,
                                           conf=min(.075, self.config["confidence"]), verbose=False)[0]
                    self.inference_ms = round((time.monotonic() - started) * 1000, 1)
                    detections = self.track_detections(result, index, ts)
                    found = self.rules.update(detections, ts)
                    for event in found:
                        self.last_event[event["track_id"]] = ts
                        self.save_event(event, frame)
                    self.last_event = {track: stamp for track, stamp in self.last_event.items() if ts - stamp < 3}
                    for detection in detections:
                        detection["alert"] = detection["track_id"] in self.last_event
                        tracks.write(json.dumps(detection) + "\n")
                    tracks.flush()
                    # The shared preview must show the frame these detections came from.
                    shown = frame.copy()
                    for detection in detections:
                        x1, y1, x2, y2 = detection["bbox"]
                        draw_detection(shown, {**detection, "bbox": [
                            x1 * frame.shape[1] / self.rules.width, y1 * frame.shape[0] / self.rules.height,
                            x2 * frame.shape[1] / self.rules.width, y2 * frame.shape[0] / self.rules.height]},
                            alert=detection.get("alert", False))
                    ok, jpeg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        atomic_write(self.directory / "frame.jpg", jpeg.tobytes())
                    self.display_times.append(time.monotonic())
                    with self.condition:
                        self.detections = detections
                        self.detection_ts = ts
                        self.inferred += 1
                        self.inference_times.append(time.monotonic())
        except Exception as exc:
            self.error = str(exc)
            self.stop.set()

    @staticmethod
    def rate(times) -> float:
        return (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0

    def write_status(self) -> None:
        atomic_write(self.directory / "status.json", json.dumps(self.status).encode())

    def run(self) -> None:
        from ultralytics import YOLO
        import torch
        cv2.setNumThreads(2)
        torch.set_num_threads(2)
        self.write_status()
        size = self.probe()
        if self.config.get("browser_input"):
            self.prepare_playback(size)
            if self.stop.is_set():
                self.status.update(state="stopped", preparation=None)
                self.write_status()
                return
            self.status["preparation"] = "Warming the detector…"
            self.write_status()
        model = YOLO(self.config["model"], task="detect")
        if normalize_names(model.names) != TARGET_NAMES:
            raise ValueError("The selected checkpoint must use the seven VEYAKSH traffic classes.")
        model.predict(np.zeros((size[1], size[0], 3), dtype=np.uint8),
                      imgsz=640, device="cpu", verbose=False)
        predictor = model.predictor.model
        if hasattr(predictor, "session"):
            # ORT defaults can oversubscribe the CPU while it also decodes video.
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = predictor.session.get_providers()
            options.enable_mem_pattern = "DmlExecutionProvider" not in providers
            predictor.backend.session = ort.InferenceSession(self.config["model"], options, providers=providers)
            predictor.backend.use_io_binding = False
            model.predict(np.zeros((360, 640, 3), dtype=np.uint8),
                          imgsz=640, device="cpu", verbose=False)
        self.backend = (predictor.session.get_providers()[0] if hasattr(predictor, "session") else "PyTorch CPU")
        self.started = time.monotonic()
        self.config["started_at"] = datetime.now(timezone.utc).isoformat()
        self.status.update(input_ready=True, backend=self.backend,
                           source_fps=round(self.source_fps, 2), duration_s=self.duration,
                           decoder="Browser video" if self.config.get("browser_input") else "GStreamer")
        self.write_status()
        decode_thread = threading.Thread(target=self.decode, args=(size,), daemon=True)
        infer_thread = threading.Thread(target=self.infer, args=(model,), daemon=True)
        decode_thread.start()
        infer_thread.start()
        last_status = 0.0
        try:
            while not self.stop.is_set():
                if (self.directory / "stop").exists():
                    self.stop.set()
                    break
                now = time.monotonic()
                with self.condition:
                    item = self.latest
                    detections = self.detections
                    detection_ts = self.detection_ts
                    inferred = self.inferred
                    inference_times = list(self.inference_times)
                if item is not None:
                    index, ts, frame, received = item
                    age = ts - detection_ts if detection_ts is not None else None
                    self.status = {
                        "state": "running", "source_fps": round(self.source_fps, 2),
                        "display_fps": round(self.rate(self.display_times), 1),
                        "detection_fps": round(self.rate(inference_times), 1), "video_ts_s": round(ts, 3),
                        "duration_s": self.duration, "decoded_frames": self.decoded,
                        "inferred_frames": inferred, "skipped_frames": max(0, index + 1 - inferred),
                        "box_age_s": round(age, 3) if age is not None else None,
                        "boxes": detections, "detection_ts_s": detection_ts, "input_ready": True,
                        "detections": len(detections), "backend": self.backend,
                        "presentation_ts_s": detection_ts,
                        "inference_ms": self.inference_ms,
                        "decoder": self.decoder_name, "decoder_warning": self.decoder_warning,
                        "width": size[0], "height": size[1], "error": self.error,
                    }
                if now - last_status >= 0.04:
                    self.write_status()
                    last_status = now
                if self.finished.is_set():
                    if self.error:
                        raise RuntimeError(self.error)
                    if not infer_thread.is_alive():
                        break
                elif ((item is None and now - self.started > (120 if self.config.get("browser_input") else 45))
                      or (item is not None and now - item[3] > (120 if self.config.get("browser_input") else 15))):
                    raise RuntimeError("Video source stopped delivering frames. Check the connection and restart.")
                self.stop.wait(0.015)
        finally:
            self.stop.set()
            with self.condition:
                self.condition.notify_all()
            if self.decoder is not None and self.decoder.poll() is None:
                self.decoder.terminate()
            decode_thread.join(timeout=3)
            infer_thread.join(timeout=3)
            self.status["state"] = "stopped" if (self.directory / "stop").exists() else "completed"
            self.write_status()
        if self.error:
            raise RuntimeError(self.error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pipeline = LivePipeline(config)
    try:
        pipeline.run()
    except Exception as exc:
        pipeline.status.update(state="failed", error=str(exc))
        pipeline.write_status()
        raise


if __name__ == "__main__":
    main()
