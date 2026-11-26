"""Play a saved clip at source timing while YOLO analyzes the newest frames.

The GStreamer decoder and display run independently from model inference. A
one-frame inference queue discards stale pending frames when YOLO falls behind,
so preview latency stays bounded. This is a preview tool, not a full-frame
offline annotator; use m3_events.run_video_pipeline to save a complete result.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from m2_detect_track.export_tracks_jsonl import TARGET_NAMES, iter_gstreamer_frames, normalize_names
from m3_events.playback import draw_detection
from ultralytics import YOLO


ALIASES = {
    "bike": "two_wheeler", "motorcycle": "two_wheeler", "motorbike": "two_wheeler",
    "rickshaw": "auto", "auto_rickshaw": "auto", "erickshaw": "e_rickshaw",
    "bus": "bus_truck", "truck": "bus_truck", "tempo": "bus_truck",
    "person": "pedestrian",
}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def target_class_map(model: YOLO) -> dict[int, str]:
    names = normalize_names(model.names)
    canonical = {name: name for name in TARGET_NAMES.values()}
    mapping = {class_id: canonical.get(name, ALIASES.get(name)) for class_id, name in names.items()}
    return {class_id: target for class_id, target in mapping.items() if target in TARGET_NAMES.values()}


def latest_put(target: queue.Queue, value: Any) -> bool:
    """Put value, replacing the queued stale item if the queue is full."""
    try:
        target.put_nowait(value)
        return False
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        target.put_nowait(value)
        return True


def video_properties(video: Path | str, fallback_fps: float) -> tuple[float, int, int]:
    if isinstance(video, str) and video.lower().startswith(("rtsp://", "rtsps://")):
        capture = cv2.VideoCapture(video, cv2.CAP_FFMPEG)
    else:
        capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot read video properties: {video}")
    first_frame = None
    if isinstance(video, str) and video.lower().startswith(("rtsp://", "rtsps://")):
        ok, first_frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"RTSP stream opened but did not deliver a video frame: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if first_frame is not None:
        height = height or int(first_frame.shape[0])
        width = width or int(first_frame.shape[1])
    if fps <= 0:
        fps = fallback_fps
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Video has invalid FPS or dimensions: {video}")
    return fps, width, height


def preview_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    ratio = min(1.0, max_width / width)
    return max(2, int(width * ratio) // 2 * 2), max(2, int(height * ratio) // 2 * 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", "--video", dest="source", default="data/incoming/1.1.mp4",
                        help="Local video path or rtsp:// URL")
    parser.add_argument("--model", type=Path, default=Path("models/yolo26s_traffic.onnx"))
    parser.add_argument("--gstreamer-python", type=Path,
                        default=Path(os.environ["GST_PYTHON"]) if os.environ.get("GST_PYTHON") else None)
    parser.add_argument("--device", default="cpu",
                        help="Keep cpu here; ONNX Runtime selected DirectML in this setup")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--max-width", type=int, default=1280,
                        help="Scale 4K frames before sending them to the preview process")
    parser.add_argument("--fps-fallback", type=float, default=30.0,
                        help="FPS to display/use for timing if an RTSP server does not report it")
    parser.add_argument("--max-box-age", type=float, default=0.75,
                        help="Stop showing a detection overlay older than this many seconds")
    parser.add_argument("--rotate-180", action="store_true",
                        help="Rotate this camera recording 180 degrees before detection and display")
    args = parser.parse_args()

    source_text = str(args.source)
    is_rtsp = source_text.lower().startswith(("rtsp://", "rtsps://"))
    video: Path | str = source_text if is_rtsp else project_path(source_text)
    model_path = project_path(args.model)
    gst_python = project_path(args.gstreamer_python) if args.gstreamer_python else None
    if not is_rtsp and not video.is_file():
        parser.error(f"Video not found: {video}")
    if not model_path.is_file():
        parser.error(f"Model not found: {model_path}")
    if gst_python is None or not gst_python.is_file():
        parser.error("Pass --gstreamer-python or set GST_PYTHON to the MSYS2 UCRT64 python.exe")
    if args.imgsz < 32 or args.max_width < 320 or args.max_box_age <= 0 or not 0 <= args.conf <= 1:
        parser.error("imgsz must be >=32, max-width >=320, max-box-age >0, and conf in [0,1]")

    source_fps, source_width, source_height = video_properties(video, args.fps_fallback)
    frame_size = preview_size(source_width, source_height, args.max_width)
    print(f"Source: {source_width}x{source_height} at {source_fps:.2f} FPS")
    print(f"Preview decode size: {frame_size[0]}x{frame_size[1]}; detections use newest-frame queue")
    if args.rotate_180:
        print("Applying 180-degree orientation correction")
    print("Loading model. Press Q or Esc in the preview window to stop.", flush=True)
    model = YOLO(str(model_path))
    class_map = target_class_map(model)
    if not class_map:
        parser.error(f"No VEYAKSH vehicle classes found in model names: {model.names}")

    display_queue: queue.Queue = queue.Queue(maxsize=2)
    inference_queue: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    decoder_done = threading.Event()
    inference_done = threading.Event()
    state_lock = threading.Lock()
    state: dict[str, Any] = {
        "detections": [], "detections_ts": None, "revision": 0,
        "inference_fps": 0.0, "inference_count": 0, "dropped": 0, "error": None,
    }

    def decode_frames() -> None:
        frames = iter_gstreamer_frames(
            video, gst_python, frame_size, realtime=True, output_size=frame_size
        )
        try:
            for source_index, source_ts, frame in frames:
                if stop_event.is_set():
                    break
                if args.rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                if source_ts is None:
                    source_ts = source_index / source_fps
                packet = (source_index, source_ts, frame)
                if latest_put(inference_queue, packet):
                    with state_lock:
                        state["dropped"] += 1
                latest_put(display_queue, packet)
        except Exception as exc:
            with state_lock:
                state["error"] = f"GStreamer decode failed: {exc}"
            stop_event.set()
        finally:
            frames.close()
            decoder_done.set()

    def infer_frames() -> None:
        durations: deque[float] = deque(maxlen=24)
        try:
            while not stop_event.is_set() or not inference_queue.empty():
                try:
                    packet = inference_queue.get(timeout=0.1)
                except queue.Empty:
                    if decoder_done.is_set():
                        break
                    continue
                source_index, source_ts, frame = packet
                started = time.perf_counter()
                result = model.track(
                    source=frame, tracker="bytetrack.yaml", persist=True, verbose=False,
                    device=args.device, imgsz=args.imgsz, conf=args.conf,
                    classes=sorted(class_map),
                )[0]
                boxes = result.boxes
                detections: list[dict[str, Any]] = []
                if boxes is not None and len(boxes):
                    coords = boxes.xyxy.cpu().tolist()
                    classes = boxes.cls.cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    track_ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(coords)
                    for bbox, class_id, confidence, track_id in zip(coords, classes, confidences, track_ids):
                        target_class = class_map.get(int(class_id))
                        if target_class is None:
                            continue
                        detections.append({
                            "frame": int(source_index), "ts": float(source_ts),
                            "track_id": int(track_id) if track_id is not None else None,
                            "class": target_class, "bbox": [float(value) for value in bbox],
                            "conf": float(confidence),
                        })
                elapsed = max(time.perf_counter() - started, 1e-9)
                durations.append(elapsed)
                measured_fps = len(durations) / sum(durations)
                with state_lock:
                    state["detections"] = detections
                    state["detections_ts"] = float(source_ts)
                    state["revision"] += 1
                    state["inference_count"] += 1
                    state["inference_fps"] = measured_fps
        except Exception as exc:
            with state_lock:
                state["error"] = f"Model inference failed: {exc}"
            stop_event.set()
        finally:
            inference_done.set()

    decoder_thread = threading.Thread(target=decode_frames, name="gstreamer-reader", daemon=True)
    inference_thread = threading.Thread(target=infer_frames, name="yolo-inference", daemon=True)
    decoder_thread.start()
    inference_thread.start()

    window_name = "VEYAKSH live detection preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    last_packet: tuple[int, float, Any] | None = None
    last_revision = -1
    try:
        while True:
            while True:
                try:
                    last_packet = display_queue.get_nowait()
                except queue.Empty:
                    break

            with state_lock:
                detections = list(state["detections"])
                detections_ts = state["detections_ts"]
                revision = int(state["revision"])
                inference_fps = float(state["inference_fps"])
                dropped = int(state["dropped"])
                error = state["error"]

            if error:
                print(error, flush=True)
                break
            if last_packet is not None and (last_packet[0] != -1 or revision != last_revision):
                source_index, source_ts, frame = last_packet
                shown = frame.copy()
                box_age = source_ts - detections_ts if detections_ts is not None else None
                if box_age is not None and 0 <= box_age <= args.max_box_age:
                    for detection in detections:
                        draw_detection(shown, detection)
                age_text = f"{box_age:.2f}s" if box_age is not None and box_age >= 0 else "waiting"
                status = (f"video {source_fps:.0f} FPS | YOLO {inference_fps:.1f} FPS | "
                          f"box age {age_text} | inference frames skipped {dropped}")
                cv2.putText(shown, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (20, 20, 20), 3, cv2.LINE_AA)
                cv2.putText(shown, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(window_name, shown)
                last_revision = revision
                last_packet = (-1, source_ts, frame)  # Keep the latest picture while waiting for the next frame.

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            if decoder_done.is_set() and inference_done.is_set() and display_queue.empty():
                break
    finally:
        stop_event.set()
        decoder_thread.join(timeout=3)
        inference_thread.join(timeout=3)
        cv2.destroyAllWindows()

    with state_lock:
        final_error = state["error"]
        total_inferences = state["inference_count"]
        total_dropped = state["dropped"]
    if final_error:
        return 1
    print(f"Preview ended. Inference calls: {total_inferences}; pending frames skipped: {total_dropped}.")
    print("This preview does not save a video. Run the offline pipeline afterward for a complete annotated file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
