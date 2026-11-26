"""Show live COCO object detections from a local RTSP webcam feed."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from nanodet import NanoDet  # noqa: E402
from nanodet_demo import classes as COCO_CLASSES, letterbox, unletterbox  # noqa: E402


DEFAULT_SOURCE = "rtsp://127.0.0.1:8555/webcam"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def latest_put(target: queue.Queue, value: Any) -> bool:
    """Replace a queued stale frame so inference stays close to live."""
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


def preview_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    ratio = min(1.0, max_width / width)
    return max(2, int(width * ratio) // 2 * 2), max(2, int(height * ratio) // 2 * 2)


def draw_object(frame, detection: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in detection["bbox"]]
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
    color = (70, 220, 100) if detection["class"] == "person" else (50, 200, 245)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    label = f"{detection['class']} {detection['conf']:.2f}"
    font_scale = 0.42
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    text_x = max(0, min(x1, width - text_width - 6))
    text_y = max(text_height + 5, y1 - 4)
    cv2.rectangle(frame, (text_x, text_y - text_height - 4),
                  (text_x + text_width + 6, text_y + baseline), (20, 20, 20), -1)
    cv2.putText(frame, label, (text_x + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="RTSP camera URL")
    parser.add_argument("--model", type=Path,
                        default=Path("models/object_detection_nanodet_2022nov.onnx"))
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--nms", type=float, default=0.6)
    parser.add_argument("--max-box-age", type=float, default=0.75)
    args = parser.parse_args()

    model_path = project_path(args.model)
    if not model_path.is_file():
        parser.error(f"Model not found: {model_path}")
    if args.max_width < 320 or not 0 <= args.conf <= 1 or not 0 < args.nms <= 1:
        parser.error("max-width must be >=320, conf must be in [0,1], and nms must be in (0,1]")
    if args.max_box_age <= 0:
        parser.error("max-box-age must be positive")
    if args.source.lower().startswith(("rtsp\\://", "rtsps\\://")):
        print("Removing an extra backslash from the RTSP URL.", flush=True)
        args.source = args.source.replace("\\://", "://", 1)

    cv2.setNumThreads(2)
    print(f"Loading COCO object detector: {model_path}", flush=True)
    detector = NanoDet(
        str(model_path), prob_threshold=args.conf, iou_threshold=args.nms,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV, target_id=cv2.dnn.DNN_TARGET_CPU,
    )
    print(f"Connecting to {args.source}", flush=True)
    capture = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000,
    ])
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open RTSP stream: {args.source}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError(f"RTSP connected but returned no video frame: {args.source}")

    original_height, original_width = first_frame.shape[:2]
    frame_size = preview_size(original_width, original_height, args.max_width)

    def resize_frame(frame):
        if frame.shape[1] == frame_size[0] and frame.shape[0] == frame_size[1]:
            return frame
        return cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)

    display_queue: queue.Queue = queue.Queue(maxsize=2)
    inference_queue: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    capture_done = threading.Event()
    inference_done = threading.Event()
    state_lock = threading.Lock()
    state: dict[str, Any] = {
        "detections": [], "detections_ts": None, "revision": 0,
        "inference_fps": 0.0, "inference_count": 0, "dropped": 0, "error": None,
    }

    first_packet = (0, time.monotonic(), resize_frame(first_frame))
    latest_put(display_queue, first_packet)
    latest_put(inference_queue, first_packet)

    def capture_frames() -> None:
        frame_index = 1
        try:
            while not stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    if stop_event.is_set():
                        break
                    raise RuntimeError("RTSP camera stopped returning frames")
                packet = (frame_index, time.monotonic(), resize_frame(frame))
                frame_index += 1
                if latest_put(inference_queue, packet):
                    with state_lock:
                        state["dropped"] += 1
                latest_put(display_queue, packet)
        except Exception as exc:
            with state_lock:
                state["error"] = f"RTSP capture failed: {exc}"
            stop_event.set()
        finally:
            capture.release()
            capture_done.set()

    def infer_frames() -> None:
        durations: deque[float] = deque(maxlen=24)
        try:
            while not stop_event.is_set() or not inference_queue.empty():
                try:
                    packet = inference_queue.get(timeout=0.1)
                except queue.Empty:
                    if capture_done.is_set():
                        break
                    continue
                frame_index, frame_ts, frame = packet
                started = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                model_frame, letterbox_scale = letterbox(rgb, (416, 416))
                predictions = detector.infer(model_frame)
                detections: list[dict[str, Any]] = []
                for row in predictions:
                    class_id = int(row[-1])
                    if 0 <= class_id < len(COCO_CLASSES):
                        bbox = unletterbox(row[:4], frame.shape[:2], letterbox_scale)
                        detections.append({
                            "frame": frame_index, "class": COCO_CLASSES[class_id],
                            "bbox": [float(value) for value in bbox], "conf": float(row[-2]),
                        })
                elapsed = max(time.perf_counter() - started, 1e-9)
                durations.append(elapsed)
                measured_fps = len(durations) / sum(durations)
                with state_lock:
                    state["detections"] = detections
                    state["detections_ts"] = frame_ts
                    state["revision"] += 1
                    state["inference_count"] += 1
                    state["inference_fps"] = measured_fps
        except Exception as exc:
            with state_lock:
                state["error"] = f"Object detector failed: {exc}"
            stop_event.set()
        finally:
            inference_done.set()

    capture_thread = threading.Thread(target=capture_frames, name="rtsp-camera-reader", daemon=True)
    inference_thread = threading.Thread(target=infer_frames, name="nanodet-inference", daemon=True)
    capture_thread.start()
    inference_thread.start()
    print(f"Camera stream: {original_width}x{original_height}; preview: {frame_size[0]}x{frame_size[1]}",
          flush=True)
    print("COCO model detects people and common objects. Press Q or Esc to stop.", flush=True)

    window_name = "RTSP webcam object detection demo"
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
                frame_index, frame_ts, frame = last_packet
                shown = frame.copy()
                box_age = frame_ts - detections_ts if detections_ts is not None else None
                if box_age is not None and 0 <= box_age <= args.max_box_age:
                    for detection in detections:
                        draw_object(shown, detection)
                age_text = f"{box_age:.2f}s" if box_age is not None and box_age >= 0 else "waiting"
                status = (f"LIVE RTSP | NanoDet {inference_fps:.1f} FPS | box age {age_text} | "
                          f"inference frames skipped {dropped}")
                cv2.putText(shown, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, (15, 15, 15), 3, cv2.LINE_AA)
                cv2.putText(shown, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, (245, 245, 245), 1, cv2.LINE_AA)
                cv2.imshow(window_name, shown)
                last_revision = revision
                last_packet = (-1, frame_ts, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            if capture_done.is_set() and inference_done.is_set() and display_queue.empty():
                break
    finally:
        stop_event.set()
        capture_thread.join(timeout=4)
        inference_thread.join(timeout=4)
        cv2.destroyAllWindows()

    with state_lock:
        error = state["error"]
        inference_count = state["inference_count"]
        dropped = state["dropped"]
    if error:
        return 1
    print(f"Preview ended. Inference calls: {inference_count}; pending frames skipped: {dropped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
