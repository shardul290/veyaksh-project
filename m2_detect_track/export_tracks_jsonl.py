"""Export ByteTrack detections to one JSONL row per target detection.

Example:
  python m2_detect_track/export_tracks_jsonl.py --video data/test.mp4 \
    --model yolo11n.pt --kind coco --output tracks/test_yolo11n.jsonl
  python m2_detect_track/export_tracks_jsonl.py --video data/test.mp4 \
    --model models/best.pt --kind veyaksh --output tracks/test_best.jsonl

COCO cannot distinguish auto/e-rickshaw/cart; its export includes only classes
with defensible mappings to the VEYAKSH taxonomy. Tracked boxes with no assigned
track ID yet are retained with track_id=null.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import torch
from m3_events.event_engine import load_zones, read_json
from m3_events.playback import draw_detection
from m3_events.rules.pedestrian_path import PedestrianPathRule
from ultralytics import YOLO

TARGET_NAMES = {
    0: "two_wheeler", 1: "auto", 2: "e_rickshaw", 3: "car",
    4: "bus_truck", 5: "cart", 6: "pedestrian",
}
COCO_TO_TARGET = {
    "bicycle": "two_wheeler", "motorcycle": "two_wheeler",
    "car": "car", "bus": "bus_truck", "truck": "bus_truck",
    "person": "pedestrian",
}


def normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--zones", type=Path, help="Calibrated pedestrian zones for right/wrong path overlays")
    parser.add_argument("--kind", required=True, choices=("coco", "veyaksh"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--annotated-output", type=Path,
                        help="Optional MP4 with tracked boxes and labels drawn on every processed frame")
    parser.add_argument("--device", default="0", help="GPU index or cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--cpu-threads", type=int, default=2, help="CPU inference threads; avoids oversubscribing laptop CPUs")
    parser.add_argument("--rotate-180", action="store_true",
                        help="Correct an upside-down processing proxy after decoding; leave unset for the original camera recording")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth source frame")
    args = parser.parse_args()

    if args.stride < 1 or args.imgsz < 32 or args.cpu_threads < 1 or not 0 <= args.conf <= 1:
        parser.error("stride must be >=1, imgsz >=32 and conf in [0,1]")

    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    if not args.model.is_file():
        parser.error(f"Model checkpoint not found: {args.model}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        parser.error(f"Could not read a valid FPS from {args.video}")

    path_rule = None
    zone_config = None
    if args.zones:
        zone_config = read_json(args.zones)
        width, height, _, _ = load_zones(zone_config)
        if zone_config.get("pedestrian_zones") and (width, height) != (frame_width, frame_height):
            parser.error("Zone frame_size must match the source video dimensions; recalibrate this camera")
        path_rule = PedestrianPathRule(width, height, zone_config.get("pedestrian_zones", []),
            {"max_gap_s": zone_config.get("max_track_gap_s", 1.0), **zone_config.get("pedestrian_path", {})})

    annotated_writer = None
    annotated_size = (frame_width, frame_height)
    if args.annotated_output:
        if frame_width <= 0 or frame_height <= 0:
            parser.error(f"Could not read video dimensions from {args.video}")
        resize_scale = min(1.0, 1920 / frame_width)
        annotated_size = (
            max(2, int(frame_width * resize_scale) // 2 * 2),
            max(2, int(frame_height * resize_scale) // 2 * 2),
        )
        args.annotated_output.parent.mkdir(parents=True, exist_ok=True)
        output_fps = fps / args.stride
        annotated_writer = cv2.VideoWriter(
            str(args.annotated_output), cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps, annotated_size,
        )
        if not annotated_writer.isOpened():
            parser.error(f"Could not create annotated video: {args.annotated_output}")

    cv2.setNumThreads(2)
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    model = YOLO(str(args.model))
    model_names = normalize_names(model.names)
    if args.kind == "coco":
        class_map = {cid: COCO_TO_TARGET[name] for cid, name in model_names.items() if name in COCO_TO_TARGET}
    else:
        canonical = {name: name for name in TARGET_NAMES.values()}
        aliases = {
            "bike": "two_wheeler", "motorcycle": "two_wheeler", "motorbike": "two_wheeler",
            "rickshaw": "auto", "auto_rickshaw": "auto", "erickshaw": "e_rickshaw",
            "e_rickshaw": "e_rickshaw", "bus": "bus_truck", "truck": "bus_truck", "tempo": "bus_truck",
            "person": "pedestrian",
        }
        class_map = {cid: canonical.get(name, aliases.get(name)) for cid, name in model_names.items()}
        class_map = {cid: name for cid, name in class_map.items() if name in TARGET_NAMES.values()}
        if not class_map:
            parser.error(f"No VEYAKSH classes found in model names: {model_names}")

    processed = 0
    rows_written = 0
    metadata_path = args.output.with_suffix(".meta.json")
    metadata_path.unlink(missing_ok=True)
    try:
        with args.output.open("w", encoding="utf-8") as out:
            def tracked_frames():
                # Own the decoder to bound 4K buffering and retain true source
                # frame indices even when inference samples every Nth frame.
                capture = cv2.VideoCapture(str(args.video), cv2.CAP_FFMPEG,
                                           [cv2.CAP_PROP_N_THREADS, 2])
                if not capture.isOpened():
                    capture.release()
                    raise ValueError(f"Could not open video decoder: {args.video}")
                source_index = -1
                try:
                    while capture.grab():
                        source_index += 1
                        if source_index % args.stride:
                            continue
                        ok, frame = capture.retrieve()
                        if not ok:
                            raise ValueError(f"Could not decode source frame {source_index}")
                        if args.rotate_180:
                            frame = cv2.rotate(frame, cv2.ROTATE_180)
                        source_ts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        if not math.isfinite(source_ts) or source_ts < 0 or (source_index > 0 and source_ts == 0):
                            source_ts = source_index / fps
                        result = model.track(source=frame, tracker="bytetrack.yaml", persist=True,
                                             verbose=False, device=args.device, imgsz=args.imgsz,
                                             conf=args.conf, classes=sorted(class_map))[0]
                        if args.device == "cpu":
                            torch.set_num_threads(args.cpu_threads)
                        yield source_index, source_ts, result
                finally:
                    capture.release()

            for frame_index, source_ts, result in tracked_frames():
                processed += 1
                if annotated_writer is not None:
                    annotated_frame = result.orig_img.copy()
                def write_annotated():
                    if annotated_writer is not None:
                        shown = annotated_frame
                        if (shown.shape[1], shown.shape[0]) != annotated_size:
                            shown = cv2.resize(shown, annotated_size, interpolation=cv2.INTER_AREA)
                        annotated_writer.write(shown)
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    write_annotated()
                    if processed % 50 == 0:
                        print(f"Processed {processed} sampled frames; wrote {rows_written} detections", flush=True)
                    continue
                xyxy = boxes.xyxy.cpu().tolist()
                cls_ids = boxes.cls.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                track_ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                for coords, class_id, conf, track_id in zip(xyxy, cls_ids, confs, track_ids):
                    class_id = int(class_id)
                    target = class_map.get(class_id)
                    if target is None:
                        continue
                    row = {
                        "frame": frame_index,
                        "ts": round(source_ts, 6),
                        "track_id": int(track_id) if track_id is not None else None,
                        "class": target,
                        "bbox": [round(float(v), 2) for v in coords],
                        "conf": round(float(conf), 6),
                    }
                    if path_rule and target == "pedestrian":
                        decision, _ = path_rule.update(row)
                        row["path"] = decision
                    if annotated_writer is not None:
                        draw_detection(annotated_frame, row)
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                    rows_written += 1
                write_annotated()
                if processed % 50 == 0:
                    print(f"Processed {processed} sampled frames; wrote {rows_written} detections", flush=True)
    finally:
        if annotated_writer is not None:
            annotated_writer.release()

    print(f"Done: {args.video} | sampled frames={processed} | detections={rows_written} | output={args.output}")
    source_stat = args.video.stat()
    metadata_path.write_text(json.dumps({
        "complete": frame_count > 0 and processed >= frame_count // args.stride,
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns, "frame_count": frame_count,
        "fps": fps, "stride": args.stride, "processed_frames": processed,
        "rotation_applied_deg": 180 if args.rotate_180 else 0,
    }), encoding="utf-8")
    if args.annotated_output:
        print(f"Annotated video: {args.annotated_output}")


if __name__ == "__main__":
    main()
