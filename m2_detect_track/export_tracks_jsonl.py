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
from pathlib import Path

import cv2
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
    parser.add_argument("--kind", required=True, choices=("coco", "veyaksh"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0", help="GPU index or cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth source frame")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    if not args.model.is_file():
        parser.error(f"Model checkpoint not found: {args.model}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        parser.error(f"Could not read a valid FPS from {args.video}")

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
    with args.output.open("w", encoding="utf-8") as out:
        results = model.track(
            source=str(args.video), tracker="bytetrack.yaml", persist=True,
            stream=True, verbose=False, device=args.device, imgsz=args.imgsz,
            conf=args.conf, classes=sorted(class_map), vid_stride=args.stride,
        )
        for result in results:
            frame_index = processed * args.stride
            processed += 1
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
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
                    "ts": round(frame_index / fps, 6),
                    "track_id": int(track_id) if track_id is not None else None,
                    "class": target,
                    "bbox": [round(float(v), 2) for v in coords],
                    "conf": round(float(conf), 6),
                }
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                rows_written += 1
            if processed % 500 == 0:
                print(f"Processed {processed} sampled frames; wrote {rows_written} detections", flush=True)

    print(f"Done: {args.video} | sampled frames={processed} | detections={rows_written} | output={args.output}")


if __name__ == "__main__":
    main()
