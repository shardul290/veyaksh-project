"""Render tracked detections with rule-event tracks highlighted."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


NORMAL_COLOR = (210, 95, 35)       # subdued blue, BGR
PEDESTRIAN_COLOR = (45, 170, 45)  # green, BGR
WRONG_WAY_COLOR = (35, 35, 235)   # red, BGR
OTHER_EVENT_COLOR = (0, 145, 240) # amber, BGR


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def draw_label(frame: Any, x: int, y: int, text: str, color: tuple[int, int, int], scale: float) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    left = max(0, min(x, frame.shape[1] - text_width - 2))
    baseline_y = y - 3
    if baseline_y - text_height < 0:
        baseline_y = min(frame.shape[0] - baseline - 1, y + text_height + 4)
    # Outline the small text for contrast without drawing a filled label plate.
    cv2.putText(frame, text, (left, baseline_y), font, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (left, baseline_y), font, scale, color, thickness, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, help="Optional frame cap for a quick review preview")
    parser.add_argument("--max-width", type=int, default=1280, help="Maximum output width; default is 1280 (720p)")
    args = parser.parse_args()

    for path in (args.video, args.tracks, args.events):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")

    tracks_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    last_frame_by_track: dict[int, int] = {}
    for row in read_jsonl(args.tracks):
        track_id = row.get("track_id")
        frame_index = row.get("frame")
        bbox = row.get("bbox")
        if (not isinstance(track_id, int) or not isinstance(frame_index, int)
                or not isinstance(bbox, list) or len(bbox) != 4):
            continue
        tracks_by_frame[frame_index].append(row)
        last_frame_by_track[track_id] = max(frame_index, last_frame_by_track.get(track_id, frame_index))

    events = read_jsonl(args.events)
    anomaly_by_track: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for event in events:
        track_id, frame_index = event.get("track_id"), event.get("frame")
        if isinstance(track_id, int) and isinstance(frame_index, int):
            anomaly_by_track[track_id].append((frame_index, str(event.get("type", "anomaly"))))

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        parser.error(f"Could not open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        parser.error(f"Could not read valid video properties: {args.video}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.max_width < 320:
        capture.release()
        parser.error("max-width must be at least 320 pixels")
    resize_scale = min(1.0, args.max_width / width)
    output_size = (max(2, int(width * resize_scale)), max(2, int(height * resize_scale)))
    writer = cv2.VideoWriter(
        str(args.output), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), fps, output_size
    )
    if not writer.isOpened():
        capture.release()
        parser.error(f"Could not create annotated video: {args.output}")

    # Keep the object outline one pixel wide and labels small at every output size.
    label_scale = max(0.26, min(0.34, min(output_size) / 2400.0))
    box_thickness = 1
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            if output_size != (width, height):
                frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
            for row in tracks_by_frame.get(frame_index, ()):
                x1, y1, x2, y2 = (int(round(float(value) * resize_scale)) for value in row["bbox"])
                x1, x2 = max(0, min(output_size[0] - 1, x1)), max(0, min(output_size[0] - 1, x2))
                y1, y2 = max(0, min(output_size[1] - 1, y1)), max(0, min(output_size[1] - 1, y2))
                track_id = row["track_id"]
                object_class = str(row.get("class", "object"))
                compact_class = "ped" if object_class == "pedestrian" else object_class
                track_tag = f" #{track_id}" if track_id is not None else ""
                label = f"{compact_class}{track_tag}"
                anomaly = next(
                    (kind for event_frame, kind in anomaly_by_track.get(track_id, ())
                     if event_frame <= frame_index),
                    None,
                )
                if anomaly == "wrong_way":
                    color = WRONG_WAY_COLOR
                    label = f"WW {label}"
                elif anomaly:
                    color = OTHER_EVENT_COLOR
                    tag = {"encroachment": "ENC", "pedestrian_wrong_path": "WP"}.get(anomaly, "ALERT")
                    label = f"{tag} {label}"
                elif object_class == "pedestrian":
                    color = PEDESTRIAN_COLOR
                else:
                    color = NORMAL_COLOR
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)
                draw_label(frame, x1, y1, label, color, label_scale)
            writer.write(frame)
            frame_index += 1
            if frame_index % 500 == 0:
                print(f"Rendered {frame_index} frames", flush=True)
    finally:
        capture.release()
        writer.release()

    print(
        f"Annotated {frame_index} frames; {len(anomaly_by_track)} tracks flagged; "
        f"{len(events)} rule events; output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
