"""Reuse tracks to prepare synchronized dashboard playback, without YOLO inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .event_engine import read_json, read_tracks, run_engine
from .playback import merge_display_tracks, read_jsonl, register_analysis, render_video


def prepare(video: Path, zones: Path, tracks: Path, output_dir: Path,
            annotated_output: Path | None = None) -> tuple[Path, Path]:
    if annotated_output and annotated_output.resolve() == video.resolve():
        raise ValueError("Annotated output must differ from the source video")
    config = read_json(zones)
    grouped = read_tracks(tracks)
    events = run_engine(grouped, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    classified = output_dir / f"{video.stem}_classified_tracks.jsonl"
    event_path = output_dir / f"{video.stem}_events.jsonl"
    rows = merge_display_tracks(read_jsonl(tracks), [row for group in grouped.values() for row in group])
    for path, records in ((classified, rows), (event_path, events)):
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    manifest = register_analysis(video, classified, event_path, config, metadata=tracks.with_suffix(".meta.json"))
    if annotated_output:
        render_video(video, classified, event_path, annotated_output)
    print(f"Playback prepared: {len(rows)} detections, {len(events)} events; manifest: {manifest}")
    return classified, event_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--zones", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--annotated-output", type=Path)
    args = parser.parse_args()
    prepare(args.video, args.zones, args.tracks, args.output_dir, args.annotated_output)


if __name__ == "__main__":
    main()
