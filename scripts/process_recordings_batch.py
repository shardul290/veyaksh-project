"""Process all videos in data/incoming sequentially and save annotated previews.

Each video needs an entry in configs/recording_zone_map.json. A zone path runs
the full tracking + M3 event-analysis pipeline; a null value runs detections and
tracking only, because spatial rules need zones calibrated for that camera view.

Run from the repository root with the project's .venv active:
  python scripts/process_recordings_batch.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def read_zone_map(path: Path) -> dict[str, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read zone map {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("The zone map must be a JSON object keyed by video filename stem")
    result: dict[str, str | None] = {}
    for stem, zone in value.items():
        if not isinstance(stem, str) or not stem:
            raise ValueError("Zone map keys must be non-empty video filename stems")
        if zone is not None and not isinstance(zone, str):
            raise ValueError(f"Zone map value for {stem!r} must be a path or null")
        result[stem] = zone
    return result


def file_signature(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def job_signature(job: dict[str, Any], model: Path, gst_python: Path | None,
                  args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source": file_signature(job["video"]),
        "model": file_signature(model),
        "zones": file_signature(job["zones"]),
        "gstreamer_python": file_signature(gst_python),
        "mode": job["mode"],
        "settings": {
            "device": args.device, "imgsz": args.imgsz, "conf": args.conf,
            "stride": args.stride, "decoder": args.decoder,
        },
    }


def completed(job: dict[str, Any], signature: dict[str, Any]) -> bool:
    tracks = job["tracks"]
    metadata = tracks.with_suffix(".meta.json")
    annotated = job["annotated"]
    marker = job["output_dir"] / ".batch_complete.json"
    if not tracks.is_file() or not metadata.is_file() or not annotated.is_file() or not marker.is_file():
        return False
    try:
        details = json.loads(metadata.read_text(encoding="utf-8"))
        completed_signature = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (completed_signature != signature or not details.get("complete")
            or details.get("source_size") != signature["source"]["size"]
            or details.get("source_mtime_ns") != signature["source"]["mtime_ns"]):
        return False
    return job["mode"] == "detect_track_only" or job["events"].is_file()


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps({
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recordings": records,
    }, indent=2), encoding="utf-8")
    temp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/incoming", help="Folder containing source clips")
    parser.add_argument("--video", help="Process only this filename or filename stem, e.g. 1.1.mp4")
    parser.add_argument("--model", default="models/yolo26s_traffic.onnx", help="ONNX model (DirectML on this laptop)")
    parser.add_argument("--zone-map", default="configs/recording_zone_map.json")
    parser.add_argument("--output-dir", default="runs/all_recordings_winml")
    parser.add_argument("--device", default="cpu", help="Keep cpu here; ONNX Runtime selects DirectML in this setup")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--stride", type=int, default=1,
                        help="1 analyzes every frame; 2 is faster but samples half the frames")
    parser.add_argument("--decoder", choices=("ffmpeg", "gstreamer"), default="ffmpeg")
    parser.add_argument("--gstreamer-python", help="MSYS2 UCRT64 python.exe used for GStreamer decoding")
    parser.add_argument("--resume", action="store_true",
                        help="Skip clips whose source, model, zones, settings, and outputs still match")
    parser.add_argument("--require-zones", action="store_true",
                        help="Stop before processing if any input lacks a calibrated zone config")
    parser.add_argument("--dry-run", action="store_true", help="List the queue without processing clips")
    args = parser.parse_args()

    if args.imgsz < 32 or args.stride < 1 or not 0 <= args.conf <= 1:
        parser.error("imgsz must be >=32, stride >=1, and conf in [0,1]")

    input_dir = project_path(args.input_dir)
    model = project_path(args.model)
    zone_map_path = project_path(args.zone_map)
    output_root = project_path(args.output_dir)
    gst_python = project_path(args.gstreamer_python) if args.gstreamer_python else None
    if not input_dir.is_dir():
        parser.error(f"Input folder not found: {input_dir}")
    if not model.is_file():
        parser.error(f"Model not found: {model}")
    if not zone_map_path.is_file():
        parser.error(f"Zone map not found: {zone_map_path}")
    if args.decoder == "gstreamer":
        if gst_python is None and os.environ.get("GST_PYTHON"):
            gst_python = project_path(os.environ["GST_PYTHON"])
        if gst_python is None or not gst_python.is_file():
            parser.error("GStreamer mode needs --gstreamer-python or a valid GST_PYTHON environment variable")

    try:
        zone_map = read_zone_map(zone_map_path)
    except ValueError as exc:
        parser.error(str(exc))

    videos = sorted(
        (path for path in input_dir.iterdir()
         if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )
    if args.video:
        requested = Path(args.video)
        requested_name = requested.name.casefold()
        requested_stem = requested.stem.casefold()
        videos = [path for path in videos
                  if path.name.casefold() == requested_name or path.stem.casefold() == requested_stem]
    if not videos:
        if args.video:
            parser.error(f"No supported video named {args.video!r} found in {input_dir}")
        parser.error(f"No supported video files found in {input_dir}")

    jobs: list[dict[str, Any]] = []
    missing_zones: list[str] = []
    for video in videos:
        zone_value = zone_map.get(video.stem)
        zones = project_path(zone_value) if zone_value else None
        if zones and not zones.is_file():
            parser.error(f"Zone config for {video.name} does not exist: {zones}")
        if zones is None:
            missing_zones.append(video.name)
        output_dir = output_root / video.stem
        jobs.append({
            "video": video,
            "zones": zones,
            "mode": "full_analysis" if zones else "detect_track_only",
            "output_dir": output_dir,
            "tracks": output_dir / f"{video.stem}_tracks.jsonl",
            "events": output_dir / f"{video.stem}_events.jsonl",
            "annotated": output_dir / f"{video.stem}_annotated.mp4",
        })

    if args.require_zones and missing_zones:
        parser.error("Calibrate/map a separate zone config for these camera views first: "
                     + ", ".join(missing_zones))

    if args.dry_run:
        print(f"Found {len(jobs)} recording(s); output root: {output_root}")
        for job in jobs:
            print(f"  {job['video'].name}: {job['mode']} -> {job['output_dir']}")
        return 0

    records: list[dict[str, Any]] = []
    summary_path = output_root / "batch_summary.json"
    failures = 0
    for index, job in enumerate(jobs, 1):
        video: Path = job["video"]
        mode: str = job["mode"]
        print(f"\n[{index}/{len(jobs)}] {video.name} ({mode})", flush=True)
        signature = job_signature(job, model, gst_python, args)
        if args.resume and completed(job, signature):
            print("  Skipped: complete outputs already match this source", flush=True)
            records.append({"video": video.name, "mode": mode, "status": "skipped_complete"})
            write_summary(summary_path, records)
            continue

        job["output_dir"].mkdir(parents=True, exist_ok=True)
        (job["output_dir"] / ".batch_complete.json").unlink(missing_ok=True)
        common = [
            "--video", str(video), "--model", str(model),
            "--device", args.device, "--imgsz", str(args.imgsz),
            "--conf", str(args.conf), "--stride", str(args.stride),
            "--decoder", args.decoder,
        ]
        if gst_python is not None:
            common.extend(["--gstreamer-python", str(gst_python)])
        if job["zones"]:
            command = [
                sys.executable, "-m", "m3_events.run_video_pipeline",
                *common, "--zones", str(job["zones"]),
                "--output-dir", str(job["output_dir"]),
                "--annotated-output", str(job["annotated"]),
            ]
        else:
            command = [
                sys.executable, "-m", "m2_detect_track.export_tracks_jsonl",
                *common, "--kind", "veyaksh",
                "--output", str(job["tracks"]),
                "--annotated-output", str(job["annotated"]),
            ]

        started = time.monotonic()
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            elapsed = round(time.monotonic() - started, 2)
            (job["output_dir"] / ".batch_complete.json").write_text(
                json.dumps(signature, indent=2), encoding="utf-8"
            )
            print(f"  Finished in {elapsed:.1f}s", flush=True)
            records.append({
                "video": video.name, "mode": mode, "status": "complete",
                "elapsed_seconds": elapsed,
                "tracks": str(job["tracks"].relative_to(PROJECT_ROOT)),
                "events": str(job["events"].relative_to(PROJECT_ROOT)) if job["zones"] else None,
                "annotated_video": str(job["annotated"].relative_to(PROJECT_ROOT)),
            })
        except subprocess.CalledProcessError as exc:
            failures += 1
            elapsed = round(time.monotonic() - started, 2)
            print(f"  FAILED with exit code {exc.returncode}; continuing to next clip", file=sys.stderr)
            records.append({
                "video": video.name, "mode": mode, "status": "failed",
                "elapsed_seconds": elapsed, "exit_code": exc.returncode,
            })
        write_summary(summary_path, records)

    succeeded = sum(record["status"] == "complete" for record in records)
    skipped = sum(record["status"] == "skipped_complete" for record in records)
    analyzed = sum(record["status"] in {"complete", "skipped_complete"}
                   and record["mode"] == "full_analysis" for record in records)
    print(f"\nBatch finished: {succeeded} processed, {skipped} skipped, "
          f"{failures} failed; {analyzed} clips have zone-based event analysis.")
    print(f"Summary: {summary_path}")
    if missing_zones:
        print("These clips received detections/tracking and annotated videos only; "
              "calibrate their camera-specific zones to enable event analysis: "
              + ", ".join(missing_zones))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
