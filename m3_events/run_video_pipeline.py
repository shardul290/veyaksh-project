"""Run a trained VEYAKSH detector with ByteTrack, then evaluate the M3 rules for one video."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="Trained VEYAKSH checkpoint or ONNX export")
    parser.add_argument("--zones", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/m3"))
    parser.add_argument("--annotated-output", type=Path,
                        help="Optional video file to write with per-frame boxes, classes, and track IDs")
    parser.add_argument("--device", default="0", help="GPU index or 'cpu'")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--decoder", choices=("ffmpeg", "gstreamer"), default="ffmpeg",
                        help="Video decoder; GStreamer requires an MSYS2/UCRT64 PyGObject interpreter")
    parser.add_argument("--gstreamer-python", type=Path,
                        default=Path(os.environ["GST_PYTHON"]) if os.environ.get("GST_PYTHON") else None,
                        help="MSYS2 UCRT64 Python containing PyGObject/GStreamer (or set GST_PYTHON)")
    args = parser.parse_args()

    video = args.video.resolve()
    model = args.model.resolve()
    zones = args.zones.resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    if not video.is_file():
        parser.error(f"Video not found: {video}")
    if not model.is_file():
        parser.error(f"Model checkpoint not found: {model}")
    if not zones.is_file():
        parser.error(f"Zone config not found: {zones}")
    if args.imgsz < 32 or args.stride < 1 or not 0 <= args.conf <= 1:
        parser.error("imgsz must be >=32, stride >=1, and conf in [0,1]")

    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = output_dir / f"{video.stem}_tracks.jsonl"
    events = output_dir / f"{video.stem}_events.jsonl"
    export_command = [sys.executable, str(PROJECT_ROOT / "m2_detect_track" / "export_tracks_jsonl.py"),
         "--video", str(video), "--model", str(model), "--kind", "veyaksh",
         "--output", str(tracks), "--device", args.device, "--imgsz", str(args.imgsz),
         "--conf", str(args.conf), "--stride", str(args.stride), "--zones", str(zones)]
    export_command.extend(["--decoder", args.decoder])
    if args.gstreamer_python is not None:
        export_command.extend(["--gstreamer-python", str(args.gstreamer_python)])
    commands = [
        export_command,
        [sys.executable, "-m", "m3_events.prepare_playback", "--video", str(video),
         "--tracks", str(tracks), "--zones", str(zones), "--output-dir", str(output_dir)],
    ]
    try:
        for command in commands:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Pipeline step failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode or 1
    if args.annotated_output:
        annotated_output = args.annotated_output.resolve()
        render_command = [
            sys.executable, "-m", "m3_events.render_annotated_video",
            "--video", str(video), "--tracks", str(tracks), "--events", str(events),
            "--output", str(annotated_output),
        ]
        try:
            subprocess.run(render_command, cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Annotation step failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
            return exc.returncode or 1
    print(f"Tracking export: {tracks}")
    print(f"Rule events: {events}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
