# M3 event rules

`event_engine.py` consumes one video's M2/ByteTrack JSONL export. It evaluates each tracked vehicle against that camera's lane and no-parking polygons, then writes the intermediate event JSONL consumed by `m5_edge_agent/send_events.py`.

## Configure camera zones

Use the calibration helper with a frame from the target camera. Draw one polygon per directional carriageway/lane group, set the normal permitted travel direction for each, and add a no-parking region if needed. Opposing traffic flows need separate polygons and headings; do not combine both directions into one ROI. The helper writes normalized coordinates and the source frame size:

```powershell
python -m m3_events.calibrate_zones --input data/test.mp4 --frame 0 --node-id N01 --cam-id C01 --output configs/zones/C01.json
```

For an image, pass its path to `--input`; for a video, use `--frame` to choose a clear frame. In the window, click polygon vertices and press Enter to finish; Backspace undoes the last point and R resets the polygon. For each heading, click a point on that lane and then a second point in the permitted direction. Answer yes to “Add another directional lane?” to calibrate the opposing flow. The tool then asks whether to add an encroachment polygon and its dwell threshold (15 seconds by default).

The example file is a template only. Review the generated values before running on real footage:

- Set `frame_size` to the source video resolution used to generate the track export.
- Draw each polygon around the actual lane or no-parking area. Coordinates are normalized fractions from 0 to 1, with `(0, 0)` at the upper-left.
- Set each lane's allowed heading in image coordinates: 0° right, 90° down, 180° left, 270° up.
- Set the camera identifiers and thresholds with the project lead. The example polygons and thresholds are illustrative only.

## Rules in this first pass

- `wrong_way` compares the vehicle-centre path over a sliding window with the lane's allowed heading. Defaults require at least 8 observations, 0.75 seconds of movement, displacement of at least 1.5% of the frame diagonal, and 3 consecutive wrong-heading windows. A track gap longer than `max_track_gap_s` resets its history.
- `min_track_class_consistency` can suppress tracks whose predicted class changes repeatedly. It is a track-quality filter, not a class correction; set it only when class flicker is causing unreliable track IDs.
- `boundary_margin_px` can make points close to a lane boundary ambiguous and exclude them from wrong-way decisions. This avoids treating a vehicle near a divider as belonging confidently to either flow.
- `encroachment` starts timing when a vehicle-centre enters a no-parking polygon. It emits once after `max_dwell_s`; a detection gap beyond the zone's `max_gap_s` restarts dwell timing. Leaving the polygon resets the rule.

These rules need calibrated zones and clip review. The detector's class confidence is not a calibrated event probability; tracking errors, occlusion, perspective, and sparse detections can still cause missed or false events.

## Run

The one-command pipeline first creates the detector/ByteTrack JSONL, then runs both M3 rules. Use `models/yolo26s_traffic.onnx` for the selected YOLO26s standard-head export. The dashboard launcher, video preview and batch runner default to this model.

```powershell
python -m m3_events.run_video_pipeline --video data/test.mp4 --model models/best.pt --zones configs/zones/C01.json --output-dir runs/m3 --device 0
```

Use `--device cpu` if no supported GPU is available. Outputs are `<video>_tracks.jsonl` and `<video>_events.jsonl` under the output directory. The rule engine is offline and does not send anything to the backend.

## Batch process local recordings

After reviewing the 1.1 ONNX preview, process the clips in `data/incoming` one at a time with:

```powershell
python scripts/process_recordings_batch.py --dry-run
python scripts/process_recordings_batch.py
```

To process just one clip for a comparison run, use `--video` and give the run its own output folder. This keeps prior outputs intact:

```powershell
python scripts/process_recordings_batch.py --video 1.1.mp4 --output-dir runs/experiments/1.1_ffmpeg
```

After `d3d11h264dec` is available and has passed the direct decoder check, compare GStreamer on the same clip in a different folder:

```powershell
$gstPython = "C:\msys64\ucrt64\bin\python.exe"
python scripts/process_recordings_batch.py --video 1.1.mp4 `
  --decoder gstreamer --gstreamer-python $gstPython `
  --output-dir runs/experiments/1.1_gstreamer
```

Use the same `--imgsz`, `--conf`, and `--stride` for a fair timing comparison. For saved-video processing keep `--stride 1`; live-stream frame dropping is not used here. If the GStreamer run is both faster and visually acceptable, run the whole folder with the same GStreamer options and a new output folder.

The batch runner uses `models/yolo26s_traffic.onnx`, FFmpeg decoding, `imgsz=640`, `conf=0.15`, and `stride=1`. It processes clips sequentially to avoid competing for the laptop's integrated GPU, puts each clip's files in its own folder under `runs/all_recordings_winml`, and writes a resumable `batch_summary.json`.

Edit `configs/recording_zone_map.json` to associate each video stem (for example, `1.1`) with that camera view's calibrated zone file. The provided map associates 1.1 with C01 and 2.2 with C02; verify those are the correct views before relying on their rule events. A `null` entry still creates tracked detections and an annotated video, but skips zone-based M3 event analysis. Camera angles need their own zone calibration for accurate direction and no-parking analysis.

Use `--require-zones` to make the batch stop before processing unless every clip has a zone file, or `--resume` to skip outputs that are already complete for the same source recording. `--stride 2` roughly halves detector calls, but samples half as many frames and can miss short events; keep stride 1 for the first complete run.

## Live playback preview with detections

To watch 1.1 play at its source timing while YOLO analyzes the newest frames, activate the project virtual environment and run:

```powershell
.\.venv\Scripts\Activate.ps1
$gstPython = "C:\msys64\ucrt64\bin\python.exe"
python scripts/live_video_preview.py `
  --video data/incoming/1.1.mp4 `
  --model models/best.onnx `
  --gstreamer-python $gstPython `
  --device cpu --imgsz 640 --conf 0.15 --max-width 1280 --rotate-180
```

Press `Q` or `Esc` in the preview window to stop. The preview has a one-frame inference queue, so stale pending frames are discarded while the decoder/display continue. The overlay shows the current detection FPS, box age, and number of inference frames skipped. Boxes are held between detector updates and can lag moving objects slightly. This preview is not saved as a video; use `m3_events.run_video_pipeline` afterward when you want a complete annotated file. If the GStreamer decoder plugin is blocked or software decoding cannot keep source timing, fix the decoder setup first; this preview cannot make YOLO inference faster than the model allows.

To send the saved 1.1 clip through RTSP first, start this server in an **MSYS2 UCRT64** terminal from the project folder:

```bash
cd "/c/Users/Kaushik/Desktop/Traffic Detection"
python gst_rtsp_server.py data/incoming/1.1.mp4 --host 127.0.0.1 --port 8554 --mount /cam1 --fps 30 --max-dimension 1280
```

Leave that terminal running. In VS Code's PowerShell terminal, point the preview to its endpoint:

```powershell
.\.venv\Scripts\Activate.ps1
$gstPython = "C:\msys64\ucrt64\bin\python.exe"
python scripts/live_video_preview.py `
  --source rtsp://127.0.0.1:8554/cam1 `
  --model models/best.onnx `
  --gstreamer-python $gstPython `
  --device cpu --imgsz 640 --conf 0.15 --max-width 1280 --rotate-180
```

The RTSP server re-encodes the clip to 1280-pixel-wide H.264 at 30 FPS, then loops it. RTSP is useful for trying the live-stream path, but it adds encoding and transport; compare the displayed FPS with direct-file preview rather than assuming it will be faster.

### Optional GStreamer decoder

The default `--decoder ffmpeg` path is unchanged. For GStreamer decoding, the exporter runs `m2_detect_track/gstreamer_decode_worker.py` in a separate MSYS2 UCRT64 Python process and streams timestamped frames to the project's normal `.venv`; the MSYS2 interpreter needs PyGObject and the GStreamer playback/video plugins described in [GSTREAMER_SETUP.md](../GSTREAMER_SETUP.md). The project's standard PyPI OpenCV wheel does not need to be replaced.

Try it on 1.1 first, with the same full-frame inference and overlay settings as the current preview:

```powershell
$gstPython = "C:\msys64\ucrt64\bin\python.exe"
.\.venv\Scripts\Activate.ps1
python -m m3_events.run_video_pipeline `
  --video data/incoming/1.1.mp4 `
  --model models/best.pt `
  --zones configs/zones/C01.json `
  --output-dir runs/1.1_gstreamer `
  --annotated-output runs/1.1_gstreamer/1.1_annotated_compact.mp4 `
  --device cpu --imgsz 640 --conf 0.15 --stride 1 `
  --decoder gstreamer --gstreamer-python $gstPython
```

Set `$gstPython` to the actual UCRT64 `python.exe` path on this computer. The GStreamer helper uses `uridecodebin`, which selects from installed decoders; it can use a hardware decoder only when a compatible decoder plugin and device are available. GStreamer affects video decoding only, not YOLO inference. The exporter reports elapsed time and frames per second, so compare this run with the existing 1.1 preview before using it for the remaining clips. Keep `--stride 1`, `--imgsz 640`, and `--conf 0.15` to retain the current model sampling and quality settings. [GStreamer `uridecodebin`](https://gstreamer.freedesktop.org/documentation/playback/uridecodebin.html) can autoplug installed decoders, including hardware decoders when present.

To run the complete local demo, publish generated events to M6, and open the frontend, use:

```powershell
python scripts/run_detection_dashboard.py --video data/test.mp4 --model models/best.pt --zones configs/zones/C01.json --device cpu
```

This launcher starts the local API if needed, runs detection and rules, sends event evidence through M5, and opens `http://127.0.0.1:8000/`. It defaults to CPU, can reuse a previous tracking export with `--tracks runs/m3/test_tracks.jsonl`, and leaves the dashboard API running after the command exits. With no system FFmpeg executable, event links use the original local video and seek to the event time; when FFmpeg is installed, M5 creates short event clips instead.

This is a saved-video demo. Live RTSP inference is not yet connected to this launcher or the dashboard.

You can also run the two existing stages separately:

```powershell
python -m m3_events.event_engine --tracks tracks/campus_best.jsonl --zones configs/zones/C01.json --output data/events.jsonl
python m5_edge_agent/send_events.py --events data/events.jsonl
```

Before the M5 command, set its `source_video`, `source_start_ts`, `node_id`, and `cam_id` to match the same video and zone file. The event engine writes `frame` and `video_ts_s`; M5 turns those offsets into the timestamp, evidence clip, snapshot, and M6 event payload.

## Event class contract

The VEYAKSH detector taxonomy includes `bus_truck`, while the prior event schema did not. This branch adds that class to the event schema, M5 validation, and M6 API model so the custom detector's class name passes end to end. Confirm this contract addition with the project maintainer before merging or deploying it.

## Pedestrian path and synchronized playback

See [PEDESTRIAN_PATH.md](PEDESTRIAN_PATH.md) for the foot-contact rule, compact overlays, source-bound playback analysis, event-time popups, persistence, and commands for Point 1 / Point 2.
