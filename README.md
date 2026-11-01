# Video Ingest and Event Capture (M1 + M6)

A minimal Python 3.11 project that reads video frames with OpenCV and stores submitted events in SQLite through FastAPI.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## M1: inspect and read video frames

A short sample clip is included at `assets/traffic_demo.mp4`:

```powershell
python ingest_video.py assets/traffic_demo.mp4
```

## M6: run the API

From this project directory:

```powershell
uvicorn m6_backend.main:app --reload
```

Submit and retrieve an event from a second terminal:

```powershell
curl.exe -X POST http://localhost:8000/events -H "Content-Type: application/json" -d '{"type":"wrong_way"}'
curl.exe http://localhost:8000/events
```

Events are persisted to `events.sqlite3`. Set `EVENTS_DB_PATH` to use another SQLite file. API docs: `http://localhost:8000/docs`.

RTSP status and failure details are in `REPORT.md`.

## Visualize saved API events on the video

With the API running and events stored, open a second terminal in this directory and run:

```powershell
.\.venv\Scripts\python.exe visualize_events.py
```

The script reads `GET /events` from `http://localhost:8000`, overlays the latest four event records on the video, and writes `assets/traffic_demo_events.mp4`. These are recorded API events displayed over the footage; they are not detections inferred from the video.

## Frame-by-frame vehicle analysis preview

The analysis stage runs the NanoDet object detector on every source frame and creates a separate annotated MP4 plus a visual HTML/JSON report. This is object detection only; it does not identify unique tracked vehicles or infer wrong-way violations.

From the project directory, run:

```powershell
.\.venv\Scripts\python.exe analyze_video.py
```

Outputs:

- `assets/traffic_demo_analysis.mp4` — frame-by-frame preview with vehicle boxes, labels, confidence, frame number, and processing speed.
- `reports/traffic_analysis.html` — visual summary with embedded video and counts by detected vehicle class.
- `reports/traffic_analysis.json` — machine-readable analysis details.

The included NanoDet ONNX model and its OpenCV Zoo helper/license are in `models/`. CPU processing can take a few minutes for the full 397-frame clip. You can use another input video with `python analyze_video.py path\to\video.mp4`.

## RTSP simulated stream

See [RTSP_SETUP.md](RTSP_SETUP.md) for MediaMTX, FFmpeg looped publishing, and the OpenCV continuity check.

