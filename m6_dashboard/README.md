# M6 dashboard

The FastAPI service serves a dark VEYAKSH Sentinel dashboard at `http://127.0.0.1:8000/`. It polls the local event API, presents summary counts and event mix, and links each event to its saved frame, clip, and plate image where available.

## Saved video analysis

Use the Video control to upload a local clip. The backend saves it under the ignored `data/dashboard_uploads/` directory, runs `models/best.pt` through YOLO/ByteTrack and the M3 rules, then publishes rule events through M5 to the dashboard API. When inference finishes, the browser plays the original clip with per-frame detection boxes rendered on a canvas overlay. Rule-event detections are highlighted in red; ordinary detections use cyan.

Supported formats: MP4, MOV, M4V, AVI, MKV, and WebM, up to 1 GiB. The zone configuration must be a valid project-relative JSON path under `configs/zones/`. CPU is selected by default; choose CUDA 0 only with a working CUDA-enabled PyTorch installation.

This is offline processing of an uploaded saved video. The CCTV Integration control currently explains the required browser-compatible stream/relay; live RTSP inference is not connected yet.

## Start the backend

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn m6_backend.main:app --host 127.0.0.1 --port 8000
```

If port 8000 is already occupied, stop the existing server in its terminal with `Ctrl+C`, then restart it so it loads current code. Do not start a second server on the same port.

## Check the service

- Dashboard: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- API health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- API events: [http://127.0.0.1:8000/events?limit=20](http://127.0.0.1:8000/events?limit=20)
- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Uploaded videos, inference outputs, database files, and generated event evidence are local runtime data and should not be committed to Git.
