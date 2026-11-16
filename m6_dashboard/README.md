# M6 dashboard and M5 event-sender demo

The dashboard is served by the FastAPI backend. Open it through the API so the map, event polling, snapshots, and media links use the same origin.

## Windows setup

From the project root, install the Python dependencies once:

```powershell
cd C:\Users\91639\OneDrive\Desktop\projects\veyaksh_project
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install FFmpeg and make sure `ffmpeg` is available on `PATH`. The sender uses OpenCV to extract the event frame and FFmpeg to create a three-second H.264 clip.

## Start the backend

In terminal 1:

```powershell
cd C:\Users\91639\OneDrive\Desktop\projects\veyaksh_project
.\.venv\Scripts\python.exe -m uvicorn m6_backend.main:app --host 127.0.0.1 --port 8000
```

If port 8000 is already in use, the API may already be running. Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and check that the dashboard reports the API as connected. After pulling code changes, stop an older server with `Ctrl+C` in its terminal and restart it so it loads the updated snapshot API. Do not start another server on the same port.

## Run a detector result through to the dashboard

From the repository root, after the video, `models/best.pt`, and calibrated zone file are in place, one command runs YOLO/ByteTrack and the M3 rules, starts the local backend if necessary, sends events through M5, and opens the dashboard:

```powershell
python scripts/run_detection_dashboard.py --video data/test.mp4 --model models/best.pt --zones configs/zones/C01.json --device cpu
```

If tracks already exist, skip the YOLO pass and publish them through the rules and dashboard with `--tracks runs/m3/test_tracks.jsonl`. This is an offline saved-video flow. Live RTSP inference is not wired into the dashboard yet. If FFmpeg is unavailable, M5 links the original local video at the event timestamp instead of making a short excerpt.

## Send the five M5 sample events

In terminal 2, from the project root:

```powershell
.\.venv\Scripts\python.exe m5_edge_agent\send_events.py
```

The sender reads `m5_edge_agent/sample_events.jsonl`, applies the node and camera from `m5_edge_agent/config.json`, converts `video_ts_s` to an ISO 8601 source timestamp, extracts one snapshot and one three-second clip for each event, and posts each event to `POST /events`. The API stores the event and its snapshot/clip paths; new map pins blink, and the event details show the snapshot and clip link.

Open the dashboard first, then run the sender. Five events are uploaded once and the script exits. To space them out by three seconds, use:

```powershell
.\.venv\Scripts\python.exe m5_edge_agent\send_events.py --interval 3
```

## Configure a different source video or camera

Edit `m5_edge_agent/config.json`:

- `node_id` must already exist in the backend database (`N01`, `N02`, or `N03` for the sample setup).
- `cam_id` identifies the source camera.
- `source_video` is the local video used to extract event snapshots and clips. Use a video at least three seconds long and keep local footage out of Git.
- `source_start_ts` is the timezone-aware start time of the source recording. Event `ts` is this time plus each JSONL `video_ts_s` offset.
- `evidence_dir` must remain under `m6_backend/media/`; generated images and clips are local and ignored by Git.

The sample event records use relative video offsets within a four-second local demo clip. If that clip is not present on your machine, set `source_video` to a local video file you are allowed to use and adjust each `video_ts_s` to fit its duration. The sender does not wait for or require the live event engine.

## Check the result

- Dashboard: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- API health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- API event list: [http://127.0.0.1:8000/events?limit=10](http://127.0.0.1:8000/events?limit=10)
- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Click an item in Latest events to view its event-frame snapshot and open its short clip. The map checks for events every two seconds, so the corresponding camera pin should blink shortly after upload.

## Git and media policy

Database files, raw videos, extracted frames, generated event evidence, and dataset archives are ignored by `.gitignore`. Keep licensed/self-recorded source footage and generated MP4 clips local. The JSONL event examples and sender code are safe to review in Git; the event snapshots are generated locally from the configured video.
