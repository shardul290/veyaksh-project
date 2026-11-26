# VEYAKSH traffic operations dashboard

## Live YOLO26 monitoring

Run `./start_dashboard.ps1` from the project folder and open `http://127.0.0.1:8000/`. In **Traffic in motion**, choose a project video or enter a full local video path or RTSP URL. Select the matching camera and calibrated zones, then click **Start analysis**. Click **Stop** before changing the source. The selected model is `models/yolo26s_traffic.onnx`.

On first use of a local file, FFmpeg prepares a cached H.264 playback copy at up to 960 pixels and 25 FPS. Preparation can take time for long 4K footage; its status appears in the dashboard. Later sessions reuse it. Copies live in `runs/live_previews/` and are regenerated if the original changes. The original source remains the evidence link. Install `requirements.txt` to include the bundled FFmpeg runtime. Playback pauses automatically when its source clock is more than 700 ms ahead of completed detection, then resumes when the detector catches up. This bounds the pending work under load; it can reduce playback speed on slow hardware.

Local files decode in the browser at their source timing. The playback tab sends 960-pixel JPEG samples over a WebSocket to one YOLO26/ByteTrack worker. The visible canvas buffers up to 32 video frames and uses an adaptive presentation delay (normally a fraction of a second). Frames and boxes share a source timestamp; the display waits when inference falls behind. Positions interpolate only between detections of the same confirmed track, without predicting future positions. Keep the playback tab open while monitoring. Other viewers share the worker's preview, whose frames are annotated with their own detections. Detection speed depends on laptop load and can be lower than playback speed.

Completed detections are pushed independently of frame uploads. ONNX Runtime uses sequential execution and two CPU threads alongside DirectML where available. The tracker uses confidence thresholds appropriate to the selected detector threshold. Valid detections remain visible while awaiting a track ID, and only confirmed tracks enter event rules. This preserves detector output without claiming to recover objects the model did not detect.

RTSP input runs in the backend, using the installed GStreamer worker. If the native decoder fails before receiving a frame, the session switches to FFmpeg and displays a warning. Set `GST_PYTHON` to the GStreamer-capable Python interpreter when using a different installation. See [GStreamer setup](../GSTREAMER_SETUP.md).

The existing wrong-way, encroachment and pedestrian-path rules evaluate tracked detections as they arrive. Confirmed events save to SQLite immediately with a snapshot, source time and review status. Local-file evidence links back to its original video. RTSP evidence contains the captured snapshot. A new session has a new event identity, while multiple viewers of one session share event identities.

The live controls show playback FPS, detection FPS, display delay, saved events and the active decoder/inference provider. One source runs at a time on this laptop. Each session stores diagnostic status and tracks in `runs/live_sessions/<session_id>/`, and evidence snapshots in `m6_backend/media/event_evidence/live/`. The session logs and configuration files are local runtime data.

### Live API

- `GET /live-options` lists cameras, calibrations and project video paths.
- `POST /live-sessions` starts a local-file or RTSP session.
- `GET /live-sessions` and `GET /live-sessions/{id}` report current sessions and telemetry.
- `POST /live-sessions/{id}/stop` stops the owned source and inference processes.
- `GET /live-sessions/{id}/frames` serves the shared annotated MJPEG preview.
- `GET /live-sessions/{id}/source` serves the unchanged original local file for playback and evidence.
- `GET /live-sessions/{id}/playback` serves the cached lightweight local playback copy.
- `WS /live-sessions/{id}/input` accepts timestamped preview frames from the playback owner.

For browser input, the start response supplies a session-scoped `input_token`. Send it in the first WebSocket JSON message as `{"token": "..."}`. Subsequent binary messages contain an eight-byte little-endian float64 source timestamp followed by a JPEG. The worker retains the newest frame. Regular session responses omit the token. The dashboard manages this automatically.

Pedestrian path labels and video-time event popups are now supported for prepared camera recordings. See [playback setup and behavior](../m3_events/PEDESTRIAN_PATH.md#video-time-alerts-in-the-dashboard). The monitored video keeps playing while a non-modal alert appears; encountering an event saves it once to the evidence trail.

For manual startup, run `start_dashboard.ps1` from the project root in PowerShell.
The launcher pins the project path and checks for the standard dashboard before
reusing a running server. Open its printed URL; press Ctrl+Shift+R if an older
browser tab still displays the Sentinel page.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn m6_backend.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/ for the integrated map, CCTV recordings and evidence
trail, or http://127.0.0.1:8000/history for the permanent event record. Restart
an older backend after updating this code. The API and dashboard share an origin.

## New surveyed recordings

Put browser-compatible H.264 MP4 files under
`m6_backend/media/camera_recordings/`, then register them in
`configs/cameras.json`. Each camera entry has a `node_id`, a unique `cam_id`, a
display `name`, and one or more labeled recordings. Several camera entries can
share one location; clicking that map dot lists all cameras and recordings for
the location.

Use the location-and-angle label as the camera display name and filename, for
example `1.1.mp4`, `1.2.mp4`, and `2.2.mp4`. Keep the backend `cam_id` unique;
Point 1 angle 2 can use `C03` while remaining under node `N01`.

The Point 1 marker currently offers three saved annotated renders. Each
recording is served locally; no copy is sent to an external service. Set
`annotated: true` for videos that already contain rendered boxes or labels.

The CCTV panel discovers camera and recording changes within two seconds
without a restart. Location dots are loaded from `configs/locations.json`; add
future locations there and refresh the dashboard to show their dots. Set the
actual timezone-aware `start_ts` when known. See
`m6_backend/media/camera_recordings/README.md` for details.

The old clips are invalid for these surveyed locations. On first upgrade,
existing events become archived legacy records in History. They do not activate
the new camera pins, appear in the evidence queue, or expose their old footage.
No historical records are deleted. Only configured locations appear on the map.

## Analyse and publish events

Adding a recording enables playback; it does not run inference. Run the existing
YOLO/ByteTrack and M3 rule pipeline on the new footage with zones calibrated for
each camera. Publish the resulting JSONL events through M5 with the matching
configuration. Set `source_start_ts` to the actual recording start time first;
the placeholder intentionally prevents publishing events with invented times.

```powershell
.\.venv\Scripts\python.exe m5_edge_agent/send_events.py --config m5_edge_agent/point_1.config.json --events runs/point_1_events.jsonl
.\.venv\Scripts\python.exe m5_edge_agent/send_events.py --config m5_edge_agent/point_2.config.json --events runs/point_2_events.jsonl
```

The default `m5_edge_agent/config.json` also targets Point 1. Existing sample
events and old zone examples are not analyses of the new recordings. Do not use
them as real evidence for these locations. The sender extracts snapshots and
three-second clips using FFmpeg, or links the source at the event offset when
FFmpeg is unavailable. Evidence filenames include a recording/event fingerprint
so a later recording does not overwrite earlier clips.

## Review workflow

Click Enable sound once after each page load to allow browser audio. Newly
received events play a tone. Camera markers remain blue until their location
has an event awaiting review, then blink red until all its active events are
reviewed. Polling runs every two seconds. Reduced-motion preferences disable
animation while retaining the red alert indicator.

Review evidence at the bottom of Overview:

- False alert saves a decision and removes the event from the evidence trail.
- Take action saves a decision and opens response options.
- Alert authorities saves a timestamped local request. No external delivery
  service or authority contact is configured; no message is sent.

All events remain in History with their source clip timestamp, exact video
offset (when provided), current status and decision timestamp. The activity trail
retains the separate timestamps for each action. Review timestamps are stored in
UTC and displayed in IST; the source timestamp is preserved unchanged. Duplicate
submissions of the same review do not create duplicate audit entries. Conflicting
reviews return an error rather than overwriting an existing decision.

## Verification and data

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --check m6_dashboard/dashboard.js
```

Tests use an isolated database and temporary HTTP server. Local socket access
must be allowed. APIs include `/health`, `/nodes`, `/cameras`, `/events`,
`/events-summary`, `/events/{id}/actions` and `PATCH /events/{id}/status`.
`/docs` provides request schemas. `/events` supports status, active-only,
node/type and ID pagination filters. The former `/map` route opens Overview.

Database files, recorded videos and generated evidence remain local and ignored
by Git. Set `EVENTS_DB_PATH` to change the database location. Keep evidence under
`m6_backend/media/` for serving.
