# Pedestrian path classification

The existing YOLO checkpoint detects `pedestrian`; ByteTrack provides identities.
The new rule classifies the bottom-center of each bounding box (estimated ground
contact), rather than its center, against normalized polygons for a fixed camera.

| Ground location | Confirmed display | Alert |
| --- | --- | --- |
| Footpath | Green RIGHT PATH | None |
| Zebra crossing, including overlap with road | Green RIGHT PATH | None |
| Road outside safe zones | Red WRONG PATH | `pedestrian_wrong_path` |
| Outside configured zones, low confidence, or near a boundary | Yellow UNKNOWN | None |
| Waiting for sustained observations | Yellow PENDING | None |

This implements the project's spatial safety policy. It does not infer legal
offences, traffic-light permission, or whether a person has completed a crossing.
Both walking along a road and crossing outside the zebra region trigger the same
road-presence alert. Stationary people on the road are included.

## Setup and run (PowerShell, from project root)

Use a Python environment with the requirements installed. The checked-in `.venv`
was missing OpenCV in this workspace; verification used the installed Python 3.11
interpreter via `python`.

python -m m3_events.run_video_pipeline --video m6_backend/media/camera_recordings/point_1.mp4 --model models/best.pt --zones configs/zones/C01.pedestrian.json --output-dir runs/pedestrian --annotated-output runs/pedestrian/paths.mp4 --device cpu
```

Draw road ground areas, all visible footpaths, and any zebra crossings. Answer
yes again to add multiple regions of a kind. A camera without a zebra crossing
needs no zebra polygon. Coordinates must match this camera viewpoint and the
configured frame dimensions must match the input video. Redraw after the camera
moves. The example JSON is illustrative and is not calibrated for the recordings.

```powershell
python -m m3_events.run_video_pipeline --video m6_backend/media/camera_recordings/point_1.mp4 --model models/best.pt --zones configs/zones/C01.pedestrian.json --output-dir runs/pedestrian --annotated-output runs/pedestrian/paths.mp4 --device cpu
```

Outputs include `*_tracks.jsonl`, `*_classified_tracks.jsonl`, `*_events.jsonl`,
and the requested MP4 with zone outlines, boxes, IDs, and path status. Track rows
contain `path.status`, `path.reason`, and `path.zone_id`. Untracked detections
cannot be temporally confirmed. Right-path decisions are recorded in track rows;
only wrong-path episodes enter the incident queue.

The existing publish launcher accepts this same zone configuration:

```powershell
python scripts/run_video_publish.py --video m6_backend/media/camera_recordings/point_1.mp4 --model models/best.pt --zones configs/zones/C01.pedestrian.json --output-dir runs/pedestrian --annotated-output runs/pedestrian/paths.mp4 --device cpu
```

Start the API first (`python -m uvicorn m6_backend.main:app`). The sender and API
accept `pedestrian_wrong_path`; the dashboard names these events "Pedestrian
wrong path". Video-job detection responses use the classified tracks when available.

For already exported tracks:

```powershell
python -m m3_events.event_engine --tracks tracks.jsonl --zones configs/zones/C01.pedestrian.json --output events.jsonl --classified-tracks classified.jsonl
```

## Temporal algorithm and tuning

Default settings, optionally provided in the zone config's `pedestrian_path`:

```json
{
  "min_confidence": 0.35,
  "min_observations": 4,
  "confirm_s": 0.6,
  "max_gap_s": 1.0,
  "boundary_margin_px": 4.0
}
```

Both the observation count and elapsed duration must pass within one continuous
candidate zone. This avoids confirming a violation from one misplaced box.
Safe zones override the road; points near either side of a safe-zone edge remain
unknown. Confidence filtering, timestamp deduplication, stale-track expiry, and
one alert per continuous wrong-path episode reduce noise. A confirmed safe path
re-arms the next road alert. Brief safe/unknown observations do not re-arm it.
After a tracking gap, a fresh episode needs confirmation again; ID switches can
still produce separate alerts. Keep the sampling interval below `max_gap_s` and
choose count/duration thresholds appropriate to the processed frame rate.

`boundary_margin_px` refers to source-image pixels. The event `conf` is mean
detector confidence during confirmation, not violation probability. Partial
occlusion, truncated feet, incorrect boxes, and bad calibration can cause errors.
Validate against labeled footage before operational use; no accuracy percentage
has been established for this camera.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Tests cover safe-zone priority, feet versus box center, road alerts, boundary
jitter, confidence filtering, timestamp duplication, track gaps, safe recovery,
repeat-alert suppression, config validation, vehicle-config compatibility, track
export ingestion, sender validation, and the existing dashboard workflows.


## Video-time alerts in the dashboard

The dashboard now draws compact, regular-weight labels and thin outlines over the
original recording. Pedestrian outlines are green by default and turn red when
the rule confirms a wrong path. Labels use a translucent background. Unknown or
pending status is still distinguished in the text; a green pedestrian outline
alone is not a confirmed safe-path decision.

The pipeline automatically registers playback analysis for source recordings
inside `m6_backend/media`. Existing tracks can be reused without rerunning YOLO:

```powershell
python -m m3_events.prepare_playback --video m6_backend/media/camera_recordings/point_2.mp4 --zones configs/zones/C02.pedestrian.json --tracks runs/point2_pedestrian/point_2_tracks.jsonl --output-dir runs/point2_pedestrian
```

Point 1 uses the same command with `point_1.mp4`, `C01.pedestrian.json`,
`runs/pedestrian/point_1_tracks.jsonl`, and output directory `runs/pedestrian`.
Add `--annotated-output <new-output.mp4>` to regenerate a standalone compact-label
video. Standalone MP4 files contain annotations; interactive alerts run in the
web dashboard.

Restart the backend after these code changes, then refresh the dashboard:

```powershell
python -m uvicorn m6_backend.main:app --host 127.0.0.1 --port 8000
```

Choose a Point 1/Point 2 camera card or map marker. The analysis-ready message
confirms the correct recording is loaded. The video-frame callback drives boxes
and popups from media time, independent of wall time and the API polling interval.
A violation confirmed at 10 seconds appears when playback reaches that timestamp.
The temporal confirmation delay remains part of the algorithm: a brief incursion
may be pending before the rule confirms it. This is synchronized replay of
precomputed video analysis, not live RTSP inference.

Alerts are non-modal, include a short playable preview around the event, and
leave the monitored video running. They are saved to the permanent event trail
when encountered. Saving includes camera, track, type, exact video offset,
source-recording identity, source clip link, and an annotated snapshot when the
frame can be extracted. Review opens the evidence dialog and pauses the monitored
video. Sound remains opt-in through Enable sound.

Pause stops new popups. Seeking forward skips events that were jumped over;
seeking backward or looping re-arms playback popups. Next event moves to two
seconds before the next event for convenient inspection. Multiple alerts at the
same timestamp are persisted separately. Replays, multiple tabs, retries, and M5
uploads use the same event key and cannot create duplicate trail entries or reset
an operator's review decision. Existing incident history remains visible.

Analysis is bound to the exact source path, file size, and modification time.
Changed recordings require analysis preparation again. If no analysis exists,
the player clearly reports raw-recording mode. A known `start_ts` in
`configs/cameras.json` is used for wall-clock timestamps; otherwise the source
file modification time is the fallback. Video offsets are unaffected.

Playback manifests live in `runs/playback`. The manifest identifies the exact
classified-track and event files; keep these files together with the project.
The current prepared exports contain 10 Point 2 events (first at approximately
15.37 seconds) and 7 Point 1 events in the full recording. Those counts describe the configured
rules and available tracks, not an independent accuracy measurement.

Additional verification:

```powershell
node --test tests/test_playback_clock.js
```


The legacy `runs/pedestrian/point_1_tracks.jsonl` ends at 15.58 seconds.
The active Point 1 analysis is now `runs/point1_full`, covering the complete
181.36-second source recording with 5,702 detections and 7 confirmed path events.
The first event is at 57.10 seconds. A coverage warning is displayed if a legacy
partial export is loaded again. To regenerate the full recording after a camera
or configuration change, run:

```powershell
python -m m3_events.run_video_pipeline --video m6_backend/media/camera_recordings/point_1.mp4 --model models/best.pt --zones configs/zones/C01.pedestrian.json --output-dir runs/point1_full --device cpu --stride 5
```

Refresh/reselect the camera after completion. New tracking exports include a
`.meta.json` completion record so an empty tail of a fully processed recording
can be distinguished from legacy exports of unknown coverage. Frame sampling
uses actual decoded source timestamps (falling back to frame index / FPS where
unavailable), avoiding an offset from skipped frames. Decoder and CPU inference
threads are bounded to reduce memory pressure on laptops. Full CPU processing of
high-resolution footage can take substantially longer than the clip duration.

The standalone compact export is `runs/point2_pedestrian/point2_paths_compact.mp4`.
Its annotations are separate from browser popups. Existing older annotated MP4
files keep their original appearance until regenerated.


The dashboard launcher now checks the running backend's playback capability;
it rejects an older server that can serve the HTML but lacks the analysis API.
An outdated backend is also explained directly in the player instead of the
ambiguous "Raw recording / Not Found" message.

Full Point 1 preprocessing preserves all 9,124 source frames in a smaller
processing copy, samples 1,141 frames for inference, and maps detections back to
the original 3840 x 2160 camera coordinates. The processing copy needed an
additional 180-degree orientation correction; it was validated against upright
source frames before registering the analysis. `--rotate-180` on the detection
exporter is for such processing proxies only. The original camera file is already
auto-oriented by the decoder and does not need that additional flag.

Detections awaiting a ByteTrack identity remain visible with an unknown path
status; they cannot generate temporal violation alerts. Incident persistence
happens before snapshot decoding. A single background worker prepares missing
snapshots and the trail displays a pending-frame state until they are ready,
avoiding delayed event saves and concurrent 4K decoder memory spikes.
