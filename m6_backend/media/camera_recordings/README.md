# Local camera recordings

Put browser-compatible H.264 MP4 files in this folder. Register each file in
`configs/cameras.json`; its `path` is relative to `m6_backend/media`, for example
`camera_recordings/point_1_view_a.mp4`. The dashboard filters out configured
files that are not present yet.

`configs/locations.json` defines the map dots. `configs/cameras.json` defines
the cameras at those locations. Use one camera entry per physical camera or
view, with a globally unique `cam_id`. Several entries can share a `node_id`.
Use the location-and-angle label as the camera `name` and video filename, such
as `1.1` for Point 1 / angle 1 and `1.2` for Point 1 / angle 2. Keep the
technical `cam_id` unique (`C01`, `C03`, etc.) even when two views share a
location.
Put one or more `{ "label", "path", "start_ts" }` recording entries under a
camera to offer multiple clips for that view. Set `annotated: true` when boxes
or labels are already drawn into the video; the player then skips the live
detection overlay. When a location has no recording yet, it still appears on
the map and shows a “recordings on the way” state.

Camera and recording changes appear in the dashboard automatically. After
adding or changing a location in `configs/locations.json`, refresh the page to
reload the map dots.

The current Point 1 entry is labeled `1.1` and offers three saved annotated
renders; Point 2 is labeled `2.2` and has one recording. More views can be
added by placing files here and adding camera or recording entries to
`configs/cameras.json`.

Adding a recording makes it available for playback; detection is a separate
step. Before publishing detector events, calibrate each camera's zones and set
the actual timezone-aware recording start time. Do not invent this timestamp.
Set `start_ts` in `configs/cameras.json` once known; until then the dashboard
honestly shows "Not recorded" for the recording start.

For each camera, use its matching sender config in `m5_edge_agent/` with
`send_events.py --config ... --events <analysed-events.jsonl>`. Set
`source_start_ts` in that config first. The sender records each event's source
timestamp and exact video offset, plus a snapshot and event clip. Generated
evidence is kept in a separate folder for each camera.

The dashboard requires one click on Enable sound after each page load. New
events then play an alert tone. Blue markers blink red for unreviewed events.
False alerts leave the evidence queue but remain in History. Take action and
authority alert requests each retain a separate UTC audit timestamp, displayed
in IST. Authority requests are recorded locally, not delivered externally.
