"""Live lifecycle, shared viewers, calibration and online event persistence."""
import copy
import contextlib
import io
import json
import subprocess
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from m6_backend import main as api
from m6_backend.live import LiveManager, safe_source, public_error
from m6_backend.live_worker import OnlineRules, LivePipeline
from m6_backend.live_preview import preview_path
from datetime import datetime, timezone


def zones():
    return {"node_id": "N01", "cam_id": "C01", "frame_size": {"width": 1920, "height": 1080},
            "max_track_gap_s": 1.0,
            "no_park_zones": [{"zone_id": "PARK", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                               "max_dwell_s": .5, "max_gap_s": 1}]}


def detection(ts, track=1):
    return {"track_id": track, "frame": round(ts * 30), "ts": ts, "class": "car",
            "bbox": [100, 200, 200, 300], "conf": .9}


class OnlineRuleTests(unittest.TestCase):
    def test_playback_cache_changes_with_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "video.mp4"
            source.write_bytes(b"first")
            first = preview_path(root, source)
            self.assertEqual(first, preview_path(root, source))
            source.write_bytes(b"updated video")
            self.assertNotEqual(first, preview_path(root, source))
            self.assertTrue(first.is_relative_to(root / "runs/live_previews"))

    def test_detector_boxes_survive_unconfirmed_tracking_and_use_raw_positions(self):
        from ultralytics.engine.results import Results
        from types import SimpleNamespace
        pipeline = LivePipeline({"zones": zones(), "confidence": .15, "directory": "."})
        # The tracker only confirms one of two valid detector results, and its
        # filtered coordinates lag behind the measured box on the source frame.
        pipeline.tracker = SimpleNamespace(update=lambda *_: np.array([[0, 0, 10, 10, 7, .8, 3, 0]]))
        result = Results(np.zeros((180,320,3), dtype=np.uint8), "test", {3:"car",6:"pedestrian"},
                         boxes=np.array([[100,20,150,80,.8,3], [200,30,220,90,.2,6]], dtype=np.float32))
        detected = pipeline.track_detections(result, 1, .2)
        self.assertEqual(len(detected),2)
        self.assertEqual([d["track_id"] for d in detected],[7,None])
        self.assertEqual(detected[0]["bbox"],[600,120,900,480])
        self.assertEqual(pipeline.rules.update(detected,.2),[])

    def test_event_once_and_expired_track_rearms(self):
        rules = OnlineRules(zones())
        found = []
        for ts in [0, .25, .5, .75, 1]:
            found.extend(rules.update([detection(ts)], ts))
        self.assertEqual([event["type"] for event in found], ["encroachment"])
        self.assertEqual(found[0]["video_ts_s"], .5)
        rules.update([], 3)
        self.assertEqual(rules.seen, {})
        self.assertEqual(rules.encroachment.states, {})
        rules.update([detection(3)], 3)
        self.assertEqual(len(rules.update([detection(3.5)], 3.5)), 1)

    def test_unassigned_boxes_cannot_create_events(self):
        rules = OnlineRules(zones())
        for ts in [0, .5, 1]:
            self.assertEqual(rules.update([detection(ts, None)], ts), [])

    def test_credentials_are_redacted(self):
        source = "rtsp://admin:secret@localhost:8554/traffic"
        self.assertNotIn("secret", safe_source(source))
        self.assertNotIn("secret", public_error("Failure " + source, source))

    def test_plugin_failure_uses_paced_file_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 180))
            for _ in range(3):
                writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
            writer.release()
            pipeline = LivePipeline({"source": str(path), "source_kind": "file",
                                     "zones": zones(), "directory": directory, "gst_python": "missing"})
            size = pipeline.probe()
            with patch("m6_backend.live_worker.iter_gstreamer_frames", side_effect=RuntimeError("Plugin blocked")):
                frames = list(pipeline.frames(size))
            self.assertEqual(len(frames), 3)
            self.assertEqual(pipeline.decoder_name, "FFmpeg")
            self.assertEqual(frames[1][2].shape, (180, 320, 3))
            self.assertAlmostEqual(frames[2][1], .2, places=2)

    def test_online_event_snapshot_matches_api_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"zones": zones(), "directory": directory, "media_dir": directory,
                      "source_kind": "file", "session_id": "b" * 32, "source_label": "test.mp4",
                      "started_at": datetime.now(timezone.utc).isoformat()}
            pipeline = LivePipeline(config)
            event = {"type": "encroachment", "track_id": 1, "class": "car", "conf": .9,
                     "zone_id": "PARK", "frame": 15, "video_ts_s": .5, "bbox": [10, 20, 100, 200]}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                pipeline.save_event(event, np.zeros((180, 320, 3), dtype=np.uint8))
            payload = json.loads(output.getvalue().split("@event ", 1)[1])
            validated = api.EventIn.model_validate(payload)
            self.assertEqual(validated.video_ts_s, .5)
            self.assertTrue((Path(directory) / validated.snapshot).is_file())
            self.assertTrue(validated.clip.endswith("/source#t=0.500"))


FAKE_WORKER = """
import json,sys,time
from pathlib import Path
config=json.loads(Path(sys.argv[1]).read_text())
folder=Path(config["directory"])
(folder/"frame.jpg").write_bytes(b"test-jpeg")
(folder/"status.json").write_text(json.dumps({"state":"running","video_ts_s":1}))
payload={"node_id":"N01","cam_id":"C01","ts":config["started_at"],"type":"encroachment",
         "track_id":1,"class":"car","conf":0.9,"zone_id":"PARK","plate":None,
         "clip":"/live-sessions/"+config["session_id"]+"/source#t=1",
         "snapshot":"event_evidence/test.jpg","video_ts_s":1,
         "event_key":"a"*64,"source_recording":"test.mp4"}
print("@event "+json.dumps(payload),flush=True)
while not (folder/"stop").exists():
    time.sleep(.02)
"""


class LiveApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "test.mp4"
        self.source.write_bytes(b"test-video")
        self.zone_dir = self.root / "configs/zones"
        self.zone_dir.mkdir(parents=True)
        (self.zone_dir / "C01.json").write_text(json.dumps(zones()))
        self.model = self.root / "model.onnx"
        self.model.touch()
        self.manager = LiveManager(self.root, self.root / "media",
                                   lambda payload: api.create_event(api.EventIn.model_validate(payload)))
        self.manager.root = Path(__file__).resolve().parents[1]
        original_popen = subprocess.Popen

        def launch(args, **kwargs):
            return original_popen([sys.executable, "-u", "-c", FAKE_WORKER, args[-1]], **kwargs)

        self.patches = [
            patch.object(api, "live_manager", self.manager),
            patch.object(api, "ROOT", self.root),
            patch.object(api, "ZONE_DIR", self.zone_dir),
            patch.object(api, "MODEL_FILE", self.model),
            patch.object(api, "DB_PATH", self.root / "events.db"),
            patch.object(api, "camera_recordings", return_value=[{"cam_id": "C01", "node_id": "N01", "name": "test"}]),
            patch.object(api, "gstreamer_python", return_value=Path(sys.executable)),
            patch("m6_backend.live.subprocess.Popen", side_effect=launch),
        ]
        # Event validation reads this inventory through ROOT.
        (self.root / "configs/cameras.json").write_text(json.dumps([{"cam_id":"C01","node_id":"N01"}]))
        for item in self.patches:
            item.start()
        self.client = TestClient(api.app)

    def tearDown(self):
        self.manager.close()
        self.client.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def start(self, **changes):
        return self.client.post("/live-sessions", json={
            "source": str(self.source), "cam_id": "C01", **changes})

    def test_shared_viewers_save_one_event_and_stop_releases_worker(self):
        response = self.start()
        self.assertEqual(response.status_code, 201, response.text)
        session = response.json()["session_id"]
        for _ in range(100):
            status = self.manager.snapshot(session)
            if status["event_count"] == 1:
                break
            time.sleep(.02)
        self.assertEqual(status["event_count"], 1)
        self.assertEqual(self.start().status_code, 409)
        first = self.manager.frames(session)
        second = self.manager.frames(session)
        self.assertIn(b"test-jpeg", next(first))
        self.assertIn(b"test-jpeg", next(second))
        first.close()
        second.close()
        self.assertEqual(len(self.client.get("/events").json()), 1)
        source = self.client.get(f"/live-sessions/{session}/source")
        self.assertEqual(source.content, b"test-video")
        self.assertEqual(self.client.get(f"/live-sessions/{session}/playback").content, b"test-video")
        self.client.post(f"/live-sessions/{session}/stop")
        process = self.manager.sessions[session]["process"]
        self.assertIsNotNone(process.poll())
        # The immutable source manifest supports historical evidence after restart.
        restored = LiveManager(self.manager.root, self.root / "media", lambda _: None)
        restored.directory = self.manager.directory
        self.assertEqual(restored.source_path(session), self.source)

    def test_invalid_source_and_cross_camera_calibration(self):
        self.assertEqual(self.start(source="missing.mp4").status_code, 422)
        self.assertEqual(self.start(source="rtsp://:bad/stream").status_code, 422)
        self.assertEqual(self.start(zone_config="../outside.json").status_code, 422)
        wrong = zones()
        wrong["cam_id"] = "C02"
        (self.zone_dir / "C02.json").write_text(json.dumps(wrong))
        self.assertEqual(self.start(zone_config="configs/zones/C02.json").status_code, 422)
        self.assertEqual(self.client.post("/live-sessions/absent/stop").status_code, 404)

    def test_browser_frames_require_owner_and_forward_timestamps(self):
        response = self.start(browser_input=True).json()
        session = response["session_id"]
        token = response["input_token"]
        ok, image = cv2.imencode(".jpg", np.zeros((180, 320, 3), dtype=np.uint8))
        self.assertTrue(ok)
        endpoint = f"/live-sessions/{session}/frame?ts=1"
        headers = {"Content-Type": "image/jpeg", "X-Playback-Token": token}
        self.assertEqual(self.client.post(endpoint, content=image.tobytes()).status_code, 409)
        self.assertEqual(self.client.post(endpoint, content=image.tobytes(), headers=headers).status_code, 200)
        self.assertEqual(self.client.post(endpoint.replace("ts=1", "ts=0"), content=image.tobytes(), headers=headers).status_code, 409)
        self.assertNotIn("input_token", self.client.get(f"/live-sessions/{session}").json())
        with self.client.websocket_connect(f"/live-sessions/{session}/input") as socket:
            socket.send_json({"token": token})
            socket.send_bytes(struct.pack("<d", 2.0) + image.tobytes())
            self.assertEqual(socket.receive_json()["session_id"], session)
            # Results must arrive even when playback sends no subsequent frame.
            folder = self.manager.sessions[session]["directory"]
            (folder / "status.json").write_text(json.dumps({"detection_ts_s": 2.0, "boxes": []}))
            self.assertEqual(socket.receive_json()["detection_ts_s"], 2.0)
            socket.send_text("end")
            self.assertEqual(socket.receive_json()["state"], "finishing")
        self.assertEqual(self.client.post(endpoint, content=b"invalid", headers=headers).status_code, 422)
        self.assertEqual(self.client.post(f"/live-sessions/{session}/end", headers=headers).status_code, 200)


if __name__ == "__main__":
    unittest.main()
