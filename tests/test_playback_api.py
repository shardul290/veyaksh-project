import json
from pathlib import Path
import tempfile
import unittest
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import cv2
import numpy as np

from fastapi.testclient import TestClient
from m3_events import playback
from m6_backend import main as api
from m6_backend import playback as loader


class PlaybackApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.media = root / "media"
        self.media.mkdir()
        self.source = self.media / "point_2.mp4"
        writer=cv2.VideoWriter(str(self.source),cv2.VideoWriter_fourcc(*"mp4v"),10,(32,32))
        for _ in range(120):writer.write(np.zeros((32,32,3),dtype=np.uint8))
        writer.release()
        self.directory = root / "playback"
        self.config = {"node_id":"N02", "cam_id":"C02", "frame_size":{"width":1920,"height":1080}}
        self.camera = [{"node_id":"N02", "cam_id":"C02", "recordings":[{"path":"point_2.mp4","start_ts":"2026-11-19T10:00:00+05:30"}]}]
        self.event = {"type":"pedestrian_wrong_path", "track_id":7,"class":"pedestrian",
                      "video_ts_s":10.0,"frame":300,"conf":.9,"zone_id":"ROAD"}
        self.tracks = root / "tracks.jsonl"
        self.tracks.write_text(json.dumps({"frame":300,"ts":10,"track_id":7,"class":"pedestrian", "bbox":[1,2,3,4]})+"\n")
        self.events = root / "events.jsonl"
        self.events.write_text(json.dumps(self.event)+"\n")
        self.patches = [patch.object(playback,"MEDIA_ROOT",self.media),patch.object(loader,"MEDIA_ROOT",self.media),
                        patch.object(loader,"PLAYBACK_ROOT",self.directory),patch.object(api,"MEDIA_DIR",self.media),
                        patch.object(api,"DB_PATH",root/"test.db"),patch.object(api,"camera_recordings",lambda:self.camera)]
        for p in self.patches:p.start()
        loader._cache.clear()
        self.manifest = playback.register_analysis(self.source,self.tracks,self.events,self.config,self.directory)
        self.analysis_id=json.loads(self.manifest.read_text())["analysis_id"]
        self.client=TestClient(api.app)

    def tearDown(self):
        self.client.close()
        api.EVIDENCE_POOL.submit(lambda:None).result(timeout=10)
        for p in reversed(self.patches):p.stop()
        loader._cache.clear()
        self.tmp.cleanup()

    def post(self, **overrides):
        return self.client.post("/cameras/C02/playback-events",json={"analysis_id":self.analysis_id,"event_index":0,**overrides})

    def test_analysis_has_only_matching_source_and_no_local_paths(self):
        result=self.client.get("/cameras/C02/analysis")
        self.assertEqual(result.status_code,200)
        body=result.json();self.assertEqual(body["recording"],"point_2.mp4")
        self.assertNotIn("source",body);self.assertNotIn("tracks",body)
        self.assertEqual(len(body["events"]),1)
        self.assertEqual(self.client.get("/events").json(),[])

    def test_encounter_persists_once_across_replay_and_keeps_review(self):
        first=self.post();self.assertEqual(first.status_code,200)
        event=first.json();self.assertEqual(event["video_ts_s"],10)
        self.assertEqual(event["ts"],"2026-11-19T10:00:10.000+05:30")
        self.assertEqual(event["clip"],"point_2.mp4#t=10.000")
        self.client.patch(f"/events/{event['id']}/status",json={"status":"false_alert"})
        replay=self.post().json();self.assertEqual(replay["id"],event["id"])
        self.assertEqual(replay["status"],"false_alert")
        self.assertEqual(len(self.client.get("/events").json()),1)

    def test_regular_publisher_and_playback_share_deduplication(self):
        event=self.post().json()
        allowed={key:event[key] for key in api.EventIn.model_fields if key not in {"vehicle_class"}}
        allowed["class"]=event["class"]
        result=self.client.post("/events",json=allowed)
        self.assertEqual(result.status_code,201)
        self.assertEqual(result.json()["id"],event["id"])

    def test_wrong_analysis_and_event_index_are_rejected(self):
        self.assertEqual(self.post(analysis_id="0"*64).status_code,409)
        self.assertEqual(self.post(event_index=99).status_code,404)
        self.assertEqual(self.post(event_index=-1).status_code,422)

    def test_wrong_camera_and_changed_recording_rejected(self):
        self.assertEqual(self.client.get("/cameras/C01/analysis").status_code,404)
        self.source.write_bytes(b"different recording")
        self.assertEqual(self.client.get("/cameras/C02/analysis").status_code,409)
        self.assertEqual(self.post().status_code,409)

    def test_missing_analysis_is_explicit(self):
        self.manifest.unlink()
        self.assertEqual(self.client.get("/cameras/C02/analysis").status_code,404)

    def test_multiple_tabs_cannot_duplicate_same_event(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses=list(pool.map(lambda _:self.post(),range(4)))
        self.assertTrue(all(r.status_code==200 for r in responses))
        self.assertEqual(len({r.json()["id"] for r in responses}),1)
        self.assertEqual(len(self.client.get("/events").json()),1)

    def test_incident_is_saved_before_slow_snapshot_finishes(self):
        started,release=threading.Event(),threading.Event()
        def slow_snapshot(*args,**kwargs):
            started.set()
            release.wait(timeout=5)
        with patch('m5_edge_agent.send_events.extract_snapshot',side_effect=slow_snapshot):
            try:
                response=self.post()
                self.assertEqual(response.status_code,200)
                self.assertTrue(started.wait(timeout=2))
                self.assertFalse(release.is_set())
                self.assertEqual(response.json()['evidence_state'],'pending')
                self.assertEqual(len(self.client.get('/events').json()),1)
            finally:
                release.set()
                api.EVIDENCE_POOL.submit(lambda:None).result(timeout=10)


if __name__=="__main__":unittest.main()
