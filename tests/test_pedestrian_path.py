import copy
import json
from pathlib import Path
import tempfile
import unittest

from m3_events.event_engine import load_zones, read_tracks, run_engine
from m3_events.rules.pedestrian_path import PedestrianPathRule
from m5_edge_agent.send_events import validate_engine_event


def rectangle(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def config():
    return {"node_id": "N01", "cam_id": "C01", "frame_size": {"width": 1000, "height": 1000},
            "pedestrian_zones": [
                {"zone_id": "ROAD", "kind": "road", "polygon": rectangle(.1, .1, .9, .9)},
                {"zone_id": "FOOT", "kind": "footpath", "polygon": rectangle(.1, .1, .25, .9)},
                {"zone_id": "ZEBRA", "kind": "zebra_crossing", "polygon": rectangle(.1, .45, .9, .55)}]}


def detection(t, x=500, y=700, label="pedestrian", confidence=.9):
    return {"track_id": 1, "frame": round(t * 30), "ts": t,
            "bbox": [x - 20, y - 200, x + 20, y], "class": label, "conf": confidence}


class PedestrianTests(unittest.TestCase):
    def rule(self):
        c = config()
        return PedestrianPathRule(1000, 1000, c["pedestrian_zones"])

    def series(self, rule, start=0, **kwargs):
        return [rule.update(detection(start + i * .2, **kwargs)) for i in range(6)]

    def test_road_alert_confirmed_once(self):
        results = self.series(self.rule())
        self.assertEqual(results[0][0]["status"], "pending")
        self.assertEqual(results[-1][0]["status"], "wrong_path")
        events = [event for _, event in results if event]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "pedestrian_wrong_path")
        validate_engine_event(events[0], 1)

    def test_footpath_and_crossing_override_road(self):
        for x, y in [(170, 700), (500, 500)]:
            results = self.series(self.rule(), x=x, y=y)
            self.assertEqual(results[-1][0]["status"], "right_path")
            self.assertFalse(any(event for _, event in results))

    def test_uses_feet_not_box_center(self):
        self.assertEqual(self.series(self.rule(), x=500, y=500)[-1][0]["status"], "right_path")

    def test_outside_and_low_confidence_unknown(self):
        for kwargs in ({"x": 950}, {"confidence": .1}):
            results = self.series(self.rule(), **kwargs)
            self.assertEqual(results[-1][0]["status"], "unknown")
            self.assertFalse(any(e for _, e in results))

    def test_boundary_both_sides_unknown(self):
        for x in (249, 251):
            self.assertEqual(self.series(self.rule(), x=x)[-1][0]["status"], "unknown")

    def test_duplicate_time_does_not_confirm(self):
        rule = self.rule()
        for _ in range(10):
            self.assertIsNone(rule.update(detection(0))[1])
        self.assertEqual(rule.states[1].count, 1)

    def test_gap_requires_new_confirmation(self):
        rule = self.rule()
        rule.update(detection(0))
        self.assertEqual(rule.update(detection(3))[0]["status"], "pending")
        self.assertEqual(rule.states[1].count, 1)

    def test_safe_recovery_rearms(self):
        rule = self.rule()
        self.series(rule)
        self.series(rule, start=1.2, x=170)
        results = self.series(rule, start=2.4)
        self.assertEqual(sum(e is not None for _, e in results), 1)

    def test_brief_safe_jitter_does_not_rearm(self):
        rule = self.rule()
        self.series(rule)
        rule.update(detection(1.2, x=170))
        self.assertFalse(any(e for _, e in self.series(rule, start=1.4)))

    def test_vehicle_ignored(self):
        self.assertFalse(any(e for _, e in self.series(self.rule(), label="car")))

    def test_engine_reads_and_classifies_pedestrians(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "tracks.jsonl"
            p.write_text("\n".join(json.dumps(detection(i * .2)) for i in range(6)))
            tracks = read_tracks(p)
            events = run_engine(tracks, config())
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["cam_id"], "C01")
            self.assertEqual(tracks[1][-1]["path"]["status"], "wrong_path")

    def test_empty_scene_is_successful_without_alerts(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "empty.jsonl"
            p.write_text("")
            self.assertEqual(run_engine(read_tracks(p), config()), [])

    def test_invalid_settings_rejected(self):
        for settings in ({"confirm_s": float("nan")}, {"min_observations": 1},
                         {"max_gap_s": 0}, {"min_confidence": 2}, {"confirm_s": True}):
            c = config()
            c["pedestrian_path"] = settings
            with self.assertRaises(ValueError):
                load_zones(c)

    def test_bad_zone_rejected(self):
        for change in ({"kind": "grass"}, {"polygon": rectangle(0, 0, 0, 0)}, {"zone_id": "FOOT"}):
            c = config()
            c["pedestrian_zones"][0].update(change)
            with self.assertRaises(ValueError):
                load_zones(c)

    def test_existing_vehicle_config(self):
        c = json.loads((Path(__file__).resolve().parents[1] / "configs/zones/C01.example.json").read_text())
        load_zones(copy.deepcopy(c))
        tracks = {1: [detection(i * .2) for i in range(6)]}
        self.assertEqual(run_engine(tracks, c), [])


if __name__ == "__main__":
    unittest.main()
