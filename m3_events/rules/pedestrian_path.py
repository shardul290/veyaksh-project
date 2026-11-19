"""Camera-calibrated pedestrian ground-contact classification with temporal confirmation.

No learned road segmentation is assumed. Unknown ground and ambiguous boundaries
are deliberately left unclassified. Coordinates are normalized camera coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from .geometry import point_in_polygon
from .wrong_way import boundary_distance_px


DEFAULTS = {
    "min_confidence": 0.35, "min_observations": 4,
    "confirm_s": 0.6, "max_gap_s": 1.0, "boundary_margin_px": 4.0,
}


def validate_pedestrian_config(config: dict[str, Any], validate_polygon) -> None:
    zones = config.get("pedestrian_zones", [])
    if not isinstance(zones, list):
        raise ValueError("pedestrian_zones must be a list")
    ids = {z["zone_id"] for z in config.get("lanes", []) + config.get("no_park_zones", [])}
    for zone in zones:
        if not isinstance(zone, dict) or not isinstance(zone.get("zone_id"), str) or not zone["zone_id"]:
            raise ValueError("each pedestrian zone needs a non-empty zone_id")
        if zone["zone_id"] in ids:
            raise ValueError(f"duplicate zone_id: {zone['zone_id']}")
        ids.add(zone["zone_id"])
        if zone.get("kind") not in {"road", "footpath", "zebra_crossing"}:
            raise ValueError("pedestrian zone kind must be road, footpath or zebra_crossing")
        zone["polygon"] = validate_polygon(zone.get("polygon"), zone["zone_id"])
        polygon = zone["polygon"]
        area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, polygon[1:] + polygon[:1])))
        if area < 1e-10:
            raise ValueError(f"{zone['zone_id']}: polygon has zero area")
    settings = config.get("pedestrian_path", {})
    if not isinstance(settings, dict):
        raise ValueError("pedestrian_path must be an object")
    for key, default in DEFAULTS.items():
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            raise ValueError(f"pedestrian_path.{key} must be finite and non-negative")
        if key == "min_observations" and (not isinstance(value, int) or value < 2):
            raise ValueError("pedestrian_path.min_observations must be an integer >= 2")
        if key == "max_gap_s" and value <= 0:
            raise ValueError("pedestrian_path.max_gap_s must be positive")
        if key == "min_confidence" and value > 1:
            raise ValueError("pedestrian_path.min_confidence must be in [0, 1]")


@dataclass
class PathState:
    last_ts: float
    candidate: tuple[str, str] | None = None
    since: float = 0.0
    count: int = 0
    confidence_sum: float = 0.0
    confirmed: tuple[str, str] | None = None


@dataclass
class PedestrianPathRule:
    frame_width: int
    frame_height: int
    zones: list[dict[str, Any]]
    settings: dict[str, Any] = field(default_factory=dict)
    states: dict[int, PathState] = field(default_factory=dict)
    _next_cleanup: float = 0.0

    def update(self, detection: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Return current display decision and one alert per confirmed unsafe episode.

        A safe-zone transition re-arms alerts only after temporal confirmation.
        Duplicate timestamps cannot advance confirmation. Stale IDs are expired.
        """
        result = {"status": "unknown", "reason": "outside_calibrated_zones", "zone_id": None}
        if detection["class"] != "pedestrian" or detection.get("track_id") is None:
            return result, None
        opts = {**DEFAULTS, **self.settings}
        ts, track_id = detection["ts"], detection["track_id"]
        if ts >= self._next_cleanup:
            self.states = {k: v for k, v in self.states.items() if ts - v.last_ts <= opts["max_gap_s"]}
            self._next_cleanup = ts + opts["max_gap_s"]
        state = self.states.get(track_id)
        if state is not None and ts <= state.last_ts:
            return {**result, "reason": "non_increasing_timestamp"}, None
        if state is None or ts - state.last_ts > opts["max_gap_s"]:
            state = PathState(ts)
            self.states[track_id] = state
        state.last_ts = ts
        x1, _, x2, y2 = detection["bbox"]
        x, y = (x1 + x2) / 2 / self.frame_width, y2 / self.frame_height
        candidate = None
        reason = result["reason"]
        if detection["conf"] < opts["min_confidence"]:
            reason = "low_detection_confidence"
        else:
            # Safe zones take priority where crossings/footpaths overlap road ROIs.
            ordered = sorted(self.zones, key=lambda z: z["kind"] == "road")
            for zone in ordered:
                if point_in_polygon(x, y, zone["polygon"]):
                    if boundary_distance_px(x * self.frame_width, y * self.frame_height,
                                            zone["polygon"], self.frame_width, self.frame_height) < opts["boundary_margin_px"]:
                        reason = "zone_boundary"
                        break
                    candidate = ("wrong_path" if zone["kind"] == "road" else "right_path", zone["zone_id"])
                    reason = "on_road_outside_zebra_crossing" if zone["kind"] == "road" else "on_" + zone["kind"]
                    break
            # An outside point near a safe zone is also ambiguous; never turn it
            # into an unsafe alert merely because of a few pixels of box jitter.
            if candidate and candidate[0] == "wrong_path" and any(
                z["kind"] != "road" and boundary_distance_px(x * self.frame_width, y * self.frame_height,
                z["polygon"], self.frame_width, self.frame_height) < opts["boundary_margin_px"] for z in self.zones
            ):
                candidate, reason = None, "zone_boundary"
        if candidate is None:
            state.candidate, state.count = None, 0
            return {**result, "reason": reason}, None
        if candidate != state.candidate:
            state.candidate, state.since, state.count, state.confidence_sum = candidate, ts, 0, 0.0
        state.count += 1
        state.confidence_sum += detection["conf"]
        if state.count < opts["min_observations"] or ts - state.since + 1e-9 < opts["confirm_s"]:
            return {"status": "pending", "reason": reason, "zone_id": candidate[1]}, None
        previous = state.confirmed
        state.confirmed = candidate
        decision = {"status": candidate[0], "reason": reason, "zone_id": candidate[1]}
        if candidate[0] != "wrong_path" or (previous and previous[0] == "wrong_path"):
            return decision, None
        return decision, {
            "type": "pedestrian_wrong_path", "track_id": track_id, "class": "pedestrian",
            "conf": state.confidence_sum / state.count, "zone_id": candidate[1],
            "frame": detection["frame"], "video_ts_s": ts, "bbox": list(detection["bbox"]),
        }
