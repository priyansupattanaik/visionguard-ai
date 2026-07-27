"""Derive bounded, detector-grounded temporal events from tracked evidence."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import hypot
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Zone:
    name: str
    left: float
    top: float
    right: float
    bottom: float

    def contains(self, x: float, y: float, width: float, height: float) -> bool:
        return self.left <= x / max(width, 1.0) <= self.right and self.top <= y / max(height, 1.0) <= self.bottom


class EventExtractor:
    """Create only events whose timestamps are linked to tracked source frames."""

    def __init__(self, zones: Iterable[dict] = (), min_dwell_seconds: float = 2.0):
        self.zones = tuple(self._zone(row) for row in zones)
        self.min_dwell_seconds = max(0.0, float(min_dwell_seconds))

    @staticmethod
    def _zone(row: dict) -> Zone:
        try:
            zone = Zone(str(row["name"]).strip(), float(row["left"]), float(row["top"]), float(row["right"]), float(row["bottom"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each zone needs name, left, top, right, and bottom values.") from exc
        if not zone.name or not (0 <= zone.left < zone.right <= 1 and 0 <= zone.top < zone.bottom <= 1):
            raise ValueError(f"Zone {zone.name or '<unnamed>'!r} must use normalized bounds between 0 and 1.")
        return zone

    @staticmethod
    def _event(kind: str, observation: dict, *, track_id: int, class_name: str, **extra) -> dict:
        return {"event_id": f"{kind}:{track_id}:{observation['frame_id']}", "type": kind,
                "timestamp": float(observation["ts"]), "frame_id": int(observation["frame_id"]),
                "frame_path": observation["frame_path"], "track_id": int(track_id), "class_name": class_name,
                "evidence_state": "event_fact", "claim_provenance": "yolo_botsort_event_graph", **extra}

    def extract(self, frames: list[dict], frame_width: int, frame_height: int) -> list[dict]:
        observations: dict[int, list[dict]] = defaultdict(list)
        for frame in frames:
            for detection in frame.get("detections", []):
                track_id, box = detection.get("track_id"), detection.get("box") or []
                if track_id is None or len(box) != 4:
                    continue
                x1, y1, x2, y2 = (float(value) for value in box)
                observations[int(track_id)].append({"frame_id": frame["frame_id"], "ts": frame["ts"],
                    "frame_path": frame["frame_path"], "class_name": str(detection.get("name", "unknown")),
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0)})
        events = []
        for track_id, rows in observations.items():
            rows.sort(key=lambda row: (row["ts"], row["frame_id"]))
            class_name = rows[0]["class_name"]
            events.append(self._event("track_appeared", rows[0], track_id=track_id, class_name=class_name))
            if len(rows) > 1:
                first, last = rows[0], rows[-1]
                dx, dy = last["center"][0] - first["center"][0], last["center"][1] - first["center"][1]
                distance = hypot(dx, dy)
                movement_threshold = max(frame_width, frame_height) * 0.03
                if distance >= movement_threshold:
                    direction = "right" if abs(dx) >= abs(dy) and dx > 0 else "left" if abs(dx) >= abs(dy) else "down" if dy > 0 else "up"
                    events.append(self._event("track_moved", last, track_id=track_id, class_name=class_name, direction=direction,
                                              distance_px=round(distance, 2), start_timestamp=float(first["ts"])))
                maximum_observation_gap = max(2.0, self.min_dwell_seconds * 2.0)
                dwell = sum(
                    max(0.0, current["ts"] - previous["ts"])
                    for previous, current in zip(rows, rows[1:])
                    if current["ts"] - previous["ts"] <= maximum_observation_gap
                )
                if dwell >= self.min_dwell_seconds:
                    events.append(self._event("track_dwell", last, track_id=track_id, class_name=class_name,
                                              duration_seconds=round(dwell, 3), start_timestamp=float(first["ts"])))
            for zone in self.zones:
                previous = None
                for row in rows:
                    inside = zone.contains(*row["center"], frame_width, frame_height)
                    if previous is not None and inside != previous:
                        events.append(self._event("zone_entered" if inside else "zone_exited", row, track_id=track_id,
                                                  class_name=class_name, zone=zone.name))
                    previous = inside
        return sorted(events, key=lambda event: (event["timestamp"], event["event_id"]))
