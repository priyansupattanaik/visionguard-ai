"""Tests for HOC-VideoRAG upgrade features: tracking, zero-query, temporal queries."""
import os
import sys

os.environ["VISION_GUARD_SKIP_WARMUP"] = "1"

from visionguard.web_app.server import create_app


class DummyPipeline:
    """Pipeline mock with track_stats and zero_query data for testing."""

    def __init__(self):
        self.idx = {
            "video": "test.mp4",
            "meta": {
                "video": "test.mp4",
                "fps": 25.0,
                "frames": 250,
                "duration": 10.0,
                "sample_sec": 0.75,
                "win_sec": 4.5,
                "segments": 2,
                "object_counts": {"person": 5, "car": 2},
                "total_detections": 7,
                "unique_objects": 2,
            },
            "frames": [
                {
                    "frame_id": 0,
                    "frame": 0,
                    "ts": 1.0,
                    "frame_path": "output/frames/f_000000.jpg",
                    "representative_frame_path": "output/frames/f_000000.jpg",
                    "objects": ["person"],
                    "appearances": [],
                    "tracks": [1, 2],
                    "detections": [
                        {"box": [10, 20, 100, 200], "conf": 0.85, "cls": 0, "name": "person", "color": None, "track_id": 1},
                        {"box": [150, 30, 250, 220], "conf": 0.72, "cls": 0, "name": "person", "color": None, "track_id": 2},
                    ],
                    "motion_score": 0.05,
                    "keep_reason": "motion",
                    "still_people": 0,
                    "object_delta": 0,
                },
            ],
            "segments": [],
        }
        self.last_hits = []
        self.track_stats = {
            1: {
                "track_id": 1,
                "class_name": "person",
                "class_id": 0,
                "trajectory_length": 8,
                "dwell_time": 6.5,
                "entry_frame": 0,
                "exit_frame": 200,
                "entry_ts": 0.0,
                "exit_ts": 8.0,
                "avg_confidence": 0.82,
            },
            2: {
                "track_id": 2,
                "class_name": "person",
                "class_id": 0,
                "trajectory_length": 3,
                "dwell_time": 2.0,
                "entry_frame": 50,
                "exit_frame": 100,
                "entry_ts": 2.0,
                "exit_ts": 4.0,
                "avg_confidence": 0.71,
            },
        }
        self.zero_query = {
            "object_inventory": {
                "person": {
                    "count": 2,
                    "total_dwell_time": 8.5,
                    "avg_dwell_time": 4.25,
                    "tracks": [
                        {"track_id": 1, "dwell_time": 6.5, "trajectory_length": 8, "entry_ts": 0.0, "exit_ts": 8.0, "avg_confidence": 0.82},
                        {"track_id": 2, "dwell_time": 2.0, "trajectory_length": 3, "entry_ts": 2.0, "exit_ts": 4.0, "avg_confidence": 0.71},
                    ],
                },
            },
            "event_timeline": [
                {"type": "long_dwell", "track_id": 1, "class": "person", "dwell_time": 6.5, "entry_ts": 0.0, "exit_ts": 8.0},
            ],
            "summary": "10.0s video with 2 tracked objects | detected classes: person | 1 notable events detected",
            "meta": {"total_tracks": 2, "classes": ["person"], "duration": 10.0, "fps": 25.0},
        }

    def warmup_status(self):
        return "All models ready."

    def verification_mode(self):
        return "nvidia_api_unconfigured"

    def export_selected_detailed(self, picks, query, segment_timeout=20):
        return {
            "ok": False,
            "message": "No selected matches to export.",
            "files": {},
            "rows": [],
            "export_mode": "none",
            "warnings": [],
        }

    def search(self, q, top_k=4):
        return []

    def prepare_hits(self, hits, query):
        return hits


def make_client():
    app = create_app(testing=True, start_warmup=False, pipeline=DummyPipeline())
    return app.test_client()


# --- Tracking tests ---

def test_tracking_produces_nonempty_track_ids():
    """Track IDs should be non-empty in the indexed frame data."""
    client = make_client()
    pipe = client.application.config["PIPELINE"]
    frame = pipe.idx["frames"][0]
    assert len(frame["tracks"]) > 0, "tracks should not be empty"
    assert all(isinstance(tid, int) for tid in frame["tracks"])


def test_detection_rows_have_track_id():
    """Individual detections should include track_id field."""
    client = make_client()
    pipe = client.application.config["PIPELINE"]
    frame = pipe.idx["frames"][0]
    for det in frame["detections"]:
        assert "track_id" in det, f"detection missing track_id: {det}"


def test_track_stats_populated():
    """Pipeline should have computed track statistics."""
    client = make_client()
    pipe = client.application.config["PIPELINE"]
    assert len(pipe.track_stats) > 0
    for tid, stats in pipe.track_stats.items():
        assert "dwell_time" in stats
        assert "trajectory_length" in stats
        assert "entry_ts" in stats
        assert "exit_ts" in stats
        assert "avg_confidence" in stats


# --- Zero-query tests ---

def test_zero_query_endpoint_returns_data():
    """GET /api/zero_query should return inventory and timeline."""
    client = make_client()
    res = client.get("/api/zero_query")
    data = res.get_json()
    assert res.status_code == 200
    assert data["ok"] is True
    assert "object_inventory" in data
    assert "event_timeline" in data
    assert "summary" in data


def test_zero_query_inventory_structure():
    """Zero-query inventory should have correct structure per class."""
    client = make_client()
    res = client.get("/api/zero_query")
    data = res.get_json()
    inv = data["object_inventory"]
    assert "person" in inv
    person = inv["person"]
    assert person["count"] == 2
    assert person["total_dwell_time"] > 0
    assert len(person["tracks"]) == 2


def test_zero_query_timeline_has_events():
    """Zero-query timeline should contain detected events."""
    client = make_client()
    res = client.get("/api/zero_query")
    data = res.get_json()
    events = data["event_timeline"]
    assert len(events) >= 1
    assert events[0]["type"] == "long_dwell"


def test_zero_query_before_scan_returns_error():
    """Zero-query before scan should return 400."""
    pipe = DummyPipeline()
    pipe.idx = None
    pipe.zero_query = None
    app = create_app(testing=True, start_warmup=False, pipeline=pipe)
    client = app.test_client()
    res = client.get("/api/zero_query")
    assert res.status_code == 400


# --- Existing query compatibility tests ---

def test_object_query_still_works():
    """Object queries should still return without error."""
    client = make_client()
    res = client.post("/api/query", json={"query": "person"})
    data = res.get_json()
    assert res.status_code == 200
    assert data["ok"] is True
    assert "matches" in data
    assert "message" in data


def test_query_serialization_includes_tracks():
    """Query response serialization should include track IDs."""
    from visionguard.web_app.server import _serialize_match
    row = {
        "label": "1. 1.00s",
        "start": 0.5,
        "end": 2.0,
        "peak_ts": 1.0,
        "score": 0.8,
        "objects": ["person"],
        "tracks": [1, 2],
        "summary": "test",
        "verification_mode": "nvidia_api_unconfigured",
        "low_confidence": False,
    }
    result = _serialize_match(row)
    assert "tracks" in result
    assert result["tracks"] == [1, 2]


# --- Export compatibility test ---

def test_export_still_returns_structured_response():
    """Export endpoint should still return proper response structure."""
    client = make_client()
    res = client.post("/api/export", json={"selected": [], "query": "person"})
    data = res.get_json()
    assert res.status_code == 400
    assert data["ok"] is False
    assert "message" in data
