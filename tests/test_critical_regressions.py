"""Regression tests for the three evidence-integrity failures found in audit."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from visionguard.search import DeterministicQueryPlanner, VideoQueryGraph
from visionguard.video_pipeline.video_pipeline import VisionGuardPipeline
from visionguard.web_app.server import create_app
from visionguard.video_pipeline.vector_index import SegmentVectorIndex


class _SpyVectorIndex:
    def __init__(self):
        self.calls = []

    def search(self, query, k):
        self.calls.append((np.asarray(query), k))
        return np.asarray([0.91], dtype=np.float32), np.asarray([0], dtype=np.uint64)


def test_active_semantic_search_uses_the_built_segment_vector_index(tmp_path):
    pipe = VisionGuardPipeline(out_dir=str(tmp_path / "output"))
    spy = _SpyVectorIndex()
    pipe.search_idx = spy
    pipe._embed_query = lambda _query: np.asarray([1.0, 0.0], dtype=np.float32)
    pipe.idx = {
        "video": "video-a.mp4",
        "meta": {"duration": 5.0, "frame_interval_sec": 0.04, "win_sec": 4.5},
        "frames": [],
        "segments": [{
            "seg_id": np.uint64(0),
            "start": 0.0,
            "end": 4.0,
            "mid": 2.0,
            "frame_path": "stored-a.jpg",
            "objects": [],
            "tracks": [],
            "tags": [],
            "semantic": {
                "caption": "words deliberately unrelated to the request",
                "scene_tags": [],
                "event_tags": [],
                "confidence": 0.8,
            },
        }],
    }

    hits = pipe.search("describe the loading bay", top_k=4)

    assert len(spy.calls) == 1
    assert hits and hits[0]["retrieval_mode"] == "vector_segment"
    assert int(hits[0]["segment_id"]) == 0


def test_time_segments_follow_source_timestamps_after_deduplication():
    frames = [{"ts": 0.0}, {"ts": 1.0}, {"ts": 20.0}, {"ts": 21.0}]

    groups = VisionGuardPipeline._group_frames_by_time(frames, window_seconds=4.5)

    assert [[row["ts"] for row in group] for group in groups] == [[0.0, 1.0], [20.0, 21.0]]


def test_count_query_counts_distinct_supported_tracks(tmp_path):
    pipe = VisionGuardPipeline(out_dir=str(tmp_path / "output"))
    pipe.idx = {
        "video": "count.mp4",
        "meta": {"duration": 5.0, "frame_interval_sec": 0.04, "win_sec": 4.5},
        "frames": [
            {"frame_id": 0, "ts": 0.0, "frame_path": "a.jpg", "objects": ["person"], "tracks": [10], "detections": [{"name": "person", "track_id": 10, "conf": 0.9, "box": [0, 0, 5, 5]}]},
            {"frame_id": 1, "ts": 1.0, "frame_path": "b.jpg", "objects": ["person"], "tracks": [10, 11], "detections": [{"name": "person", "track_id": 10, "conf": 0.9, "box": [0, 0, 5, 5]}, {"name": "person", "track_id": 11, "conf": 0.8, "box": [5, 0, 10, 5]}]},
        ],
        "segments": [],
        "events": [],
    }

    hits = pipe.search("how many people are there?", top_k=4)

    assert hits and hits[0]["count"] == 2
    assert hits[0]["evidence_state"] == "detector_fact"
    assert pipe.last_query_message == "Found 2 distinct tracked person object(s)."


def test_temporal_query_filters_detector_evidence_by_stored_timestamp(tmp_path):
    pipe = VisionGuardPipeline(out_dir=str(tmp_path / "output"))
    pipe._query_detector_classes = lambda _query: ([0], {0: "person"})
    pipe._q_objs = lambda _query: ["person"]
    pipe._query_colors = lambda _query: []
    pipe.idx = {
        "video": "time.mp4",
        "meta": {"duration": 10.0, "frame_interval_sec": 0.04, "win_sec": 1.0},
        "frames": [
            {"frame_id": 0, "ts": 2.0, "frame_path": "early.jpg", "objects": ["person"], "tracks": [1], "appearances": [], "detections": [{"name": "person", "cls": 0, "track_id": 1, "conf": 0.9, "box": [0, 0, 5, 5]}]},
            {"frame_id": 1, "ts": 8.0, "frame_path": "late.jpg", "objects": ["person"], "tracks": [2], "appearances": [], "detections": [{"name": "person", "cls": 0, "track_id": 2, "conf": 0.9, "box": [0, 0, 5, 5]}]},
        ],
        "segments": [],
        "events": [],
    }

    hits = pipe.search("find person before 5 seconds", top_k=4)

    assert [hit["peak_ts"] for hit in hits] == [2.0]


def test_temporal_query_can_use_a_stored_reference_event(tmp_path):
    pipe = VisionGuardPipeline(out_dir=str(tmp_path / "output"))
    pipe._query_detector_classes = lambda _query: ([0], {0: "person"})
    pipe._q_objs = lambda _query: ["person"]
    pipe._query_colors = lambda _query: []
    pipe.idx = {
        "video": "event-time.mp4",
        "meta": {"duration": 10.0, "frame_interval_sec": 0.04, "win_sec": 1.0},
        "frames": [
            {"frame_id": 0, "ts": 2.0, "frame_path": "early.jpg", "objects": ["person"], "tracks": [1], "appearances": [], "detections": [{"name": "person", "cls": 0, "track_id": 1, "conf": 0.9, "box": [0, 0, 5, 5]}]},
            {"frame_id": 1, "ts": 8.0, "frame_path": "late.jpg", "objects": ["person"], "tracks": [2], "appearances": [], "detections": [{"name": "person", "cls": 0, "track_id": 2, "conf": 0.9, "box": [0, 0, 5, 5]}]},
        ],
        "segments": [],
        "events": [{"type": "zone_entered", "timestamp": 5.0, "class_name": "car"}],
    }

    hits = pipe.search("find person after car entered", top_k=4)

    assert [hit["peak_ts"] for hit in hits] == [8.0]


def test_pipeline_snapshot_restores_video_scoped_vectors(tmp_path):
    pipe = VisionGuardPipeline(out_dir=str(tmp_path / "output"))
    pipe.idx = {"video": "a.mp4", "frames": [], "segments": []}
    pipe.frame_idx = SegmentVectorIndex()
    pipe.search_idx = SegmentVectorIndex()
    pipe.crop_idx = SegmentVectorIndex()
    pipe.search_idx.build(np.asarray([[1.0, 0.0]], dtype=np.float32), np.asarray([7], dtype=np.uint64))
    snapshot_a = pipe.capture_snapshot()

    pipe.idx = {"video": "b.mp4", "frames": [], "segments": []}
    pipe.search_idx.build(np.asarray([[0.0, 1.0]], dtype=np.float32), np.asarray([9], dtype=np.uint64))
    pipe.activate_snapshot(snapshot_a)
    _scores, ids = pipe.search_idx.search(np.asarray([1.0, 0.0], dtype=np.float32), 1)

    assert pipe.idx["video"] == "a.mp4"
    assert ids.tolist() == [7]


def test_rejected_explicit_verification_cannot_produce_an_affirmative_answer():
    evidence = [{"peak_ts": 2.0, "frame_path": "stored.jpg"}]
    graph = VideoQueryGraph(
        DeterministicQueryPlanner(),
        lambda *_: evidence,
        lambda *_: evidence,
        lambda *_: evidence,
        lambda *_: evidence,
        verify=lambda _query, _evidence: {"state": "rejected", "confirmed": False},
    )

    result = graph.invoke("confirm the unusual activity", [])

    assert result["verification"]["confirmed"] is False
    assert result["abstained"] is True
    assert result["evidence"] == []
    assert result["answer"] == "Insufficient grounded evidence found for this request."


class _RacePipeline:
    """A controlled pipeline that exposes mutation before its second run completes."""

    def __init__(self, root: Path):
        self.root = root
        self.idx = None
        self.last_hits = []
        self.last_query_plan = {"intent": "object_search"}
        self.last_query_message = ""
        self.zero_query = None
        self.second_started = threading.Event()
        self.release_second = threading.Event()
        self.calls = 0

    def verification_mode(self):
        return "verification_disabled"

    def embedding_mode(self):
        return "semantic_embeddings"

    def index_video_iter(self, video_path):
        self.calls += 1
        label = f"video-{self.calls}"
        image_path = self.root / f"{label}.jpg"
        image = np.full((24, 32, 3), 40 * self.calls, dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)
        detection = {"box": [1, 1, 20, 20], "conf": 0.9, "cls": 0, "name": label}
        self.idx = {
            "video": str(video_path),
            "meta": {"fps": 25.0, "duration": 1.0},
            "frames": [{
                "frame_id": 0,
                "frame": 0,
                "ts": 0.0,
                "frame_path": str(image_path),
                "objects": [label],
                "tracks": [],
                "detections": [detection],
                "deduplication": "unique",
            }],
            "segments": [],
        }
        if self.calls == 2:
            self.second_started.set()
            assert self.release_second.wait(timeout=5)
        yield {"kind": "done", "meta": {
            "nonzero_frame_vectors": 1,
            "embedding_mode": "semantic_embeddings",
            "segments": 1,
            "retriever": "test-vector-index",
            "object_counts": {label: 1},
            "scan_timings": {"total": 0.01},
        }}

    def search(self, query, top_k=4):
        frame = self.idx["frames"][0]
        return [{
            "start": 0.0,
            "end": 0.1,
            "peak_ts": 0.0,
            "score": 0.9,
            "objects": frame["objects"],
            "tracks": [],
            "summary": frame["objects"][0],
            "representative_frame_path": frame["frame_path"],
        }]

    def prepare_hits(self, hits, query):
        self.last_hits = [{**hit, "label": "1. 0.00s"} for hit in hits]
        return self.last_hits


def test_query_is_rejected_while_another_video_mutates_the_shared_index(tmp_path):
    pipeline = _RacePipeline(tmp_path)
    app = create_app(testing=True, start_warmup=False, pipeline=pipeline)
    client = app.test_client()

    first = client.post("/api/videos/upload", json={"sample": "asset3.mp4"}).get_json()
    assert client.post(f"/api/videos/{first['video_id']}/index").status_code == 202
    app.config["PROCESS_THREADS"][first["job_id"]].join(timeout=5)
    assert client.get(f"/api/videos/{first['video_id']}/status").get_json()["status"] == "completed"

    second = client.post("/api/videos/upload", json={"sample": "asset4.mp4"}).get_json()
    assert client.post(f"/api/videos/{second['video_id']}/index").status_code == 202
    assert pipeline.second_started.wait(timeout=2)
    try:
        response = client.post(
            f"/api/videos/{first['video_id']}/query",
            json={"query": "find video one"},
        )
        assert response.status_code == 409
        assert "index" in response.get_json()["message"].casefold()
    finally:
        pipeline.release_second.set()
        app.config["PROCESS_THREADS"][second["job_id"]].join(timeout=5)
