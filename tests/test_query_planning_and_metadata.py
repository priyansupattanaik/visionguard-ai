import numpy as np

from visionguard.model_services.metadata_encoder import MetadataSearchEncoder
from visionguard.model_services.model_provider import NoneModelProvider
from visionguard.search import DeterministicQueryPlanner
from visionguard.video_pipeline.video_pipeline import VisionGuardPipeline


def test_metadata_embeddings_are_nonzero_and_query_aligned():
    encoder = MetadataSearchEncoder(dimension=256)
    red_car = encoder.encode_metadata({
        "objects": {"car": 1},
        "appearances": ["red car"],
        "detections": [{"name": "car", "color": "red"}],
    })
    blue_person = encoder.encode_metadata({
        "objects": {"person": 1},
        "appearances": ["blue person"],
        "detections": [{"name": "person", "color": "blue"}],
    })
    query = encoder.encode_text("find the red car")

    assert np.linalg.norm(red_car) > 0.99
    assert np.linalg.norm(query) > 0.99
    assert float(red_car @ query) > float(blue_person @ query)


def test_planner_does_not_use_a_finite_object_alias_table():
    plan = DeterministicQueryPlanner().plan(
        "show me someone carrying a bag",
        detector_labels=["person", "backpack", "handbag", "suitcase"],
    )

    assert plan.entities == []
    assert plan.intent == "event_search"
    assert "visual_semantic" in plan.retrieval_routes


def test_planner_routes_open_attribute_description_without_guessing_an_object():
    plan = DeterministicQueryPlanner().plan("show me the yellow one")

    assert plan.entities == []
    assert "visual_semantic" in plan.retrieval_routes


def test_planner_discovers_custom_detector_label_and_generic_plural():
    planner = DeterministicQueryPlanner()

    assert planner.resolve_entities("find the forklift", ["forklift"]) == ["forklift"]
    assert planner.resolve_entities("find the forklifts", ["forklift"]) == ["forklift"]


def test_planner_routes_supported_track_event_without_semantic_guessing():
    plan = DeterministicQueryPlanner().plan("where did the person enter?", detector_labels=["person"])

    assert plan.entities == ["person"]
    assert "track_events" in plan.retrieval_routes
    assert "visual_semantic" not in plan.retrieval_routes


def test_planner_resolves_common_aliases_to_supported_detector_classes():
    planner = DeterministicQueryPlanner()

    assert planner.resolve_entities("show pedestrians", ["person", "car"]) == ["person"]
    assert planner.resolve_entities("find vehicles", ["person", "car", "truck", "motorcycle"]) == ["car", "motorcycle", "truck"]


def test_color_metadata_is_generated_for_a_runtime_discovered_class(tmp_path):
    pipe = VisionGuardPipeline(out_dir=str(tmp_path / "output"))
    pipe.model_provider = NoneModelProvider()
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 255)

    tags = pipe._appearance_tags(frame, [{"name": "custom crate", "box": [5, 5, 75, 75]}])

    assert "custom crate" in tags
    assert "red custom crate" in tags


def test_open_query_uses_bounded_visual_verification_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "configured-for-test")
    pipe = VisionGuardPipeline(out_dir=str(tmp_path / "output"))
    pipe.model_provider = NoneModelProvider()
    pipe.trk.names = lambda: {0: "known class"}
    pipe.enc.fallback = True
    pipe.ver.warmup()
    pipe.idx = {
        "video": "test.mp4",
        "meta": {"duration": 10.0, "frame_interval_sec": 0.04},
        "frames": [{
            "frame_id": 0,
            "ts": 2.0,
            "frame_path": "frame.jpg",
            "objects": ["known class"],
            "tracks": [],
            "appearances": [],
        }],
        "segments": [],
    }

    plan = pipe.plan_query("find an unfamiliar object near the doorway")
    candidates = pipe._exhaustive_visual_candidates("find an unfamiliar object near the doorway")

    assert plan["executable"] is True
    assert "exhaustive_visual_verification" in plan["retrieval_routes"]
    assert len(candidates) == 1
    assert candidates[0]["retrieval_mode"] == "exhaustive_visual_verification"


def test_detector_retrieval_returns_a_calibrated_evidence_segment(tmp_path, monkeypatch):
    monkeypatch.setenv("MIN_EVIDENCE_CONFIDENCE", "0.25")
    pipe = VisionGuardPipeline(out_dir=str(tmp_path / "output"))
    pipe._query_detector_classes = lambda query: ([2], {2: "car"})
    pipe._q_objs = lambda query: ["car"]
    pipe._query_colors = lambda query: []
    pipe.idx = {
        "meta": {"duration": 12.0, "frame_interval_sec": 0.04, "win_sec": 3.0},
        "frames": [
            {"frame_id": 1, "ts": 2.0, "frame_path": "first.jpg", "objects": ["car"], "tracks": [1], "appearances": [], "detections": [{"name": "car", "cls": 2, "box": [0, 0, 5, 5], "conf": 0.8}]},
            {"frame_id": 2, "ts": 4.0, "frame_path": "second.jpg", "objects": ["car"], "tracks": [1], "appearances": [], "detections": [{"name": "car", "cls": 2, "box": [0, 0, 5, 5], "conf": 0.9}]},
            {"frame_id": 3, "ts": 10.0, "frame_path": "third.jpg", "objects": ["car"], "tracks": [2], "appearances": [], "detections": [{"name": "car", "cls": 2, "box": [0, 0, 5, 5], "conf": 0.3}]},
        ],
    }

    hits = pipe._refine_detector_hits("find car", top_k=4)

    assert len(hits) == 2
    assert hits[0]["peak_ts"] == 4.0
    assert hits[0]["start"] == 1.96
    assert hits[0]["end"] == 4.04
    assert hits[0]["cache_key"].startswith("detector-segment:")
