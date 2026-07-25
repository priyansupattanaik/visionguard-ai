import numpy as np

from visionguard.evidence_api.agents.planner import DeterministicQueryPlanner
from visionguard.model_services.metadata_encoder import MetadataSearchEncoder


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
    query = encoder.encode_text("find the red vehicle")

    assert np.linalg.norm(red_car) > 0.99
    assert np.linalg.norm(query) > 0.99
    assert float(red_car @ query) > float(blue_person @ query)


def test_planner_resolves_human_language_to_detector_entities():
    plan = DeterministicQueryPlanner().plan(
        "show me someone carrying a bag",
        detector_labels=["person", "backpack", "handbag", "suitcase"],
    )

    assert "person" in plan.entities
    assert {"backpack", "handbag", "suitcase"}.issubset(set(plan.entities))
    assert plan.intent == "event_search"
    assert "visual_semantic" in plan.retrieval_routes


def test_planner_requests_clarification_for_attribute_without_object():
    plan = DeterministicQueryPlanner().plan("show me the yellow one")

    assert plan.intent == "ambiguous"
    assert plan.clarification


def test_planner_routes_supported_track_event_without_semantic_guessing():
    plan = DeterministicQueryPlanner().plan("where did the person enter?", detector_labels=["person"])

    assert plan.entities == ["person"]
    assert "track_events" in plan.retrieval_routes
    assert "visual_semantic" not in plan.retrieval_routes
