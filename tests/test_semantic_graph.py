from visionguard.search import DeterministicQueryPlanner, VideoQueryGraph
from visionguard.semantic.events import EventExtractor
from visionguard.search.query_planner import load_query_rules


def test_event_extractor_preserves_source_frames_and_zone_transitions():
    frames = [
        {"frame_id": 1, "ts": 0.0, "frame_path": "f1.jpg", "detections": [{"track_id": 7, "name": "car", "box": [0, 40, 20, 60]}]},
        {"frame_id": 2, "ts": 2.0, "frame_path": "f2.jpg", "detections": [{"track_id": 7, "name": "car", "box": [45, 40, 65, 60]}]},
        {"frame_id": 3, "ts": 4.0, "frame_path": "f3.jpg", "detections": [{"track_id": 7, "name": "car", "box": [85, 40, 100, 60]}]},
    ]
    events = EventExtractor([{"name": "gate", "left": 0.4, "top": 0.0, "right": 0.7, "bottom": 1.0}], 1.0).extract(frames, 100, 100)

    entered = next(event for event in events if event["type"] == "zone_entered")
    assert entered["timestamp"] == 2.0
    assert entered["frame_id"] == 2
    assert entered["frame_path"] == "f2.jpg"
    assert entered["evidence_state"] == "event_fact"
    assert entered["claim_provenance"] == "yolo_botsort_event_graph"


def test_langgraph_routes_event_query_without_using_semantic_retrieval():
    calls = []

    def record(name):
        def route(plan, top_k):
            calls.append(name)
            return [{"peak_ts": 1.25, "source": name}]
        return route

    graph = VideoQueryGraph(DeterministicQueryPlanner(), record("object"), record("event"), record("zone"), record("semantic"))
    result = graph.invoke("when did the car enter", ["car"])

    assert calls == ["event"]
    assert result["abstained"] is False
    assert result["evidence"][0]["source"] == "event"


def test_langgraph_abstains_when_the_selected_route_has_no_evidence():
    graph = VideoQueryGraph(DeterministicQueryPlanner(), lambda *_: [], lambda *_: [], lambda *_: [], lambda *_: [])
    result = graph.invoke("describe the scene", ["car"])

    assert result["abstained"] is True
    assert result["answer"] == "Insufficient grounded evidence found for this request."


def test_stationary_track_is_not_mislabeled_as_movement_or_disappearance_at_video_end():
    frames = [
        {"frame_id": 1, "ts": 0.0, "frame_path": "f1.jpg", "detections": [{"track_id": 2, "name": "person", "box": [10, 10, 30, 50]}]},
        {"frame_id": 2, "ts": 1.0, "frame_path": "f2.jpg", "detections": [{"track_id": 2, "name": "person", "box": [10, 10, 30, 50]}]},
    ]

    events = EventExtractor([], 0.5).extract(frames, 100, 100)
    kinds = {event["type"] for event in events}

    assert "track_moved" not in kinds
    assert "track_disappeared" not in kinds
    assert "track_dwell" in kinds


def test_query_planner_marks_unimplemented_events_for_precise_abstention():
    plan = DeterministicQueryPlanner().plan("find when the person fell", ["person"])

    assert plan.intent == "unsupported_event"
    assert "fell" in plan.events
    assert plan.limitations


def test_every_advertised_event_is_implemented_or_explicitly_unsupported():
    rules = load_query_rules()
    implemented = set(rules["track_events"])
    planner = DeterministicQueryPlanner()

    for event in rules["events"]:
        plan = planner.plan(f"find person {event}", ["person"])
        if event in implemented:
            assert plan.intent == "event_search"
        else:
            assert plan.intent == "unsupported_event"
            assert plan.limitations
