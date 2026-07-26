from types import SimpleNamespace

from visionguard.model_services.tracker import ObjectTracker


class _Values:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values

    def int(self):
        return self


def test_tracking_inference_keeps_untracked_detections(monkeypatch):
    boxes = SimpleNamespace(
        xyxy=_Values([[1, 2, 10, 20], [3, 4, 12, 24]]),
        conf=_Values([0.9, 0.8]),
        cls=_Values([0, 2]),
        id=None,
    )
    model = SimpleNamespace(
        names={0: "person", 2: "car"},
        track=lambda *args, **kwargs: [SimpleNamespace(boxes=boxes)],
    )
    tracker = ObjectTracker()
    monkeypatch.setattr(tracker, "load", lambda: setattr(tracker, "m", model))

    rows = tracker.track([[0]])

    assert [row["name"] for row in rows] == ["person", "car"]
    assert all("id" not in row for row in rows)
