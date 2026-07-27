from pathlib import Path

from visionguard.video_pipeline.video_pipeline import VisionGuardPipeline
from visionguard.model_services.report_generator import ReportGenerator
from visionguard.model_services.segmenter import GroundedSegmenter


class FakeClip:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def clip_path(self, video, st, ed, name, pad=1.5):
        return str(self.out_dir / f"{name}.mp4")

    def extract_clip(self, video, st, ed, name, pad=1.5):
        path = Path(self.clip_path(video, st, ed, name, pad=pad))
        path.write_bytes(b"raw clip")
        return str(path)


def test_export_falls_back_to_raw_clip_when_segmentation_fails(tmp_path):
    pipe = VisionGuardPipeline(out_dir=str(tmp_path / "output"))
    run_dir = tmp_path / "output" / "run"
    (run_dir / "clips").mkdir(parents=True)
    (run_dir / "reports").mkdir(parents=True)
    (run_dir / "segments").mkdir(parents=True)
    pipe.run_dir = str(run_dir)
    pipe.clip = FakeClip(run_dir / "clips")
    from visionguard.model_services.report_generator import ReportGenerator

    pipe.rep = ReportGenerator(run_dir / "reports")
    pipe.idx = {"video": "sample_videos/asset3.mp4"}
    pipe.last_hits = [{
        "match_id": 1,
        "label": "1. 5.33s",
        "start": 4.38,
        "end": 5.88,
        "score": 0.9,
        "summary": "detector-only dev passthrough",
        "objects": ["person"],
        "tracks": [],
        "clip": None,
        "raw_clip": None,
        "frames": [],
        "segmented": False,
        "query": "person",
        "det_boxes": [],
    }]

    def fail_segment(row, query):
        raise RuntimeError("SAM2 unavailable")

    pipe._segment_payload = fail_segment
    result = pipe.export_selected_detailed(["1. 5.33s"], "person", segment_timeout=0.1)
    assert result["ok"] is True
    assert result["export_mode"] == "raw_fallback"
    assert "Raw clip export fallback" in result["message"]
    assert Path(result["files"]["zip"]).exists()
    assert Path(result["files"]["html"]).exists()
    assert Path(result["files"]["csv"]).exists()


def test_html_report_autoescapes_user_and_model_text(tmp_path):
    report = ReportGenerator(tmp_path)
    path = report.write_html(tmp_path / "report.html", {
        "query": '<script>alert("query")</script>',
        "video": "video.mp4",
        "hits": [{"start": 0.0, "end": 1.0, "score": 0.5, "summary": '<img src=x onerror=alert("summary")>', "objects": [], "clip": ""}],
    })

    html = Path(path).read_text(encoding="utf-8")
    assert "<script>" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;" in html


def test_segmenter_does_not_fabricate_confidence_for_boxes(tmp_path):
    class BoxVerifier:
        @staticmethod
        def ground_phrase(*_args, **_kwargs):
            return [[1, 2, 10, 12]]

    segmenter = GroundedSegmenter(verifier=BoxVerifier())
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"not-read-by-this-test")

    boxes, scores, _texts = segmenter.detect(str(frame_path), "person")

    assert boxes == [[1, 2, 10, 12]]
    assert scores == [None]
