from pathlib import Path

from pipeline import VisionGuardPipeline


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
    from report_generator import ReportGenerator

    pipe.rep = ReportGenerator(run_dir / "reports")
    pipe.idx = {"video": "assets/asset3.mp4"}
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
