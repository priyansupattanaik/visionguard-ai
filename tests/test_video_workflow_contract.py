import cv2
import numpy as np
from pathlib import Path

from visionguard.web_app.server import create_app
from visionguard.web_app.video_jobs import make_chunks, probe_video


class ContractPipeline:
    """Fast test-only pipeline for exercising the HTTP resource contract."""

    def __init__(self, image_path):
        self.image_path = str(image_path)
        self.idx = None
        self.last_hits = []
        self.last_query_plan = {"executable": True}
        self.last_query_message = ""
        self.zero_query = None

    def warmup_status(self):
        return "Ready"

    def verification_mode(self):
        return "nvidia_api_unconfigured"

    def embedding_mode(self):
        return "metadata_embeddings"

    def index_video_iter(self, video_path):
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        frame[:, :] = (20, 40, 180)
        assert cv2.imwrite(self.image_path, frame)
        detection = {"box": [2, 3, 20, 28], "conf": 0.9, "cls": 0, "name": "test class", "color": "red"}
        yield {
            "kind": "preview",
            "status": "scanning 0.0s / 1.0s | 100%",
            "frame_number": 0,
            "timestamp_ms": 0,
            "processed_samples": 1,
            "total_samples": 1,
            "kept_frames": 1,
            "detections": 1,
            "frame_path": self.image_path,
            "objects": ["test class"],
            "tracks": [],
            "detection_rows": [detection],
            "motion_score": 0.2,
            "selection_reason": "initial",
        }
        self.idx = {
            "video": video_path,
            "meta": {"fps": 25.0, "duration": 1.0},
            "frames": [{
                "frame_id": 0,
                "frame": 0,
                "ts": 0.0,
                "frame_path": self.image_path,
                "objects": ["test class"],
                "tracks": [],
                "detections": [detection],
                "motion_score": 0.2,
                "keep_reason": "initial",
            }],
            "segments": [],
        }
        yield {"kind": "done", "meta": {
            "nonzero_frame_vectors": 1,
            "embedding_mode": "metadata_embeddings",
            "segments": 1,
            "retriever": "test-index",
            "object_counts": {"test class": 1},
            "scan_timings": {"total": 0.01},
        }}

    def search(self, query, top_k=4):
        if "absent" in query:
            return []
        return [{
            "start": 0.0,
            "end": 0.5,
            "peak_ts": 0.0,
            "score": 0.9,
            "objects": ["test class"],
            "tracks": [],
            "summary": "test-only stored evidence",
            "verification_mode": "nvidia_api_unconfigured",
            "representative_frame_path": self.image_path,
        }]

    def prepare_hits(self, hits, query):
        rows = []
        for index, hit in enumerate(hits, 1):
            rows.append({**hit, "label": f"{index}. 0.00s"})
        self.last_hits = rows
        return rows


def test_real_sample_metadata_and_overlapping_chunks():
    metadata = probe_video(Path("sample_videos/asset3.mp4"))
    assert metadata["duration_ms"] > 0
    assert metadata["fps"] > 0
    assert metadata["width"] > 0
    assert metadata["height"] > 0
    assert metadata["frame_count"] > 0
    assert metadata["file_size"] > 0

    chunks = make_chunks("video_test", {**metadata, "duration_ms": 70_000, "frame_count": 1750, "fps": 25.0})
    assert [(row["start_ms"], row["end_ms"]) for row in chunks] == [(0, 30_000), (25_000, 55_000), (50_000, 70_000)]
    assert chunks[1]["start_frame"] == 625


def test_video_job_frame_and_query_contract(tmp_path):
    pipeline = ContractPipeline(tmp_path / "evidence.jpg")
    app = create_app(testing=True, start_warmup=False, pipeline=pipeline)
    client = app.test_client()

    upload = client.post("/api/videos/upload", json={"sample": "asset3.mp4"})
    assert upload.status_code == 201
    identifiers = upload.get_json()
    assert identifiers["video_id"].startswith("video_")
    assert identifiers["job_id"].startswith("job_")
    assert identifiers["status"] == "uploaded"

    before_index = client.get(f"/api/videos/{identifiers['video_id']}/status").get_json()
    assert before_index["status"] == "waiting"
    indexed = client.post(f"/api/videos/{identifiers['video_id']}/index")
    assert indexed.status_code == 202

    app.config["PROCESS_THREADS"][identifiers["job_id"]].join(timeout=5)
    status = client.get(f"/api/videos/{identifiers['video_id']}/status").get_json()
    assert status["status"] == "completed"
    by_name = {row["name"]: row for row in status["stages"]}
    assert by_name["query_ready"]["status"] == "completed"
    assert by_name["duplicates_removed"]["status"] == "completed"
    assert by_name["frame_metadata_collected"]["status"] == "completed"
    assert by_name["ocr_completed"]["status"] == "skipped"
    assert by_name["captions_generated"]["status"] == "skipped"

    frames = client.get(f"/api/videos/{identifiers['video_id']}/frames").get_json()["frames"]
    assert len(frames) == 1
    assert frames[0]["timestamp_ms"] == int(frames[0]["frame_number"] * 1000 / 25.0)
    assert client.get(frames[0]["image_url"]).status_code == 200

    found = client.post(f"/api/videos/{identifiers['video_id']}/query", json={"query": "find test class"}).get_json()
    assert found["insufficient_evidence"] is False
    assert found["frames"][0]["frame_id"] == frames[0]["frame_id"]
    assert "00:00:00.000" in found["answer"]
    assert found["citations"][0]["timestamp_ms"] == 0

    answer_only = client.post(
        f"/api/videos/{identifiers['video_id']}/query",
        json={"query": "find test class", "response_mode": "answer"},
    ).get_json()
    assert answer_only["frames"] == []
    assert answer_only["matches"] == []
    assert "00:00:00.000" in answer_only["answer"]

    absent = client.post(f"/api/videos/{identifiers['video_id']}/query", json={"query": "absent object"}).get_json()
    assert absent["insufficient_evidence"] is True
    assert absent["frames"] == []
    assert absent["answer"] == "Insufficient evidence found for this query."
