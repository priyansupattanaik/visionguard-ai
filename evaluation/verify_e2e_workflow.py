"""Run a real local upload-to-query verification through the Flask API."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("sample_videos/asset3.mp4"))
    parser.add_argument("--query", default="find the person")
    parser.add_argument("--absent-query", default="find the elephant")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"Video does not exist: {args.video}")

    os.environ["VISION_GUARD_SKIP_WARMUP"] = "1"
    os.environ["VERIFIER_READY_TIMEOUT"] = "0"
    os.environ["NVIDIA_API_KEY"] = " "  # Keep this local verification independent of hosted APIs.

    from visionguard.web_app.server import create_app

    app = create_app(testing=True, start_warmup=False)
    client = app.test_client()
    upload = client.post(
        "/api/videos/upload",
        data={"video": (io.BytesIO(args.video.read_bytes()), "cctv-real-upload.mp4")},
        content_type="multipart/form-data",
    )
    if upload.status_code != 202:
        raise RuntimeError(f"Upload failed: {upload.status_code} {upload.get_data(as_text=True)}")
    identifiers = upload.get_json()

    snapshots = []
    deadline = time.time() + args.timeout
    status = None
    while time.time() < deadline:
        status_response = client.get(f"/api/videos/{identifiers['video_id']}/status")
        status = status_response.get_json()
        snapshots.append({
            "status": status["status"],
            "running": [stage["name"] for stage in status["stages"] if stage["status"] == "running"],
            "terminal": sum(stage["status"] in {"completed", "skipped", "failed"} for stage in status["stages"]),
        })
        if status["status"] in {"completed", "failed"}:
            break
        time.sleep(0.2)
    if status is None or status["status"] != "completed":
        raise RuntimeError(f"Processing did not complete: {status}")

    video = client.get(f"/api/videos/{identifiers['video_id']}").get_json()
    frame_payload = client.get(f"/api/videos/{identifiers['video_id']}/frames").get_json()
    frames = frame_payload["frames"]
    if not frames:
        raise RuntimeError("No evidence frames were exposed by the API.")
    image_responses = [client.get(frame["image_url"]) for frame in frames]
    found = client.post(
        f"/api/videos/{identifiers['video_id']}/query",
        json={"query": args.query},
    ).get_json()
    absent = client.post(
        f"/api/videos/{identifiers['video_id']}/query",
        json={"query": args.absent_query},
    ).get_json()
    events = client.get(f"/api/jobs/{identifiers['job_id']}/events").get_json()["events"]
    uploaded = sorted(Path("output/uploads").glob("*cctv-real-upload.mp4"), key=lambda path: path.stat().st_mtime)
    stored = uploaded[-1]

    result = {
        "upload_http": upload.status_code,
        "video_id": identifiers["video_id"],
        "job_id": identifiers["job_id"],
        "filename": identifiers["filename"],
        "source_bytes": args.video.stat().st_size,
        "stored_bytes": stored.stat().st_size,
        "metadata": {key: video[key] for key in ("duration_ms", "fps", "width", "height", "frame_count", "codec", "file_size")},
        "chunks": video["chunks"],
        "status": status,
        "status_snapshots": snapshots[:3] + snapshots[-3:],
        "event_count": len(events),
        "frame_count_indexed": len(frames),
        "timestamp_formula_valid": all(
            frame["timestamp_ms"] == int(frame["frame_number"] * 1000 / video["fps"])
            for frame in frames
        ),
        "all_image_urls_ok": all(response.status_code == 200 and response.data for response in image_responses),
        "first_frame": frames[0],
        "query": found,
        "absent_query": absent,
    }
    checks = {
        "stored_upload_is_exact": result["source_bytes"] == result["stored_bytes"],
        "query_has_evidence": bool(found.get("frames")) and found.get("insufficient_evidence") is False,
        "absent_query_is_insufficient": absent.get("frames") == [] and absent.get("insufficient_evidence") is True,
        "all_stages_terminal": all(stage["status"] in {"completed", "skipped"} for stage in status["stages"]),
        "timestamp_formula_valid": result["timestamp_formula_valid"],
        "all_image_urls_ok": result["all_image_urls_ok"],
    }
    result["checks"] = checks
    if not all(checks.values()):
        raise RuntimeError(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
