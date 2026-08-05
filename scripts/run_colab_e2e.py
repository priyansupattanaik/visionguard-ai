"""Fast GPU end-to-end VisionGuard run for Google Colab (or any CUDA host).

What this does:
  1. Applies the Colab fast-GPU env profile
  2. Bootstraps YOLO nano into .models/
  3. Preloads SigLIP so indexing does not stall mid-run
  4. Runs upload → index → grounded present/absent queries

Anti-hallucination contract (honest limits):
  - Object answers come only from YOLO detector evidence
  - Absent-object query must return insufficient evidence
  - NVIDIA segment captions are stored as unverified semantic_description
  - No optional LLM reasoning (MODEL_PROVIDER=none)
  - Perfect zero error is impossible (detector false positives remain)

Usage:
  export NVIDIA_API_KEY=nvapi-...
  python scripts/run_colab_e2e.py
  python scripts/run_colab_e2e.py --video sample_videos/asset3.mp4 --query "find the person"
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def _apply_profile(nvidia_key: str, device: str) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "apply_colab_fast_env.py"),
        "--write-dotenv",
        "--device",
        device,
    ]
    if nvidia_key:
        cmd.extend(["--nvidia-key", nvidia_key])
    else:
        cmd.append("--from-env")
    _run(cmd)
    # Re-load into this process (subprocess wrote .env; also set env for current).
    from visionguard.runtime.env import load_project_env

    load_project_env(ROOT)


def _bootstrap_yolo(model: str) -> None:
    _run([sys.executable, str(ROOT / "scripts" / "bootstrap_models.py"), "--yolo", model])


def _preload_siglip(model_name: str) -> None:
    print(f"Preloading embedding model: {model_name}", flush=True)
    from transformers import AutoModel, AutoProcessor

    AutoProcessor.from_pretrained(model_name)
    AutoModel.from_pretrained(model_name)
    print("SigLIP ready", flush=True)


def _e2e(video: Path, query: str, absent_query: str, timeout: float) -> dict:
    os.environ["VISION_GUARD_SKIP_WARMUP"] = "1"
    os.environ["VERIFIER_READY_TIMEOUT"] = "0"

    from visionguard.web_app.server import create_app

    app = create_app(testing=True, start_warmup=False)
    client = app.test_client()

    t0 = time.perf_counter()
    upload = client.post(
        "/api/videos/upload",
        data={"video": (io.BytesIO(video.read_bytes()), video.name)},
        content_type="multipart/form-data",
    )
    if upload.status_code != 201:
        raise RuntimeError(f"Upload failed: {upload.status_code} {upload.get_data(as_text=True)}")
    ids = upload.get_json()

    index_response = client.post(f"/api/videos/{ids['video_id']}/index")
    if index_response.status_code != 202:
        raise RuntimeError(
            f"Index start failed: {index_response.status_code} {index_response.get_data(as_text=True)}"
        )

    status = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/videos/{ids['video_id']}/status").get_json()
        stages = ", ".join(f"{s['name']}={s['status']}" for s in status.get("stages", []))
        print(f"  status={status['status']} | {stages}", flush=True)
        if status["status"] in {"completed", "failed"}:
            break
        time.sleep(1.0)
    if status is None or status["status"] != "completed":
        raise RuntimeError(f"Processing did not complete: {status}")

    found = client.post(
        f"/api/videos/{ids['video_id']}/query",
        json={"query": query, "response_mode": "both"},
    ).get_json()
    absent = client.post(
        f"/api/videos/{ids['video_id']}/query",
        json={"query": absent_query, "response_mode": "both"},
    ).get_json()
    frames = client.get(f"/api/videos/{ids['video_id']}/frames").get_json()["frames"]
    elapsed = time.perf_counter() - t0

    checks = {
        "index_completed": status["status"] == "completed",
        "semantic_stage_completed": next(
            s for s in status["stages"] if s["name"] == "semantic_analysis"
        )["status"]
        == "completed",
        "present_query_has_evidence": bool(found.get("frames"))
        and found.get("insufficient_evidence") is False,
        "absent_query_abstains": absent.get("frames") == []
        and absent.get("insufficient_evidence") is True,
        "frame_count": len(frames),
    }
    report = {
        "video": str(video),
        "video_id": ids["video_id"],
        "elapsed_sec": round(elapsed, 2),
        "device": os.getenv("VISION_GUARD_DEVICE"),
        "yolo": os.getenv("YOLO_MODEL"),
        "win_sec": os.getenv("WIN_SEC"),
        "semantic_workers": os.getenv("SEMANTIC_WORKERS"),
        "checks": checks,
        "present_query": {
            "query": query,
            "insufficient_evidence": found.get("insufficient_evidence"),
            "frame_count": len(found.get("frames") or []),
            "answer": (found.get("answer") or found.get("message") or "")[:300],
            "claim_provenance_sample": [
                (m.get("claim_provenance") or m.get("evidence_state"))
                for m in (found.get("matches") or [])[:3]
            ],
        },
        "absent_query": {
            "query": absent_query,
            "insufficient_evidence": absent.get("insufficient_evidence"),
            "frame_count": len(absent.get("frames") or []),
            "answer": (absent.get("answer") or absent.get("message") or "")[:300],
        },
        "all_passed": all(checks.values()),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "sample_videos" / "asset3.mp4")
    parser.add_argument("--query", default="find the person")
    parser.add_argument("--absent-query", default="find the elephant")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--nvidia-key", default="", help="Optional; else use NVIDIA_API_KEY env")
    parser.add_argument("--skip-preload", action="store_true", help="Skip SigLIP preload")
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")

    os.chdir(ROOT)
    _apply_profile(args.nvidia_key, args.device)

    yolo = os.getenv("YOLO_MODEL", "yolo11n.pt")
    _bootstrap_yolo(yolo)

    if not args.skip_preload:
        _preload_siglip(os.getenv("CLIP_MODEL", "google/siglip2-so400m-patch14-384"))

    print("\n=== Running grounded end-to-end workflow ===", flush=True)
    report = _e2e(args.video, args.query, args.absent_query, args.timeout)
    print(json.dumps(report, indent=2))
    out = ROOT / "output" / "colab_e2e_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)

    if not report["all_passed"]:
        raise SystemExit(
            "E2E checks failed. Present query needs detector evidence; "
            "absent query must abstain (anti-hallucination)."
        )
    print("\nPASS: fast GPU E2E completed with grounded present/absent checks.", flush=True)


if __name__ == "__main__":
    main()
