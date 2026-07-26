"""Truthful video, job, chunk, and frame records for the local web application."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import cv2


STAGE_LABELS = (
    ("upload_received", "Upload received"),
    ("metadata_extracted", "Metadata extracted"),
    ("video_normalized", "Video normalized"),
    ("chunks_created", "Chunks created"),
    ("frames_extracted", "Frames extracted"),
    ("keyframes_selected", "Keyframes selected"),
    ("duplicates_removed", "Duplicates removed"),
    ("objects_detected", "Objects detected"),
    ("frame_metadata_collected", "Frame metadata collected"),
    ("ocr_completed", "OCR completed"),
    ("captions_generated", "Captions generated"),
    ("embeddings_generated", "Embeddings generated"),
    ("vector_index_updated", "Vector index updated"),
    ("query_ready", "Query ready"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def probe_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("The uploaded file could not be opened as a video.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((codec_value >> (8 * offset)) & 0xFF) for offset in range(4)).strip("\x00 ")
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("The uploaded video has invalid or unreadable metadata.")
    duration_ms = int(round(frame_count * 1000.0 / fps))
    return {
        "duration_ms": duration_ms,
        # Keep the authoritative decoder FPS value. Rounding it here can shift
        # long-video frame timestamps by a millisecond or more.
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "codec": codec or None,
        "file_size": path.stat().st_size,
    }


def make_chunks(video_id: str, metadata: dict, chunk_seconds: float = 30.0, overlap_seconds: float = 5.0) -> list[dict]:
    duration_ms = int(metadata["duration_ms"])
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    chunk_ms = max(1, int(round(chunk_seconds * 1000)))
    step_ms = max(1, int(round((chunk_seconds - overlap_seconds) * 1000)))
    chunks = []
    start_ms = 0
    while start_ms < duration_ms:
        end_ms = min(duration_ms, start_ms + chunk_ms)
        start_frame = min(frame_count - 1, int(start_ms * fps / 1000.0))
        end_frame = min(frame_count - 1, max(start_frame, int(math.ceil(end_ms * fps / 1000.0)) - 1))
        chunks.append({
            "chunk_id": f"{video_id}_chunk_{len(chunks):04d}",
            "video_id": video_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
        if end_ms >= duration_ms:
            break
        start_ms += step_ms
    return chunks


def make_job(job_id: str, video_id: str) -> dict:
    now = utc_now()
    stages = []
    for name, label in STAGE_LABELS:
        stages.append({
            "name": name,
            "label": label,
            "status": "waiting",
            "processed": 0,
            "total": 0,
            "started_at": None,
            "updated_at": now,
            "message": "",
        })
    return {
        "job_id": job_id,
        "video_id": video_id,
        "status": "waiting",
        "stages": stages,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "events": [],
    }


def update_stage(job: dict, name: str, status: str, *, processed: int | None = None, total: int | None = None, message: str | None = None) -> None:
    if status not in {"waiting", "running", "completed", "failed", "skipped"}:
        raise ValueError(f"Unsupported stage status: {status}")
    stage = next(row for row in job["stages"] if row["name"] == name)
    now = utc_now()
    if status == "running" and stage["started_at"] is None:
        stage["started_at"] = now
    if status in {"completed", "failed", "skipped"} and stage["started_at"] is None:
        stage["started_at"] = now
    stage["status"] = status
    if processed is not None:
        stage["processed"] = int(processed)
    if total is not None:
        stage["total"] = int(total)
    if message is not None:
        stage["message"] = message
    stage["updated_at"] = now
    job["updated_at"] = now
    job["events"].append({
        "event_id": len(job["events"]) + 1,
        "stage": name,
        "status": status,
        "processed": stage["processed"],
        "total": stage["total"],
        "message": stage["message"],
        "timestamp": now,
    })


def public_job(job: dict) -> dict:
    return {
        "job_id": job["job_id"],
        "video_id": job["video_id"],
        "status": job["status"],
        "stages": [dict(stage) for stage in job["stages"]],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "error": job["error"],
    }


def materialize_frames(video_id: str, pipeline) -> list[dict]:
    index = pipeline.idx or {}
    metadata = index.get("meta", {})
    fps = float(metadata.get("fps", 0))
    if fps <= 0:
        return []
    frames = []
    for row in index.get("frames", []):
        image_path = Path(row.get("frame_path", "")).resolve()
        if not image_path.is_file():
            continue
        frame_number = int(row["frame"])
        timestamp_ms = int(frame_number * 1000.0 / fps)
        frame_id = f"{video_id}_f{frame_number:09d}"
        detections = []
        for detection in row.get("detections", []):
            detections.append({
                "frame_id": frame_id,
                "class_name": str(detection.get("name", "")),
                "confidence": round(float(detection.get("conf", 0.0)), 6),
                "bbox": [round(float(value), 3) for value in detection.get("box", [])],
                "track_id": detection.get("track_id"),
                "color": detection.get("color"),
            })
        frames.append({
            "frame_id": frame_id,
            "video_id": video_id,
            "frame_number": frame_number,
            "timestamp_ms": timestamp_ms,
            "image_url": f"/api/videos/{video_id}/frames/{frame_id}/image",
            "objects": list(row.get("objects", [])),
            "tracks": list(row.get("tracks", [])),
            "detections": detections,
            "motion_score": float(row.get("motion_score", 0.0)),
            "selection_reason": row.get("keep_reason", ""),
            "_image_path": str(image_path),
        })
    return frames


def frame_from_progress_event(video_id: str, event: dict) -> dict | None:
    image_path = Path(event.get("frame_path", "")).resolve()
    if not image_path.is_file():
        return None
    frame_number = int(event["frame_number"])
    frame_id = f"{video_id}_f{frame_number:09d}"
    detections = []
    for detection in event.get("detection_rows", []):
        detections.append({
            "frame_id": frame_id,
            "class_name": str(detection.get("name", "")),
            "confidence": round(float(detection.get("conf", 0.0)), 6),
            "bbox": [round(float(value), 3) for value in detection.get("box", [])],
            "track_id": detection.get("track_id"),
            "color": detection.get("color"),
        })
    return {
        "frame_id": frame_id,
        "video_id": video_id,
        "frame_number": frame_number,
        "timestamp_ms": int(event["timestamp_ms"]),
        "image_url": f"/api/videos/{video_id}/frames/{frame_id}/image",
        "objects": list(event.get("objects", [])),
        "tracks": list(event.get("tracks", [])),
        "detections": detections,
        "motion_score": float(event.get("motion_score", 0.0)),
        "selection_reason": event.get("selection_reason", ""),
        "_image_path": str(image_path),
    }


def public_frame(frame: dict) -> dict:
    return {key: value for key, value in frame.items() if not key.startswith("_")}
