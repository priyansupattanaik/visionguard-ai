"""Evaluate bundled sample videos without presenting confidence as accuracy."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visionguard.model_services.tracker import ObjectTracker
from visionguard.runtime.env import load_project_env


def inspect_video(path: Path, detector: ObjectTracker, sample_seconds: float) -> dict:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(fps * sample_seconds))
    sampled = detected_frames = 0
    class_counts: Counter[str] = Counter()
    confidences: list[float] = []
    started = time.perf_counter()
    for frame_index in range(0, frame_count, stride):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        sampled += 1
        detections = detector.detect(frame)
        if detections:
            detected_frames += 1
        for detection in detections:
            class_counts[detection["name"]] += 1
            confidences.append(float(detection["conf"]))
    capture.release()
    return {
        "file": path.name,
        "readable": fps > 0 and frame_count > 0,
        "duration_seconds": round(frame_count / fps, 3) if fps else 0,
        "resolution": f"{width}x{height}",
        "sampled_frames": sampled,
        "frames_with_detections": detected_frames,
        "detection_coverage": round(detected_frames / sampled, 4) if sampled else 0,
        "average_detection_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "detections_by_class": dict(class_counts.most_common()),
        "processing_seconds": round(time.perf_counter() - started, 3),
    }


def labelled_accuracy(report: list[dict], labels: dict) -> dict | None:
    expected = labels.get("expected_objects_by_video", {})
    if not expected:
        return None
    true_positive = false_positive = false_negative = 0
    for row in report:
        truth = set(expected.get(row["file"], []))
        predicted = set(row["detections_by_class"])
        true_positive += len(truth & predicted)
        false_positive += len(predicted - truth)
        false_negative += len(truth - predicted)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument("--labels", type=Path, default=ROOT / "evaluation" / "ground_truth.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "latest_asset_report.json")
    args = parser.parse_args()
    if args.sample_seconds <= 0:
        raise SystemExit("--sample-seconds must be positive")
    load_project_env(ROOT)
    detector = ObjectTracker()
    videos = sorted((ROOT / "sample_videos").glob("*.mp4"))
    rows = [inspect_video(path, detector, args.sample_seconds) for path in videos]
    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels.exists() else {}
    result = {
        "sample_interval_seconds": args.sample_seconds,
        "videos": rows,
        "accuracy": labelled_accuracy(rows, labels),
        "accuracy_note": "Accuracy requires human ground-truth labels; detector confidence is not accuracy.",
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
