"""Detector observations converted into traceable temporal evidence segments."""
from __future__ import annotations

from collections.abc import Callable


class DetectorEvidenceRetriever:
    """Rank indexed detector evidence without owning models or filesystem state."""

    def __init__(self, matching_detections: Callable, clip_bounds: Callable[[float], tuple[float, float]], minimum_confidence: float) -> None:
        self._matching_detections = matching_detections
        self._clip_bounds = clip_bounds
        self.minimum_confidence = minimum_confidence

    def retrieve(self, index: dict, query: str, qobjs: set[str], qcolors: set[str], class_ids, cls_to_name: dict, top_k: int) -> list[dict]:
        if not class_ids:
            return []
        observations = []
        for frame in index.get("frames", []):
            matched = self._matching_detections(frame, qobjs, qcolors, cls_to_name)
            if not matched:
                continue
            confidence = max(float(item.get("conf", 0.0)) for item in matched)
            if confidence < self.minimum_confidence:
                continue
            calibrated = (confidence - self.minimum_confidence) / max(1e-6, 1.0 - self.minimum_confidence)
            observations.append({
                "score": 0.50 + 0.40 * calibrated + 0.04 * min(3, len(matched) - 1),
                "frame_id": frame.get("frame_id"), "ts": frame["ts"], "frame_path": frame["frame_path"],
                "appearances": frame.get("appearances", []), "tracks": frame.get("tracks", []),
                "matched_detections": matched,
            })
        if not observations:
            return []
        observations.sort(key=lambda row: row["ts"])
        meta = index.get("meta", {})
        gap_sec = max(float(meta.get("win_sec", 0.0)), float(meta.get("frame_interval_sec", 1.0)) * 1.25, 1.0)
        groups = [[observations[0]]]
        for observation in observations[1:]:
            if observation["ts"] - groups[-1][-1]["ts"] <= gap_sec:
                groups[-1].append(observation)
            else:
                groups.append([observation])
        return sorted((self._segment(query, group) for group in groups), key=lambda row: row["score"], reverse=True)[:top_k]

    def _segment(self, query: str, group: list[dict]) -> dict:
        peak = max(group, key=lambda row: row["score"])
        labels = sorted({item["name"] for row in group for item in row["matched_detections"]})
        tracks = sorted({track for row in group for track in row["tracks"]})
        start, _ = self._clip_bounds(group[0]["ts"])
        _, end = self._clip_bounds(group[-1]["ts"])
        return {
            "query": query, "score": peak["score"] + min(0.08, 0.02 * (len(group) - 1)),
            "base_score": peak["score"], "retrieval_mode": "detector",
            "cache_key": f"detector-segment:{group[0].get('frame_id', group[0]['ts'])}",
            "start": start, "end": end, "peak_ts": peak["ts"], "frame_path": peak["frame_path"],
            "objects": labels, "tracks": tracks,
            "appearances": sorted({tag for row in group for tag in row["appearances"]}),
            "matched_detections": peak["matched_detections"], "tags": [],
            "evidence_state": "detector_fact", "claim_provenance": "yolo_botsort",
            "summary": f"detector-evidence segment {start:.2f}s-{end:.2f}s | detected: {', '.join(labels)}",
        }
