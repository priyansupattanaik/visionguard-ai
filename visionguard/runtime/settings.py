"""Typed runtime settings for the local VisionGuard execution path."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _bounded_float(name: str, default: float, lower: float, upper: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(upper, max(lower, value))


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """Validated settings used by indexing and retrieval orchestration."""

    out_dir: str
    minimum_evidence_confidence: float
    track_detection_iou: float
    sample_sec: float
    win_sec: float
    enable_crop_embeddings: bool
    verifier_ready_timeout: float
    verifier_poll_interval: float
    max_exhaustive_verification_frames: int

    @classmethod
    def from_env(cls, out_dir: str = "output") -> "PipelineSettings":
        enabled = os.getenv("ENABLE_CROP_EMBEDDINGS", "0").strip().casefold()
        try:
            max_frames = int(os.getenv("MAX_EXHAUSTIVE_VERIFICATION_FRAMES", "24"))
        except ValueError:
            max_frames = 24
        return cls(
            out_dir=os.getenv("VISION_GUARD_OUT_DIR") or out_dir,
            minimum_evidence_confidence=_bounded_float("MIN_EVIDENCE_CONFIDENCE", 0.25, 0.0, 0.95),
            track_detection_iou=_bounded_float("TRACK_DETECTION_IOU", 0.5, 0.0, 1.0),
            sample_sec=_bounded_float("SAMPLE_SEC", 1.5, 0.05, 120.0),
            win_sec=_bounded_float("WIN_SEC", 4.5, 0.05, 600.0),
            enable_crop_embeddings=enabled not in {"0", "false", "no", "off"},
            verifier_ready_timeout=_bounded_float("VERIFIER_READY_TIMEOUT", 30.0, 0.0, 300.0),
            verifier_poll_interval=_bounded_float("VERIFIER_POLL_INTERVAL", 0.25, 0.05, 30.0),
            max_exhaustive_verification_frames=max(1, min(max_frames, 500)),
        )
