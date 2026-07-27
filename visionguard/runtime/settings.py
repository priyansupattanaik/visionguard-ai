"""Typed runtime settings for the local VisionGuard execution path."""
from __future__ import annotations

import os
import json
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
    minimum_vector_similarity: float
    win_sec: float
    enable_crop_embeddings: bool
    verifier_ready_timeout: float
    verifier_poll_interval: float
    max_exhaustive_verification_frames: int
    semantic_provider: str
    semantic_zones: tuple[dict, ...]
    semantic_min_dwell_seconds: float

    @classmethod
    def from_env(cls, out_dir: str = "output") -> "PipelineSettings":
        enabled = os.getenv("ENABLE_CROP_EMBEDDINGS", "0").strip().casefold()
        try:
            max_frames = int(os.getenv("MAX_EXHAUSTIVE_VERIFICATION_FRAMES", "24"))
        except ValueError:
            max_frames = 24
        semantic_provider = os.getenv("SEMANTIC_PROVIDER", "nvidia").strip().casefold()
        if semantic_provider != "nvidia":
            raise ValueError("SEMANTIC_PROVIDER must be 'nvidia'; no semantic fallback provider is supported.")
        zones_raw = os.getenv("VISION_GUARD_ZONES_JSON", "[]")
        try:
            zones = json.loads(zones_raw)
        except json.JSONDecodeError as exc:
            raise ValueError("VISION_GUARD_ZONES_JSON must be a JSON array of normalized zones.") from exc
        if not isinstance(zones, list) or not all(isinstance(row, dict) for row in zones):
            raise ValueError("VISION_GUARD_ZONES_JSON must be a JSON array of zone objects.")
        return cls(
            out_dir=os.getenv("VISION_GUARD_OUT_DIR") or out_dir,
            minimum_evidence_confidence=_bounded_float("MIN_EVIDENCE_CONFIDENCE", 0.25, 0.0, 0.95),
            minimum_vector_similarity=_bounded_float("MIN_VECTOR_SIMILARITY", 0.14, -1.0, 1.0),
            win_sec=_bounded_float("WIN_SEC", 4.5, 0.05, 600.0),
            enable_crop_embeddings=enabled not in {"0", "false", "no", "off"},
            verifier_ready_timeout=_bounded_float("VERIFIER_READY_TIMEOUT", 30.0, 0.0, 300.0),
            verifier_poll_interval=_bounded_float("VERIFIER_POLL_INTERVAL", 0.25, 0.05, 30.0),
            max_exhaustive_verification_frames=max(1, min(max_frames, 500)),
            semantic_provider=semantic_provider,
            semantic_zones=tuple(zones),
            semantic_min_dwell_seconds=_bounded_float("SEMANTIC_MIN_DWELL_SECONDS", 2.0, 0.0, 3600.0),
        )
