from __future__ import annotations

from typing import Any, Protocol

from visionguard.evidence_api.schemas.domain import VideoAsset


class ModelAdapter(Protocol):
    name: str

    def analyze(self, video: VideoAsset, artifacts: dict[str, Any]) -> list[dict]: ...


class UnavailableModelAdapter:
    """Explicit offline fallback: report no facts instead of synthesizing them."""

    def __init__(self, name: str, reason: str):
        self.name = name
        self.reason = reason

    def analyze(self, video: VideoAsset, artifacts: dict[str, Any]) -> list[dict]:
        return []
