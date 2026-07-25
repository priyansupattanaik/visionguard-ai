from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from visionguard.evidence_api.schemas.domain import Evidence, VideoAsset


@dataclass(slots=True)
class ProcessingContext:
    video: VideoAsset
    artifacts: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)


class ProcessingStage(ABC):
    name: str

    @abstractmethod
    def run(self, context: ProcessingContext) -> Iterable[Evidence]:
        raise NotImplementedError


class AdapterStage(ProcessingStage):
    """Boundary for YOLO/SAM2/Whisper/etc. Missing adapters yield no facts."""

    def __init__(self, name: str, adapter: Any | None = None):
        self.name = name
        self.adapter = adapter

    def run(self, context: ProcessingContext) -> Iterable[Evidence]:
        if self.adapter is None:
            return []
        produced = self.adapter.analyze(context.video, context.artifacts)
        return [Evidence.model_validate(row) for row in produced]


def default_stages() -> list[ProcessingStage]:
    return [
        AdapterStage("preprocessing"), AdapterStage("detection"),
        AdapterStage("tracking"), AdapterStage("segmentation"),
        AdapterStage("audio"), AdapterStage("scene"), AdapterStage("events"),
        AdapterStage("ocr"), AdapterStage("embeddings"),
    ]
