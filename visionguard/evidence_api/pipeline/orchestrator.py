from __future__ import annotations

from collections.abc import Callable

from visionguard.evidence_api.database.sqlite import SQLiteEvidenceRepository
from visionguard.evidence_api.pipeline.stages import ProcessingContext, ProcessingStage, default_stages
from visionguard.evidence_api.schemas.domain import Evidence, VideoAsset


ProgressCallback = Callable[[str, float], None]


class VideoProcessingOrchestrator:
    def __init__(self, repository: SQLiteEvidenceRepository, stages: list[ProcessingStage] | None = None):
        self.repository = repository
        self.stages = stages if stages is not None else default_stages()

    def process(self, video: VideoAsset, progress: ProgressCallback | None = None) -> list[Evidence]:
        cached = self.repository.list_evidence(video.id)
        if cached:
            if progress:
                progress("cached", 1.0)
            return cached
        context = ProcessingContext(video=video)
        total = max(1, len(self.stages))
        for index, stage in enumerate(self.stages):
            if progress:
                progress(stage.name, index / total)
            produced = list(stage.run(context))
            context.evidence.extend(produced)
            context.artifacts[stage.name] = produced
        self.repository.add_evidence(context.evidence)
        if progress:
            progress("completed", 1.0)
        return context.evidence
