from visionguard.evidence_api.database import SQLiteEvidenceRepository
from visionguard.evidence_api.pipeline.orchestrator import VideoProcessingOrchestrator
from visionguard.evidence_api.pipeline.stages import ProcessingContext, ProcessingStage
from visionguard.evidence_api.schemas.domain import Evidence, EvidenceKind, VideoAsset

class CountingStage(ProcessingStage):
    name="counting"
    def __init__(self): self.calls=0
    def run(self, context: ProcessingContext):
        self.calls += 1
        return [Evidence(id="ev_once", video_id=context.video.id, kind=EvidenceKind.SCENE, start_seconds=0, end_seconds=1, text="road scene", confidence=.9, source="fixture")]

def test_processing_is_cached_by_persisted_evidence(tmp_path):
    repository=SQLiteEvidenceRepository(tmp_path/"cache.sqlite3")
    video=VideoAsset(id="video_once", source_path="x.mp4", sha256="b"*64, filename="x.mp4", duration_seconds=1, fps=25, frame_count=25, width=32, height=32)
    repository.save_video(video); stage=CountingStage(); pipeline=VideoProcessingOrchestrator(repository,[stage])
    assert len(pipeline.process(video)) == len(pipeline.process(video)) == 1
    assert stage.calls == 1
