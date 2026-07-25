from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import RLock

from visionguard.evidence_api.pipeline.orchestrator import VideoProcessingOrchestrator
from visionguard.evidence_api.schemas.domain import JobStatus, ProcessingJob, VideoAsset


class InProcessJobQueue:
    """Development queue. The interface can be replaced by Redis/RQ without changing APIs."""

    def __init__(self, orchestrator: VideoProcessingOrchestrator, workers: int = 1):
        self.orchestrator = orchestrator
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="visionguard")
        self.jobs: dict[str, ProcessingJob] = {}
        self._lock = RLock()

    def submit(self, video: VideoAsset) -> ProcessingJob:
        job = ProcessingJob(video_id=video.id)
        with self._lock:
            self.jobs[job.id] = job
        self.executor.submit(self._run, job.id, video)
        return job

    def _run(self, job_id: str, video: VideoAsset) -> None:
        self._update(job_id, status=JobStatus.RUNNING, stage="starting", progress=0)
        try:
            self.orchestrator.process(video, lambda stage, value: self._update(job_id, stage=stage, progress=value))
        except Exception as exc:
            self._update(job_id, status=JobStatus.FAILED, stage="failed", error=str(exc))
        else:
            self._update(job_id, status=JobStatus.COMPLETED, stage="completed", progress=1)

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            self.jobs[job_id] = self.jobs[job_id].model_copy(update=changes)

    def get(self, job_id: str) -> ProcessingJob | None:
        with self._lock:
            return self.jobs.get(job_id)
