from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from visionguard.evidence_api import __version__
from visionguard.evidence_api.api.dependencies import Container, build_container
from visionguard.evidence_api.pipeline.ingestion import InvalidVideoError
from visionguard.evidence_api.schemas.domain import GroundedAnswer, ProcessingJob, VideoAsset


class IngestRequest(BaseModel):
    path: str = Field(min_length=1)


class IngestResponse(BaseModel):
    video: VideoAsset
    cached: bool
    job: ProcessingJob | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


def create_app(container: Container | None = None) -> FastAPI:
    services = container or build_container()
    app = FastAPI(title="VisionGuard AI", version=__version__)
    app.state.container = services

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__, "mode": "offline-first"}

    @app.get("/", include_in_schema=False)
    async def evidence_console() -> FileResponse:
        return FileResponse(Path(__file__).resolve().parents[3] / "web_interface" / "evidence_console.html")

    @app.post("/v1/videos", response_model=IngestResponse)
    async def ingest(request: IngestRequest) -> IngestResponse:
        try:
            video, cached = services.ingestor.ingest(Path(request.path))
        except InvalidVideoError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        job = None if cached else services.jobs.submit(video)
        return IngestResponse(video=video, cached=cached, job=job)

    @app.get("/v1/jobs/{job_id}", response_model=ProcessingJob)
    async def job_status(job_id: str) -> ProcessingJob:
        job = services.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.post("/v1/videos/{video_id}/query", response_model=GroundedAnswer)
    async def query(video_id: str, request: QueryRequest) -> GroundedAnswer:
        return services.query_service.answer(video_id, request.query)

    return app


app = create_app()
