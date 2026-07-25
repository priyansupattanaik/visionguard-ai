from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class EvidenceKind(str, Enum):
    FRAME = "frame"
    OBJECT = "object"
    TRACK = "track"
    EVENT = "event"
    OCR = "ocr"
    SPEECH = "speech"
    SOUND = "sound"
    SCENE = "scene"


class VideoAsset(BaseModel):
    id: str = Field(default_factory=lambda: new_id("video"))
    source_path: str
    sha256: str
    filename: str
    duration_seconds: float = Field(ge=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    codec: str | None = None
    has_audio: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def valid_extent(self) -> "BoundingBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must have positive area")
        return self


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ev"))
    video_id: str
    kind: EvidenceKind
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    frame_id: int | None = Field(default=None, ge=0)
    track_ids: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    text: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    source: str

    @model_validator(mode="after")
    def valid_interval(self) -> "Evidence":
        if self.end_seconds < self.start_seconds:
            raise ValueError("evidence end must not precede start")
        return self


class TemporalRelation(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    BETWEEN = "between"
    NONE = "none"


class QueryPlan(BaseModel):
    query: str = Field(min_length=1)
    normalized_query: str = ""
    intent: str = "visual_search"
    entities: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    unknown_terms: list[str] = Field(default_factory=list)
    retrieval_routes: list[str] = Field(default_factory=list)
    clarification: str | None = None
    limitations: list[str] = Field(default_factory=list)
    temporal_relation: TemporalRelation = TemporalRelation.NONE
    reference_event: str | None = None
    requires_count: bool = False
    required_kinds: list[EvidenceKind] = Field(default_factory=list)
    retrieval_limit: int = Field(default=20, ge=1, le=100)


class CandidateEvidence(BaseModel):
    evidence: Evidence
    score: float = Field(ge=0, le=1)
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class VerifiedEvidence(BaseModel):
    evidence: Evidence
    retrieval_score: float = Field(ge=0, le=1)
    verification_confidence: float = Field(ge=0, le=1)
    accepted: bool
    rationale: str


class Citation(BaseModel):
    evidence_id: str
    timestamp_start: float
    timestamp_end: float
    frame_id: int | None = None
    track_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)
    citations: list[Citation] = Field(default_factory=list)
    uncertainty: str | None = None
    reasoning_summary: str = ""
    verified: bool = False

    @model_validator(mode="after")
    def grounded_claims_only(self) -> "GroundedAnswer":
        if self.verified and not self.citations:
            raise ValueError("a verified answer requires evidence citations")
        return self


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingJob(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    video_id: str
    status: JobStatus = JobStatus.QUEUED
    stage: str = "queued"
    progress: float = Field(default=0, ge=0, le=1)
    error: str | None = None
