from __future__ import annotations

from dataclasses import dataclass

from visionguard.evidence_api.agents import (
    DeterministicQueryPlanner, EvidenceReasoner, GroundedResponder,
    RetrievalAgent, VerificationAgent,
)
from visionguard.evidence_api.agents.verifier import DeterministicEvidenceVerifier
from visionguard.evidence_api.config import Settings
from visionguard.evidence_api.database import SQLiteEvidenceRepository
from visionguard.evidence_api.pipeline import VideoIngestor, VideoProcessingOrchestrator
from visionguard.evidence_api.retrieval import HybridRetriever
from visionguard.evidence_api.services import QueryService
from visionguard.evidence_api.workers import InProcessJobQueue


@dataclass(slots=True)
class Container:
    settings: Settings
    repository: SQLiteEvidenceRepository
    ingestor: VideoIngestor
    orchestrator: VideoProcessingOrchestrator
    query_service: QueryService
    jobs: InProcessJobQueue


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or Settings.from_env()
    repository = SQLiteEvidenceRepository(settings.database_path)
    orchestrator = VideoProcessingOrchestrator(repository)
    retriever = RetrievalAgent(HybridRetriever(
        repository, lexical_weight=settings.lexical_weight,
        kind_weight=settings.kind_weight, confidence_weight=settings.confidence_weight,
        kind_mismatch_score=settings.kind_mismatch_score,
        pool_multiplier=settings.retrieval_pool_multiplier,
        min_score=settings.min_retrieval_score,
    ))
    query_service = QueryService(
        DeterministicQueryPlanner(retrieval_limit=settings.max_evidence), retriever,
        VerificationAgent(DeterministicEvidenceVerifier(settings.min_verification_confidence)),
        EvidenceReasoner(), GroundedResponder(),
    )
    return Container(
        settings=settings, repository=repository,
        ingestor=VideoIngestor(repository), orchestrator=orchestrator,
        query_service=query_service, jobs=InProcessJobQueue(orchestrator),
    )
