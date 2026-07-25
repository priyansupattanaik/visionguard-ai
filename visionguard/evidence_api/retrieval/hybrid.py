from __future__ import annotations

import re

from visionguard.evidence_api.database.sqlite import SQLiteEvidenceRepository
from visionguard.evidence_api.schemas.domain import CandidateEvidence, Evidence, QueryPlan
from visionguard.evidence_api.query_language import load_query_vocabulary


class HybridRetriever:
    """Deterministic keyword/metadata/temporal fusion with extension points for vectors and graph."""

    def __init__(self, repository: SQLiteEvidenceRepository, *, lexical_weight: float = 0.55,
                 kind_weight: float = 0.20, confidence_weight: float = 0.25,
                 kind_mismatch_score: float = 0.25, pool_multiplier: int = 3,
                 min_score: float = 0.0):
        self.repository = repository
        self.lexical_weight = lexical_weight
        self.kind_weight = kind_weight
        self.confidence_weight = confidence_weight
        self.kind_mismatch_score = kind_mismatch_score
        self.pool_multiplier = max(1, pool_multiplier)
        self.min_score = min_score
        self.stop_words = set(load_query_vocabulary()["stop_words"])
        if abs((lexical_weight + kind_weight + confidence_weight) - 1.0) > 1e-6:
            raise ValueError("retrieval fusion weights must total 1.0")

    def _tokens(self, plan: QueryPlan) -> list[str]:
        values = plan.entities + plan.attributes + plan.events
        if not values:
            values = re.findall(r"[a-z0-9]+", plan.query.casefold())
        return sorted({value.casefold() for value in values if value.casefold() not in self.stop_words})

    def _score(self, evidence: Evidence, tokens: list[str], plan: QueryPlan) -> CandidateEvidence:
        corpus = f"{evidence.text} {evidence.attributes}".casefold()
        lexical = sum(token in corpus for token in tokens) / max(1, len(tokens))
        kind = 1.0 if not plan.required_kinds or evidence.kind in plan.required_kinds else self.kind_mismatch_score
        source_confidence = evidence.confidence
        score = min(1.0, self.lexical_weight * lexical + self.kind_weight * kind + self.confidence_weight * source_confidence)
        return CandidateEvidence(
            evidence=evidence, score=score,
            score_breakdown={"lexical": lexical, "kind": kind, "source": source_confidence},
        )

    def retrieve(self, video_id: str, plan: QueryPlan) -> list[CandidateEvidence]:
        tokens = self._tokens(plan)
        rows = self.repository.search_text(video_id, tokens, plan.retrieval_limit * self.pool_multiplier)
        candidates = [self._score(row, tokens, plan) for row in rows]
        candidates = [row for row in candidates if row.score >= self.min_score]
        candidates.sort(key=lambda row: (-row.score, row.evidence.start_seconds, row.evidence.id))
        return candidates[:plan.retrieval_limit]
