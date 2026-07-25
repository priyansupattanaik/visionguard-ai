from __future__ import annotations

from typing import Protocol

from visionguard.evidence_api.schemas.domain import CandidateEvidence, QueryPlan, VerifiedEvidence


class EvidenceVerifier(Protocol):
    def verify(self, plan: QueryPlan, candidate: CandidateEvidence) -> tuple[bool, float, str]: ...


class DeterministicEvidenceVerifier:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def verify(self, plan: QueryPlan, candidate: CandidateEvidence) -> tuple[bool, float, str]:
        confidence = min(candidate.score, candidate.evidence.confidence)
        accepted = confidence >= self.threshold
        rationale = "meets deterministic provenance and confidence thresholds" if accepted else "insufficient retrieval or source confidence"
        return accepted, confidence, rationale


class VerificationAgent:
    def __init__(self, verifier: EvidenceVerifier):
        self.verifier = verifier

    def verify(self, plan: QueryPlan, candidates: list[CandidateEvidence]) -> list[VerifiedEvidence]:
        results = []
        for candidate in candidates:
            accepted, confidence, rationale = self.verifier.verify(plan, candidate)
            results.append(VerifiedEvidence(
                evidence=candidate.evidence, retrieval_score=candidate.score,
                verification_confidence=confidence, accepted=accepted, rationale=rationale,
            ))
        return results
