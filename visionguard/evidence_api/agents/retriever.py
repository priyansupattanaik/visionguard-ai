from visionguard.evidence_api.retrieval.hybrid import HybridRetriever
from visionguard.evidence_api.schemas.domain import CandidateEvidence, QueryPlan


class RetrievalAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def retrieve(self, video_id: str, plan: QueryPlan) -> list[CandidateEvidence]:
        return self.retriever.retrieve(video_id, plan)
