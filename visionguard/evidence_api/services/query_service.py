from visionguard.evidence_api.agents.planner import DeterministicQueryPlanner
from visionguard.evidence_api.agents.reasoner import EvidenceReasoner
from visionguard.evidence_api.agents.responder import GroundedResponder
from visionguard.evidence_api.agents.retriever import RetrievalAgent
from visionguard.evidence_api.agents.verifier import VerificationAgent
from visionguard.evidence_api.schemas.domain import GroundedAnswer


class QueryService:
    def __init__(self, planner: DeterministicQueryPlanner, retriever: RetrievalAgent,
                 verifier: VerificationAgent, reasoner: EvidenceReasoner,
                 responder: GroundedResponder):
        self.planner = planner
        self.retriever = retriever
        self.verifier = verifier
        self.reasoner = reasoner
        self.responder = responder

    def answer(self, video_id: str, query: str) -> GroundedAnswer:
        plan = self.planner.plan(query)
        candidates = self.retriever.retrieve(video_id, plan)
        verified = self.verifier.verify(plan, candidates)
        reasoning = self.reasoner.reason(plan, verified)
        return self.responder.respond(plan, reasoning)
