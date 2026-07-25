from pathlib import Path
import pytest
from visionguard.evidence_api.agents import DeterministicQueryPlanner, EvidenceReasoner, GroundedResponder, RetrievalAgent, VerificationAgent
from visionguard.evidence_api.agents.verifier import DeterministicEvidenceVerifier
from visionguard.evidence_api.database import SQLiteEvidenceRepository
from visionguard.evidence_api.retrieval import HybridRetriever
from visionguard.evidence_api.schemas.domain import Evidence, EvidenceKind, GroundedAnswer, VideoAsset
from visionguard.evidence_api.services import QueryService

def video():
    return VideoAsset(id="video_test", source_path="test.mp4", sha256="a"*64, filename="test.mp4", duration_seconds=10, fps=25, frame_count=250, width=640, height=480)

def service(tmp_path: Path, evidence=None):
    repository = SQLiteEvidenceRepository(tmp_path / "test.sqlite3")
    repository.save_video(video()); repository.add_evidence(evidence or [])
    return QueryService(DeterministicQueryPlanner(), RetrievalAgent(HybridRetriever(repository)), VerificationAgent(DeterministicEvidenceVerifier(0.65)), EvidenceReasoner(), GroundedResponder())

def test_no_evidence_refuses_to_answer(tmp_path):
    answer = service(tmp_path).answer("video_test", "Was anyone carrying a backpack?")
    assert not answer.verified and answer.citations == []
    assert answer.answer == "I cannot verify this from the available evidence."

def test_low_confidence_candidate_is_rejected(tmp_path):
    row = Evidence(id="ev_low", video_id="video_test", kind=EvidenceKind.OBJECT, start_seconds=2, end_seconds=2.1, frame_id=50, text="person carrying backpack", confidence=0.4, source="detector", track_ids=["track_1"])
    assert not service(tmp_path, [row]).answer("video_test", "Was a person carrying a backpack?").verified

def test_verified_answer_cites_exact_evidence(tmp_path):
    row = Evidence(id="ev_supported", video_id="video_test", kind=EvidenceKind.EVENT, start_seconds=2, end_seconds=3, frame_id=50, text="person entered restricted zone", confidence=0.98, source="event_rule_v1", track_ids=["track_1"], event_ids=["event_1"])
    answer = service(tmp_path, [row]).answer("video_test", "Which person entered?")
    assert answer.verified and answer.citations[0].evidence_id == "ev_supported"

def test_schema_blocks_verified_answer_without_citations():
    with pytest.raises(ValueError): GroundedAnswer(answer="unsupported", confidence=.9, verified=True)
