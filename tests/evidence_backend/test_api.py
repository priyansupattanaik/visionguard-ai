from fastapi.testclient import TestClient

from visionguard.evidence_api.api.app import create_app
from visionguard.evidence_api.api.dependencies import build_container
from visionguard.evidence_api.config import Settings


def test_health_and_grounded_abstention(tmp_path):
    container = build_container(Settings(data_dir=tmp_path))
    with TestClient(create_app(container)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["mode"] == "offline-first"
        answer = client.post("/v1/videos/missing/query", json={"query": "Was there a dog?"})
        assert answer.status_code == 200
        assert answer.json()["verified"] is False
        assert answer.json()["citations"] == []
