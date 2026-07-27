import io
import json
from urllib.error import HTTPError, URLError

import pytest
import cv2
import numpy as np

from visionguard.model_services.model_provider import (
    LlamaCppProvider,
    ModelProviderError,
    NoneModelProvider,
    OpenAICompatibleProvider,
    create_model_provider,
    model_health_snapshot,
)
from visionguard.model_services.nvidia_verifier import NvidiaFrameVerifier
from visionguard.semantic.nvidia_semantic import NvidiaSemanticAnalyzer


class FakeResponse:
    def __init__(self, payload):
        self.stream = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self.stream

    def __exit__(self, exc_type, exc, traceback):
        self.stream.close()


def test_provider_selection_defaults_to_local_no_provider(monkeypatch):
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    assert isinstance(create_model_provider(), NoneModelProvider)


def test_provider_selection_supports_none(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "none")
    assert isinstance(create_model_provider(), NoneModelProvider)


def test_llama_cpp_successful_chat_response():
    def opener(request, timeout):
        assert request.full_url == "http://127.0.0.1:8080/v1/chat/completions"
        return FakeResponse({"choices": [{"message": {"content": "local answer"}}]})

    provider = LlamaCppProvider(text_url="http://127.0.0.1:8080", opener=opener)
    assert provider.chat([{"role": "user", "content": "hello"}]) == "local answer"


def test_openai_compatible_base_url_does_not_duplicate_v1():
    def opener(request, timeout):
        assert request.full_url == "https://provider.example/v1/chat/completions"
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        "nvidia",
        "https://provider.example/v1",
        "model",
        api_key="configured-in-test-only",
        opener=opener,
    )

    assert provider.chat([{"role": "user", "content": "hello"}]) == "ok"


def test_llama_cpp_json_mode_retries_without_response_format():
    calls = []

    def opener(request, timeout):
        payload = json.loads(request.data)
        calls.append(payload)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 400, "unsupported response_format", {}, None)
        return FakeResponse({"choices": [{"message": {"content": '{"intent":"object_search"}'}}]})

    provider = LlamaCppProvider(text_url="http://127.0.0.1:8080", opener=opener)
    result = json.loads(provider.chat([{"role": "user", "content": "find a person"}], json_mode=True))

    assert result == {"intent": "object_search"}
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_llama_cpp_unreachable_is_controlled():
    def opener(request, timeout):
        raise URLError("connection refused")

    provider = LlamaCppProvider(text_url="http://127.0.0.1:8080", opener=opener)
    health = provider.health()

    assert health["configured"] is True
    assert health["reachable"] is False
    with pytest.raises(ModelProviderError, match="request failed"):
        provider.chat([{"role": "user", "content": "hello"}])


def test_missing_nvidia_key_is_irrelevant_in_llama_mode(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "llama_cpp")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    provider = create_model_provider()

    assert provider.provider_name == "llama_cpp"
    assert provider.configured is True


def test_nvidia_verifier_warms_when_first_used(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "configured-in-test-only")
    monkeypatch.setattr(
        "visionguard.model_services.nvidia_verifier.OpenAICompatibleProvider.health",
        lambda _self: {"configured": True, "reachable": True, "message": "reachable"},
    )
    verifier = NvidiaFrameVerifier()

    assert verifier.backend == "unconfigured"
    assert verifier.is_ready() is True
    assert verifier.verification_mode() == "nvidia_api"


def test_nvidia_health_reports_the_shared_multimodal_endpoint():
    class ReachableNvidiaProvider:
        provider_name = "nvidia"

        @staticmethod
        def health():
            return {
                "configured": True,
                "reachable": True,
                "url": "https://integrate.api.nvidia.com/v1",
                "message": "nvidia endpoint is reachable.",
            }

    health = model_health_snapshot(ReachableNvidiaProvider())

    assert health["text_model"]["reachable"] is True
    assert health["vision_model"]["reachable"] is True
    assert health["vision_model"]["message"] == "NVIDIA multimodal endpoint is reachable."


def test_semantic_health_requires_an_authenticated_live_response(monkeypatch):
    requests = []

    def opener(request, timeout):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer configured-in-test-only"
        return FakeResponse({"data": [{"id": "model"}]})

    monkeypatch.setattr("visionguard.semantic.nvidia_semantic.urlopen", opener)
    analyzer = NvidiaSemanticAnalyzer(
        "configured-in-test-only", "https://provider.example/v1", "model", timeout=2
    )

    health = analyzer.health()

    assert health["ready"] is True
    assert requests[0].full_url == "https://provider.example/v1/models"


def test_verifier_cache_is_scoped_to_image_content(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "configured-in-test-only")
    verifier = NvidiaFrameVerifier()
    verifier._ready = True
    verifier.backend = "nvidia_api"
    calls = []
    verifier._ask = lambda frame_path, _prompt: calls.append(frame_path) or json.dumps({
        "matched": True, "confidence": 0.9, "description": "visible", "boxes": [[0, 0, 5, 5]],
    })
    first = np.zeros((10, 10, 3), dtype=np.uint8)
    second = np.full((10, 10, 3), 255, dtype=np.uint8)
    first_path, second_path = tmp_path / "first.jpg", tmp_path / "second.jpg"
    assert cv2.imwrite(str(first_path), first)
    assert cv2.imwrite(str(second_path), second)

    verifier.verify_query(str(first_path), "person", frame_key="video:frame:0")
    verifier.verify_query(str(second_path), "person", frame_key="video:frame:0")

    assert len(calls) == 2
