"""Small, failure-safe chat-provider abstraction for local and optional hosted models."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelProviderError(RuntimeError):
    """Controlled model-provider failure that callers may safely degrade around."""


def _normalise_url(value: str | None) -> str:
    return (value or "").strip().rstrip("/")


def _extract_json_object(text: str) -> dict:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", text or "", flags=re.S)
        if not match:
            raise ModelProviderError("Model response did not contain a JSON object.")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ModelProviderError("Model returned invalid JSON.") from exc
    if not isinstance(value, dict):
        raise ModelProviderError("Model JSON response must be an object.")
    return value


def _message_content(payload: dict) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelProviderError("Model response did not contain message content.") from exc
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        ).strip()
    raise ModelProviderError("Model message content had an unsupported shape.")


@dataclass(slots=True)
class ProviderHealth:
    configured: bool
    reachable: bool
    url: str | None
    message: str

    def to_dict(self) -> dict:
        return {
            "configured": self.configured,
            "reachable": self.reachable,
            "url": self.url,
            "message": self.message,
        }


class BaseModelProvider:
    provider_name = "none"

    def health(self) -> dict:
        raise NotImplementedError

    def chat(self, messages, json_mode=False, temperature=0.1) -> str:
        raise NotImplementedError


class NoneModelProvider(BaseModelProvider):
    provider_name = "none"

    def __init__(self, message="Model-assisted reasoning is disabled."):
        self.message = message

    def health(self) -> dict:
        return ProviderHealth(False, False, None, self.message).to_dict()

    def chat(self, messages, json_mode=False, temperature=0.1) -> str:
        raise ModelProviderError(self.message)


class OpenAICompatibleProvider(BaseModelProvider):
    """OpenAI-compatible chat client used by llama.cpp and optional providers."""

    def __init__(self, provider_name, base_url, model, api_key="", timeout=120, opener=None):
        self.provider_name = provider_name
        self.base_url = _normalise_url(base_url)
        self.model = (model or "local").strip()
        self.api_key = (api_key or "").strip()
        self.timeout = max(1.0, float(timeout))
        self._opener = opener or urlopen
        self._last_health = None

    @property
    def configured(self):
        if not self.base_url:
            return False
        return self.provider_name == "llama_cpp" or bool(self.api_key)

    def _request_json(self, url, payload=None, timeout=None):
        headers = {"Accept": "application/json"}
        data = None
        method = "GET"
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
            method = "POST"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(url, data=data, headers=headers, method=method)
        with self._opener(request, timeout=timeout or self.timeout) as response:
            return json.load(response)

    def _api_url(self, path):
        """Join OpenAI-compatible paths without duplicating a configured /v1."""
        if self.base_url.endswith("/v1") and path.startswith("/v1/"):
            return f"{self.base_url}{path[3:]}"
        return f"{self.base_url}{path}"

    def health(self) -> dict:
        if not self.configured:
            message = (
                f"{self.provider_name} requires a base URL and API key."
                if self.provider_name != "llama_cpp"
                else "llama.cpp text endpoint is not configured."
            )
            self._last_health = ProviderHealth(False, False, self.base_url or None, message).to_dict()
            return self._last_health
        errors = []
        for path in ("/health", "/v1/models"):
            try:
                self._request_json(self._api_url(path), timeout=min(self.timeout, 2.0))
                self._last_health = ProviderHealth(
                    True, True, self.base_url, f"{self.provider_name} endpoint is reachable."
                ).to_dict()
                return self._last_health
            except HTTPError as exc:
                errors.append(f"HTTP {exc.code}")
                if exc.code not in {404, 405}:
                    break
            except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
                errors.append(str(exc) or exc.__class__.__name__)
                break
        detail = errors[-1] if errors else "connection failed"
        self._last_health = ProviderHealth(
            True,
            False,
            self.base_url,
            f"{self.provider_name} server is not reachable at {self.base_url}: {detail}",
        ).to_dict()
        return self._last_health

    def _chat_once(self, messages, json_mode, temperature):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = self._request_json(self._api_url("/v1/chat/completions"), payload)
        return _message_content(response)

    def chat(self, messages, json_mode=False, temperature=0.1) -> str:
        if not self.configured:
            raise ModelProviderError(self.health()["message"])
        try:
            content = self._chat_once(messages, json_mode=json_mode, temperature=temperature)
        except HTTPError as exc:
            if not json_mode:
                raise ModelProviderError(f"{self.provider_name} returned HTTP {exc.code}.") from exc
            try:
                content = self._chat_once(messages, json_mode=False, temperature=temperature)
            except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as retry_exc:
                raise ModelProviderError(f"{self.provider_name} request failed after JSON-mode retry.") from retry_exc
        except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelProviderError(f"{self.provider_name} request failed: {exc}") from exc
        if json_mode:
            return json.dumps(_extract_json_object(content), separators=(",", ":"))
        if not content:
            raise ModelProviderError(f"{self.provider_name} returned an empty response.")
        return content


class LlamaCppProvider(OpenAICompatibleProvider):
    provider_name = "llama_cpp"

    def __init__(self, text_url=None, vision_url=None, timeout=None, opener=None):
        super().__init__(
            provider_name="llama_cpp",
            base_url=text_url or os.getenv("LLAMA_CPP_TEXT_URL", "http://127.0.0.1:8080"),
            model=os.getenv("LLAMA_CPP_TEXT_MODEL", "local"),
            timeout=timeout or os.getenv("LLAMA_CPP_TIMEOUT_SECONDS", "120"),
            opener=opener,
        )
        self.vision_url = _normalise_url(
            vision_url if vision_url is not None else os.getenv("LLAMA_CPP_VISION_URL", "http://127.0.0.1:8081")
        )


def create_model_provider(provider_name=None, opener=None):
    selected = (provider_name or os.getenv("MODEL_PROVIDER", "none")).strip().casefold()
    if selected == "llama_cpp":
        return LlamaCppProvider(opener=opener)
    if selected == "nvidia":
        return OpenAICompatibleProvider(
            "nvidia",
            os.getenv("NVIDIA_BASE_URL") or os.getenv("NVIDIA_API_BASE_URL", ""),
            os.getenv("NVIDIA_TEXT_MODEL") or os.getenv("NVIDIA_VLM_MODEL", ""),
            os.getenv("NVIDIA_API_KEY", ""),
            os.getenv("NVIDIA_API_TIMEOUT", "30"),
            opener,
        )
    if selected == "groq":
        return OpenAICompatibleProvider(
            "groq",
            os.getenv("GROQ_BASE_URL", ""),
            os.getenv("GROQ_MODEL", ""),
            os.getenv("GROQ_API_KEY", ""),
            os.getenv("GROQ_API_TIMEOUT", "30"),
            opener,
        )
    if selected == "none":
        return NoneModelProvider()
    return NoneModelProvider(f"Unknown MODEL_PROVIDER value: {selected}")


def _probe_optional_endpoint(url, timeout=2.0):
    endpoint = _normalise_url(url)
    if not endpoint:
        return ProviderHealth(False, False, None, "Vision endpoint is not configured.").to_dict()
    provider = OpenAICompatibleProvider("llama_cpp", endpoint, "local", timeout=timeout)
    result = provider.health()
    if not result["reachable"]:
        result["message"] = (
            f"Vision endpoint is not reachable at {endpoint}. "
            "Captioning and visual verification will be skipped."
        )
    return result


def model_health_snapshot(provider=None):
    provider = provider or create_model_provider()
    selected = provider.provider_name
    text = provider.health()
    if selected == "llama_cpp":
        vision_url = os.getenv("LLAMA_CPP_VISION_URL", "http://127.0.0.1:8081")
        vision = _probe_optional_endpoint(vision_url)
    elif selected == "nvidia":
        # NVIDIA's configured VLM uses the same authenticated multimodal
        # endpoint as the text-compatible health probe.
        vision = dict(text)
        vision["message"] = (
            "NVIDIA multimodal endpoint is reachable."
            if vision["reachable"]
            else "NVIDIA multimodal endpoint is unavailable."
        )
    else:
        vision = ProviderHealth(
            False, False, None, "No vision endpoint is configured for the selected provider."
        ).to_dict()
    return {
        "selected_provider": selected,
        "text_model": text,
        "vision_model": vision,
        "external_providers": {
            "nvidia": "selected" if selected == "nvidia" else "disabled",
            "groq": "selected" if selected == "groq" else "disabled",
        },
    }
