"""Required NVIDIA multimodal enrichment for indexed evidence segments."""
from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SemanticAnalysisError(RuntimeError):
    """A required semantic stage failed; indexing must not continue silently."""


@dataclass(frozen=True, slots=True)
class SemanticSegment:
    caption: str
    scene_tags: list[str]
    event_tags: list[str]
    confidence: float

    def to_dict(self) -> dict:
        return {"caption": self.caption, "scene_tags": self.scene_tags, "event_tags": self.event_tags, "confidence": self.confidence, "provider": "nvidia"}


class NvidiaSemanticAnalyzer:
    """Calls NVIDIA for each evidence segment and validates structured output."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 30.0):
        self.api_key, self.base_url, self.model = (api_key or "").strip(), (base_url or "").strip().rstrip("/"), (model or "").strip()
        self.timeout = max(1.0, float(timeout))
        if not self.api_key:
            raise SemanticAnalysisError("NVIDIA_API_KEY is required for the semantic indexing stage.")
        if not self.base_url or not self.model:
            raise SemanticAnalysisError("NVIDIA_API_BASE_URL and NVIDIA_VLM_MODEL are required for semantic indexing.")

    @property
    def endpoint(self) -> str:
        return self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"

    def health(self) -> dict:
        request = Request(
            f"{self.endpoint}/models",
            method="GET",
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=min(self.timeout, 5.0)) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise SemanticAnalysisError("NVIDIA model probe returned an invalid response.")
            return {"ready": True, "provider": "nvidia", "model": self.model,
                    "message": "NVIDIA semantic endpoint authenticated successfully."}
        except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError, SemanticAnalysisError) as exc:
            return {"ready": False, "provider": "nvidia", "model": self.model,
                    "message": f"NVIDIA semantic endpoint is not reachable or authenticated: {exc}"}

    @staticmethod
    def _tags(value) -> list[str]:
        if not isinstance(value, list):
            raise SemanticAnalysisError("NVIDIA semantic response must include tag lists.")
        return sorted({str(item).strip().casefold() for item in value if str(item).strip()})

    def analyze(self, frame_path: str, *, objects: list[str], tracks: list[int], start: float, end: float) -> SemanticSegment:
        path = Path(frame_path)
        if not path.is_file():
            raise SemanticAnalysisError(f"Semantic evidence frame is missing: {path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        image = base64.b64encode(path.read_bytes()).decode("ascii")
        prompt = ("You are enriching a CCTV evidence segment. Return JSON only with caption (string), scene_tags (array), "
                  "event_tags (array), and confidence (0..1). Describe only visible facts; do not infer identities or "
                  f"unseen causes. Detector context: objects={objects}; tracks={tracks}; interval={start:.3f}-{end:.3f}s.")
        body = {"model": self.model, "temperature": 0, "max_tokens": 300, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}},
        ]}]}
        request = Request(f"{self.endpoint}/chat/completions", data=json.dumps(body).encode("utf-8"), method="POST",
                          headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
            content = result["choices"][0]["message"]["content"]
            payload = json.loads(content if isinstance(content, str) else "")
        except (HTTPError, URLError, OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise SemanticAnalysisError(f"NVIDIA semantic analysis failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise SemanticAnalysisError("NVIDIA semantic response must be a JSON object.")
        caption = str(payload.get("caption", "")).strip()
        if not caption:
            raise SemanticAnalysisError("NVIDIA semantic response did not contain a caption.")
        try:
            confidence = min(1.0, max(0.0, float(payload.get("confidence"))))
        except (TypeError, ValueError) as exc:
            raise SemanticAnalysisError("NVIDIA semantic response must include numeric confidence.") from exc
        return SemanticSegment(caption, self._tags(payload.get("scene_tags")), self._tags(payload.get("event_tags")), confidence)
