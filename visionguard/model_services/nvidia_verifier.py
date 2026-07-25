"""Evidence-frame verifier for selected llama.cpp vision or optional NVIDIA VLM."""
import base64
import json
import mimetypes
import os
import re
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from visionguard.model_services.model_provider import OpenAICompatibleProvider


class NvidiaFrameVerifier:
    def __init__(self, model=None, timeout=None):
        self.selected_provider = os.getenv("MODEL_PROVIDER", "llama_cpp").strip().casefold()
        if self.selected_provider == "llama_cpp":
            self.base_url = os.getenv("LLAMA_CPP_VISION_URL", "http://127.0.0.1:8081").strip().rstrip("/")
            self.endpoint = f"{self.base_url}/v1" if self.base_url and not self.base_url.endswith("/v1") else self.base_url
            self.model_name = os.getenv("LLAMA_CPP_VISION_MODEL", "local")
            self.api_key = ""
            self.timeout = float(timeout or os.getenv("LLAMA_CPP_TIMEOUT_SECONDS", "120"))
            self.available_mode = "llama_cpp_vision"
        elif self.selected_provider == "nvidia":
            self.base_url = (os.getenv("NVIDIA_BASE_URL") or os.getenv("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1")).rstrip("/")
            self.endpoint = self.base_url
            self.model_name = model or os.getenv("NVIDIA_VLM_MODEL", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1")
            self.api_key = os.getenv("NVIDIA_API_KEY", "").strip()
            self.timeout = float(timeout or os.getenv("NVIDIA_API_TIMEOUT", "30"))
            self.available_mode = "nvidia_api"
        else:
            self.base_url = ""
            self.endpoint = ""
            self.model_name = ""
            self.api_key = ""
            self.timeout = float(timeout or 30)
            self.available_mode = "verification_disabled"
        self.backend = "unconfigured"
        self.cache, self.lock = {}, threading.Lock()
        self._last_error = None
        self._ready = False

    def warmup(self):
        if self.selected_provider == "llama_cpp":
            if not self.base_url:
                self.backend = "unconfigured"
                self._ready = False
                self._last_error = None
                return False
            result = OpenAICompatibleProvider(
                "llama_cpp", self.base_url, self.model_name, timeout=min(self.timeout, 2.0)
            ).health()
            self._ready = bool(result["reachable"])
            self._last_error = None if self._ready else result["message"]
            self.backend = self.available_mode if self._ready else "unavailable"
            return self._ready
        if self.selected_provider != "nvidia" or not self.api_key:
            self.backend = "unconfigured"
            self._ready = False
            self._last_error = None
            return False
        self.backend = self.available_mode
        self._ready = True
        self._last_error = None
        return True

    def verification_mode(self):
        if self.available_mode == "verification_disabled":
            return "verification_disabled"
        if self.selected_provider == "nvidia" and not self.api_key:
            return "nvidia_api_unconfigured"
        if self.selected_provider == "llama_cpp" and not self.base_url:
            return "llama_cpp_vision_unconfigured"
        if self._last_error:
            return f"{self.available_mode}_unavailable"
        return self.available_mode if self._ready else f"{self.available_mode}_unavailable"

    def is_ready(self):
        return bool(self._ready) and self._last_error is None

    def _threshold(self, query):
        return float(os.getenv("NVIDIA_VERIFY_THRESHOLD", "0.45"))

    @staticmethod
    def _extract_json(text):
        match = re.search(r"\{.*\}", text or "", flags=re.S)
        try:
            return json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _clean_boxes(boxes, size):
        w, h = size
        clean = []
        for box in boxes or []:
            if isinstance(box, dict):
                box = box.get("box") or box.get("bbox") or box.get("coordinates")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                vals = [float(v) for v in box]
            except (TypeError, ValueError):
                continue
            if max(vals) <= 1.5:
                vals = [vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h]
            x1, y1, x2, y2 = (
                max(0, min(w, vals[0])),
                max(0, min(h, vals[1])),
                max(0, min(w, vals[2])),
                max(0, min(h, vals[3])),
            )
            if x2 > x1 and y2 > y1:
                clean.append([x1, y1, x2, y2])
        return clean

    def _empty_result(self, mode=None):
        return {
            "matched": False,
            "confidence": 0.0,
            "caption": "",
            "boxes": [],
            "verification_mode": mode or self.verification_mode(),
        }

    def _ask(self, frame_path, prompt):
        if not self.is_ready():
            return ""
        if not frame_path or not os.path.isfile(frame_path):
            return ""
        mime = mimetypes.guess_type(frame_path)[0] or "image/jpeg"
        with open(frame_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "max_tokens": 300,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            }],
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
            content = body["choices"][0]["message"]["content"] or ""
            self._last_error = None
            self.backend = self.available_mode
            return content
        except HTTPError as exc:
            self._last_error = f"HTTP {exc.code}"
            self.backend = "unavailable"
            return ""
        except (URLError, OSError, KeyError, IndexError, json.JSONDecodeError, TimeoutError) as exc:
            self._last_error = str(exc) or exc.__class__.__name__
            self.backend = "unavailable"
            return ""

    def verify_query(self, frame_path, query, frame_key=None):
        if not self.is_ready():
            return self._empty_result(self.verification_mode())
        key = (frame_key or frame_path, " ".join(query.lower().split()))
        with self.lock:
            if key in self.cache:
                return dict(self.cache[key])
        try:
            image = Image.open(frame_path).convert("RGB") if frame_path and os.path.isfile(frame_path) else None
        except Exception:
            image = None
        if image is None:
            return self._empty_result(self.verification_mode())
        prompt = (
            f"Verify this CCTV frame against the exact query: {query}. Be conservative. "
            'Return JSON only: {"matched": boolean, "confidence": number from 0 to 1, '
            '"description": "short evidence", "boxes": [[x1,y1,x2,y2]]}. '
            "Set matched false if the query is not visibly proven or a box cannot be supplied."
        )
        raw = self._ask(frame_path, prompt)
        if not raw and self._last_error:
            return self._empty_result(self.verification_mode())
        data = self._extract_json(raw)
        boxes = self._clean_boxes(data.get("boxes"), image.size)
        confidence = float(data.get("confidence", 0) or 0)
        result = {
            "matched": bool(data.get("matched")) and bool(boxes) and confidence >= self._threshold(query),
            "confidence": max(0.0, min(1.0, confidence)),
            "caption": str(data.get("description", "")).strip(),
            "boxes": boxes,
            "verification_mode": self.verification_mode(),
        }
        with self.lock:
            self.cache[key] = dict(result)
        return result

    def ground_phrase(self, frame_path, phrase, multi=True, frame_key=None):
        try:
            return self.verify_query(frame_path, phrase, frame_key).get("boxes", [])
        except Exception:
            return []
