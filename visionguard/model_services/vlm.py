import os

import cv2
import numpy as np
import torch
from PIL import Image
from visionguard.runtime.cache import resolve_device
from visionguard.model_services.metadata_encoder import MetadataSearchEncoder


def _normalize_embedding(vec):
    """Force an embedding tensor/array to a 1D float32 unit vector of shape (D,)."""
    arr = np.asarray(vec, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return np.zeros((1,), dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    n = float(np.linalg.norm(arr))
    if n == 0.0:
        return arr.astype(np.float32, copy=False)
    return (arr / n).astype(np.float32)


class SearchEncoder:
    def __init__(self, model="google/siglip2-so400m-patch14-384", device=None):
        self.model_name = os.getenv("CLIP_MODEL") or model
        self.dev = resolve_device(device)
        env_batch_size = os.getenv("IMAGE_BATCH_SIZE")
        self.image_batch_size = int(env_batch_size) if env_batch_size else self._default_image_batch_size()
        self.p = None
        self.m = None
        self.compiled = False
        self.fallback = False
        self.fallback_reason = None
        self.fallback_dimension = int(os.getenv("EMBEDDING_FALLBACK_DIMENSION", "256"))
        self.metadata_encoder = MetadataSearchEncoder(self.fallback_dimension)

    def _default_image_batch_size(self):
        if self.dev != "cuda":
            return 8
        try:
            gpu_name = torch.cuda.get_device_name(0).lower()
        except Exception:
            gpu_name = ""
        return 32 if "a100" in gpu_name else 16

    def load(self):
        if self.m is not None or self.fallback:
            return
        local_only = os.getenv("VISION_GUARD_LOCAL_MODELS_ONLY", "1") == "1"
        if local_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoModel, AutoProcessor
        try:
            self.p = AutoProcessor.from_pretrained(self.model_name, local_files_only=local_only)
            dtype = torch.float16 if self.dev == "cuda" else torch.float32
            self.m = AutoModel.from_pretrained(self.model_name, dtype=dtype, device_map=None, local_files_only=local_only)
            self.m.to(self.dev)
            self.m.eval()
            self._maybe_compile()
        except (OSError, ValueError) as exc:
            if os.getenv("VISION_GUARD_ALLOW_EMBEDDING_FALLBACK", "1") != "1":
                raise
            self.fallback = True
            self.fallback_reason = str(exc)

    def mode(self):
        if self.fallback:
            return "metadata_embeddings"
        if self.m is not None:
            return "semantic_embeddings"
        return "not_loaded"

    def embed_metadata(self, metadata):
        self.load()
        if self.fallback:
            return self.metadata_encoder.encode_metadata(metadata)
        raise RuntimeError("metadata embeddings are only used by the offline fallback")

    def _maybe_compile(self):
        # torch.compile with CUDA graphs is incompatible with Gradio's
        # worker-thread execution model (TLS assertion failure at runtime).
        # SigLIP2-So400m is fast enough on GPU without compilation.
        self.compiled = False

    def _vec(self, x):
        if hasattr(x, "pooler_output"):
            return x.pooler_output
        if hasattr(x, "image_embeds"):
            return x.image_embeds
        if hasattr(x, "text_embeds"):
            return x.text_embeds
        return x

    def _norm(self, x):
        raw = self._vec(x).detach().cpu().numpy()
        # Handle (1, D) or (D,) outputs from single-item inference.
        if raw.ndim >= 2:
            raw = raw[0]
        return _normalize_embedding(raw)

    def embed_text(self, txt):
        self.load()
        if self.fallback:
            return self.metadata_encoder.encode_text(txt)
        txt = f"this is a photo of {txt.strip().lower()}"
        inp = self.p(text=[txt], return_tensors="pt").to(self.dev)
        with torch.no_grad():
            vec = self.m.get_text_features(**inp)
        return self._norm(vec)

    def embed_frame(self, frame):
        self.load()
        if self.fallback:
            # Raw images cannot be truthfully mapped into a text feature space
            # without a vision model. Call embed_frames(..., metadata=...) while
            # indexing so the fallback is grounded in detector/tracker facts.
            return np.zeros((self.fallback_dimension,), dtype=np.float32)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inp = self.p(images=img, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            vec = self.m.get_image_features(**inp)
        return self._norm(vec)

    def embed_frames(self, frames, metadata=None):
        self.load()
        if not frames:
            return []
        if self.fallback:
            if metadata is None:
                return [np.zeros((self.fallback_dimension,), dtype=np.float32) for _ in frames]
            if len(metadata) != len(frames):
                raise ValueError("metadata and frames length mismatch")
            return [self.metadata_encoder.encode_metadata(row) for row in metadata]
        out = []
        for offset in range(0, len(frames), self.image_batch_size):
            batch = frames[offset: offset + self.image_batch_size]
            imgs = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in batch]
            inp = self.p(images=imgs, return_tensors="pt").to(self.dev)
            if self.dev == "cuda":
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    vecs = self._vec(self.m.get_image_features(**inp)).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    vecs = self._vec(self.m.get_image_features(**inp)).detach().cpu().numpy()
            vecs = np.asarray(vecs)
            # Guarantee batch axis even if model returns a single (D,) vector.
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            elif vecs.ndim > 2:
                # e.g. (B, 1, D) -> (B, D)
                vecs = np.reshape(vecs, (vecs.shape[0], -1))
            for vec in vecs:
                out.append(_normalize_embedding(vec))
        return out
