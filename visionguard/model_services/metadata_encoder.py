"""Deterministic text/metadata embeddings used when a vision model is unavailable.

The fallback deliberately indexes only facts produced by the detector and tracker.
It does not pretend to understand raw pixels. Query terms and indexed metadata share
the same feature space, which makes the fallback searchable and explainable.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

import numpy as np

from visionguard.evidence_api.query_language import load_query_vocabulary


class MetadataSearchEncoder:
    def __init__(self, dimension: int = 256):
        self.dimension = max(64, int(dimension))
        vocabulary = load_query_vocabulary()
        self.aliases = {
            str(alias).casefold(): [str(value).casefold() for value in values]
            for alias, values in vocabulary.get("object_aliases", {}).items()
        }

    @staticmethod
    def _words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", str(text).casefold())

    def _features(self, text: str) -> Counter:
        words = self._words(text)
        features = Counter(words)
        for left, right in zip(words, words[1:]):
            features[f"{left}_{right}"] += 1.5
        for alias, canonical_names in self.aliases.items():
            alias_words = self._words(alias)
            if not alias_words:
                continue
            phrase = " ".join(words)
            if re.search(rf"\b{re.escape(' '.join(alias_words))}\b", phrase):
                for name in canonical_names:
                    features[name] += 2.0
        return features

    def _encode_features(self, features: Counter) -> np.ndarray:
        vector = np.zeros((self.dimension,), dtype=np.float32)
        for feature, weight in features.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % self.dimension
            vector[index] += float(weight)
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0.0 else vector / norm

    def encode_text(self, text: str) -> np.ndarray:
        return self._encode_features(self._features(text))

    def encode_metadata(self, metadata: dict | None) -> np.ndarray:
        metadata = metadata or {}
        features = Counter()

        objects = metadata.get("objects", {})
        if isinstance(objects, dict):
            for name, count in objects.items():
                features.update({str(name).casefold(): max(1.0, float(count)) * 2.0})
        else:
            for name in objects or []:
                features.update(self._features(str(name)))

        for appearance in metadata.get("appearances", []) or []:
            for feature, weight in self._features(str(appearance)).items():
                features[feature] += weight * 2.0

        for detection in metadata.get("detections", []) or []:
            name = str(detection.get("name", "")).casefold().strip()
            color = str(detection.get("color") or "").casefold().strip()
            if name:
                features[name] += 1.5
            if color:
                features[color] += 1.5
                features[f"{color}_{name}"] += 2.5

        if float(metadata.get("motion_score", 0.0) or 0.0) >= 0.025:
            features["motion"] += 1.0
            features["moving"] += 1.0
        return self._encode_features(features)
