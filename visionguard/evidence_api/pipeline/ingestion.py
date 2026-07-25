from __future__ import annotations

import hashlib
from pathlib import Path

from visionguard.evidence_api.database.sqlite import SQLiteEvidenceRepository
from visionguard.evidence_api.schemas.domain import VideoAsset


class InvalidVideoError(ValueError):
    pass


class VideoIngestor:
    def __init__(self, repository: SQLiteEvidenceRepository):
        self.repository = repository

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def ingest(self, source: Path | str) -> tuple[VideoAsset, bool]:
        try:
            import cv2
        except ImportError as exc:
            raise InvalidVideoError(
                "OpenCV is not installed. Install requirements.txt to enable video ingestion."
            ) from exc
        path = Path(source).resolve()
        if not path.is_file():
            raise InvalidVideoError(f"video does not exist: {path}")
        sha256 = self._hash(path)
        cached = self.repository.get_video_by_hash(sha256)
        if cached:
            return cached, True
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise InvalidVideoError(f"unsupported or corrupt video: {path.name}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec_int = int(capture.get(cv2.CAP_PROP_FOURCC))
        capture.release()
        if fps <= 0 or count <= 0 or width <= 0 or height <= 0:
            raise InvalidVideoError("video metadata is incomplete")
        codec = "".join(chr((codec_int >> 8 * i) & 0xFF) for i in range(4)).strip("\x00")
        asset = VideoAsset(
            source_path=str(path), sha256=sha256, filename=path.name,
            duration_seconds=count / fps, fps=fps, frame_count=count,
            width=width, height=height, codec=codec or None,
        )
        self.repository.save_video(asset)
        return asset, False
