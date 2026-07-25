from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path = Path(".visionguard")
    database_name: str = "visionguard.sqlite3"
    min_retrieval_score: float = 0.15
    min_verification_confidence: float = 0.65
    max_evidence: int = 20
    frame_interval_seconds: float = 1.0
    host: str = "127.0.0.1"
    port: int = 8000
    lexical_weight: float = 0.55
    kind_weight: float = 0.20
    confidence_weight: float = 0.25
    kind_mismatch_score: float = 0.25
    retrieval_pool_multiplier: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        from visionguard.runtime.env import load_project_env
        load_project_env(Path(__file__).resolve().parents[2])
        return cls(
            data_dir=Path(os.getenv("VISIONGUARD_DATA_DIR", ".visionguard")),
            min_retrieval_score=float(os.getenv("VISIONGUARD_MIN_RETRIEVAL_SCORE", "0.15")),
            min_verification_confidence=float(os.getenv("VISIONGUARD_MIN_VERIFICATION_CONFIDENCE", "0.65")),
            max_evidence=int(os.getenv("VISIONGUARD_MAX_EVIDENCE", "20")),
            frame_interval_seconds=float(os.getenv("VISIONGUARD_FRAME_INTERVAL_SECONDS", "1.0")),
            host=os.getenv("VISIONGUARD_HOST", "127.0.0.1"),
            port=int(os.getenv("VISIONGUARD_PORT", "8000")),
            lexical_weight=float(os.getenv("VISIONGUARD_LEXICAL_WEIGHT", "0.55")),
            kind_weight=float(os.getenv("VISIONGUARD_KIND_WEIGHT", "0.20")),
            confidence_weight=float(os.getenv("VISIONGUARD_CONFIDENCE_WEIGHT", "0.25")),
            kind_mismatch_score=float(os.getenv("VISIONGUARD_KIND_MISMATCH_SCORE", "0.25")),
            retrieval_pool_multiplier=int(os.getenv("VISIONGUARD_RETRIEVAL_POOL_MULTIPLIER", "3")),
        )

    @property
    def database_path(self) -> Path:
        return self.data_dir / self.database_name
