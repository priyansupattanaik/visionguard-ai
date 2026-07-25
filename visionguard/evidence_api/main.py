import uvicorn
from pathlib import Path

from visionguard.runtime.env import load_project_env
from visionguard.evidence_api.config import Settings


if __name__ == "__main__":
    load_project_env(Path(__file__).resolve().parents[2])
    settings = Settings.from_env()
    uvicorn.run(
        "visionguard.evidence_api.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
