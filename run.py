"""Human-readable project entry point.

Run the working video-search UI by default. Set VISIONGUARD_APP=api to run the
new evidence-ledger API and console instead.
"""
from __future__ import annotations

import os
from pathlib import Path

from visionguard.runtime.env import load_project_env

ROOT = Path(__file__).resolve().parent
load_project_env(ROOT)


def main() -> None:
    mode = os.getenv("VISIONGUARD_APP", "video_ui").strip().casefold()
    if mode == "api":
        import uvicorn
        from visionguard.evidence_api.config import Settings

        settings = Settings.from_env()
        uvicorn.run("visionguard.evidence_api.api.app:app", host=settings.host, port=settings.port)
        return
    if mode != "video_ui":
        raise SystemExit("VISIONGUARD_APP must be 'video_ui' or 'api'")
    from visionguard.web_app.server import create_app

    host = os.getenv("VISION_GUARD_HOST", "127.0.0.1")
    port = int(os.getenv("VISION_GUARD_PORT", "7860"))
    print(f"VisionGuard video UI: http://{host}:{port}")
    create_app().run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
