"""Start the VisionGuard video-search application."""
from __future__ import annotations

import os
from pathlib import Path

from visionguard.runtime.env import load_project_env

ROOT = Path(__file__).resolve().parent
load_project_env(ROOT)


def main() -> None:
    from visionguard.web_app.server import app

    host = os.getenv("VISION_GUARD_HOST", "127.0.0.1")
    port = int(os.getenv("VISION_GUARD_PORT", "7860"))
    print(f"VisionGuard video UI: http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True, load_dotenv=False)


if __name__ == "__main__":
    main()
