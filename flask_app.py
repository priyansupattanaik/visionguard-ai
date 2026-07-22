"""Compatibility launcher for VisionGuard's Flask application."""
from app.web.server import _serialize_match, app, create_app


if __name__ == "__main__":
    import os

    host = os.getenv("VISION_GUARD_HOST", "127.0.0.1")
    port = int(os.getenv("VISION_GUARD_PORT", "7860"))
    print(f"Open Vision Guard at http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
