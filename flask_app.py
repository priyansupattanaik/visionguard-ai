import os
import threading
from pathlib import Path
from uuid import uuid4

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from cache_utils import setup_cache
from pipeline import VisionGuardPipeline

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
DOWNLOADS = {}


def _asset_rows():
    assets = ROOT / "assets"
    if not assets.exists():
        return []
    rows = []
    for path in sorted(assets.glob("*.mp4")):
        rows.append({
            "name": path.name,
            "path": str(path),
            "size": path.stat().st_size,
        })
    return rows


def _verification_label(mode):
    if mode == "real_qwen":
        return "Real Qwen verification"
    if mode == "detector_only_dev_passthrough":
        return "Detector-only dev passthrough: Qwen skipped on Windows CPU"
    return "Verification unavailable"


def _verification_warning(mode):
    if mode == "detector_only_dev_passthrough":
        return "Qwen did not run on this Windows CPU environment. Results are detector/retrieval matches, not real Qwen verification."
    if mode == "unknown":
        return "Verification backend is not ready or unavailable."
    return ""


def _safe_output_file(path):
    candidate = Path(path).resolve()
    output_root = OUTPUT_DIR.resolve()
    try:
        candidate.relative_to(output_root)
    except ValueError:
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


def _register_download(path):
    safe = _safe_output_file(path)
    if safe is None:
        return None
    token = uuid4().hex
    DOWNLOADS[token] = str(safe)
    return {
        "id": token,
        "name": safe.name,
        "url": f"/api/download/{token}",
        "size": safe.stat().st_size,
    }


def _serialize_match(row):
    mode = row.get("verification_mode") or "unknown"
    return {
        "id": row.get("label"),
        "label": row.get("label"),
        "start": round(float(row.get("start", 0.0)), 2),
        "end": round(float(row.get("end", 0.0)), 2),
        "peak_ts": round(float(row.get("peak_ts", row.get("start", 0.0))), 2),
        "score": round(float(row.get("score", 0.0)), 4),
        "objects": row.get("objects", []),
        "summary": row.get("summary", ""),
        "verification_mode": mode,
        "verification_label": _verification_label(mode),
        "low_confidence": bool(row.get("low_confidence")),
    }


def create_app(testing=False, start_warmup=True, pipeline=None):
    setup_cache()
    app = Flask(__name__)
    app.config["TESTING"] = testing
    app.config["PIPELINE"] = pipeline or VisionGuardPipeline()
    app.config["LOCK"] = threading.RLock()
    app.config["LAST_QUERY"] = ""
    OUTPUT_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if start_warmup and not testing:
        threading.Thread(target=app.config["PIPELINE"].warmup_models, daemon=True).start()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def api_status():
        pipe = app.config["PIPELINE"]
        mode = pipe.verification_mode()
        return jsonify({
            "ok": True,
            "status": pipe.warmup_status(),
            "scanned": bool(pipe.idx),
            "verification_mode": mode,
            "verification_label": _verification_label(mode),
            "warning": _verification_warning(mode),
        })

    @app.get("/api/assets")
    def api_assets():
        return jsonify({"ok": True, "assets": _asset_rows()})

    @app.post("/api/scan")
    def api_scan():
        pipe = app.config["PIPELINE"]
        video_path = None
        warnings = []

        if request.files.get("video"):
            upload = request.files["video"]
            name = secure_filename(upload.filename or "upload.mp4")
            if not name.lower().endswith(".mp4"):
                return jsonify({"ok": False, "message": "Only MP4 uploads are supported."}), 400
            target = UPLOAD_DIR / f"{uuid4().hex}_{name}"
            upload.save(target)
            video_path = str(target)
        else:
            data = request.get_json(silent=True) or request.form
            sample = str(data.get("sample", "")).strip()
            if sample:
                sample_name = Path(sample).name
                match = ROOT / "assets" / sample_name
                if not match.exists() or match.suffix.lower() != ".mp4":
                    return jsonify({"ok": False, "message": "Selected sample asset was not found."}), 404
                video_path = str(match)

        if not video_path:
            return jsonify({"ok": False, "message": "Choose a sample asset or upload an MP4 video."}), 400

        try:
            with app.config["LOCK"]:
                meta = None
                for event in pipe.index_video_iter(video_path):
                    if event.get("kind") == "done":
                        meta = event["meta"]
                if meta is None:
                    return jsonify({"ok": False, "message": "Scan did not complete."}), 500
                mode = pipe.verification_mode()
                warning = _verification_warning(mode)
                if warning:
                    warnings.append(warning)
                return jsonify({
                    "ok": True,
                    "message": "Scan complete.",
                    "video": os.path.basename(meta["video"]),
                    "indexed_windows": int(meta.get("segments", 0)),
                    "objects": sorted(meta.get("object_counts", {}).keys()),
                    "object_counts": meta.get("object_counts", {}),
                    "backend": {
                        "retriever": meta.get("retriever", "unknown"),
                        "segment_retriever": meta.get("segment_retriever", "unknown"),
                        "verification_mode": mode,
                        "verification_label": _verification_label(mode),
                    },
                    "warnings": warnings,
                })
        except Exception as exc:
            return jsonify({"ok": False, "message": str(exc)}), 500

    @app.post("/api/query")
    def api_query():
        pipe = app.config["PIPELINE"]
        data = request.get_json(silent=True) or {}
        query = str(data.get("query", "")).strip()
        if not pipe.idx:
            return jsonify({"ok": False, "message": "Scan a video before searching."}), 400
        if not query:
            return jsonify({"ok": False, "message": "Enter a query before searching."}), 400
        try:
            with app.config["LOCK"]:
                hits = pipe.search(query, top_k=4)
                rows = pipe.prepare_hits(hits, query)
                app.config["LAST_QUERY"] = query
                mode = pipe.verification_mode()
                return jsonify({
                    "ok": True,
                    "query": query,
                    "matches": [_serialize_match(row) for row in rows],
                    "verification_mode": mode,
                    "verification_label": _verification_label(mode),
                    "warning": _verification_warning(mode),
                })
        except Exception as exc:
            return jsonify({"ok": False, "message": str(exc)}), 500

    @app.post("/api/export")
    def api_export():
        pipe = app.config["PIPELINE"]
        data = request.get_json(silent=True) or {}
        selected = data.get("selected") or []
        query = str(data.get("query") or app.config.get("LAST_QUERY") or "").strip()
        if not pipe.idx:
            return jsonify({"ok": False, "message": "Scan and search before exporting."}), 400
        if not selected:
            return jsonify({"ok": False, "message": "Select at least one match to export."}), 400
        try:
            timeout = float(data.get("segment_timeout", 20))
        except (TypeError, ValueError):
            timeout = 20.0
        try:
            with app.config["LOCK"]:
                result = pipe.export_selected_detailed(selected, query, segment_timeout=timeout)
                files = {
                    key: _register_download(path)
                    for key, path in result.get("files", {}).items()
                    if path
                }
                files = {key: value for key, value in files.items() if value}
                return jsonify({
                    "ok": bool(result.get("ok")),
                    "message": result.get("message", ""),
                    "export_mode": result.get("export_mode", "unknown"),
                    "warnings": result.get("warnings", []),
                    "files": files,
                }), 200 if result.get("ok") else 400
        except Exception as exc:
            return jsonify({"ok": False, "message": str(exc), "export_mode": "failed"}), 500

    @app.get("/api/download/<download_id>")
    def api_download(download_id):
        path = DOWNLOADS.get(download_id)
        if not path:
            abort(404)
        safe = _safe_output_file(path)
        if safe is None:
            abort(404)
        return send_file(safe, as_attachment=True, download_name=safe.name)

    return app


app = create_app(start_warmup=os.getenv("VISION_GUARD_SKIP_WARMUP") != "1")


if __name__ == "__main__":
    host = os.getenv("VISION_GUARD_HOST", "127.0.0.1")
    port = int(os.getenv("VISION_GUARD_PORT", "7860"))
    print(f"Open Vision Guard at http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
