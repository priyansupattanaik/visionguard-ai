import json
import os
import threading
from pathlib import Path
from uuid import uuid4

from flask import Flask, Response, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from visionguard.runtime.cache import setup_cache
from visionguard.runtime.env import load_project_env
from visionguard.video_pipeline.video_pipeline import VisionGuardPipeline
from visionguard.web_app.video_jobs import (
    frame_from_progress_event,
    make_chunks,
    make_job,
    materialize_frames,
    probe_video,
    public_frame,
    public_job,
    update_stage,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[2]
load_project_env(ROOT)
OUTPUT_DIR = Path(os.getenv("VISION_GUARD_OUT_DIR", str(ROOT / "output")))
UPLOAD_DIR = OUTPUT_DIR / "uploads"
DOWNLOADS = {}


def _asset_rows():
    assets = ROOT / "sample_videos"
    if not assets.exists():
        return []
    rows = []
    for path in sorted(assets.glob("*.mp4")):
        rows.append({
            "name": path.name,
            "url": f"/api/assets/{path.name}",
            "size": path.stat().st_size,
        })
    return rows


def _verification_label(mode):
    if mode == "nvidia_api":
        return "NVIDIA API visual verification"
    if mode == "nvidia_api_unconfigured":
        return "NVIDIA API key is not configured"
    if mode == "nvidia_api_unavailable":
        return "NVIDIA API verification temporarily unavailable"
    return "Verification unavailable"


def _verification_warning(mode):
    if mode == "nvidia_api_unconfigured":
        return "Add NVIDIA_API_KEY to .env to enable visual verification. Results are detector/retrieval matches only."
    if mode == "nvidia_api_unavailable":
        return "NVIDIA verification API did not respond successfully. Showing detector/retrieval matches only."
    if mode == "unknown":
        return "Verification backend is not ready or unavailable."
    return ""


def _embedding_warning(mode):
    if mode == "detector_only":
        return "Semantic embeddings are unavailable; only detector-backed object queries are reliable."
    if mode == "metadata_embeddings":
        return "Semantic vision embeddings are unavailable; using searchable detector/tracker metadata instead."
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


def _serialize_match(row, evidence_frame=None):
    mode = row.get("verification_mode") or "unknown"
    payload = {
        "id": row.get("label"),
        "label": row.get("label"),
        "start": round(float(row.get("start", 0.0)), 2),
        "end": round(float(row.get("end", 0.0)), 2),
        "peak_ts": round(float(row.get("peak_ts", row.get("start", 0.0))), 2),
        "score": round(float(row.get("score", 0.0)), 4),
        "objects": row.get("objects", []),
        "tracks": row.get("tracks", []),
        "summary": row.get("summary", ""),
        "verification_mode": mode,
        "verification_label": _verification_label(mode),
        "low_confidence": bool(row.get("low_confidence")),
    }
    if evidence_frame is not None:
        payload["frame"] = public_frame(evidence_frame)
    return payload


def _evidence_frame_for_hit(frames, hit):
    if not frames:
        return None
    candidate_path = hit.get("representative_frame_path") or hit.get("frame_path")
    if candidate_path:
        resolved = str(Path(candidate_path).resolve())
        for frame in frames:
            if frame.get("_image_path") == resolved:
                return frame
    timestamp_ms = int(round(float(hit.get("peak_ts", hit.get("start", 0.0))) * 1000))
    return min(frames, key=lambda frame: abs(frame["timestamp_ms"] - timestamp_ms))


def create_app(testing=False, start_warmup=True, pipeline=None):
    setup_cache()
    app = Flask(__name__, template_folder=str(ROOT / "web_interface" / "templates"), static_folder=str(ROOT / "web_interface" / "static"))
    app.config["TESTING"] = testing
    app.config["PIPELINE"] = pipeline or VisionGuardPipeline()
    app.config["LOCK"] = threading.RLock()
    app.config["STATE_LOCK"] = threading.RLock()
    app.config["PROCESS_LOCK"] = threading.Lock()
    app.config["VIDEOS"] = {}
    app.config["JOBS"] = {}
    app.config["PROCESS_THREADS"] = {}
    app.config["ACTIVE_VIDEO_ID"] = None
    app.config["LAST_QUERY"] = ""
    OUTPUT_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if start_warmup and not testing:
        threading.Thread(target=app.config["PIPELINE"].warmup_models, daemon=True).start()

    def register_video(video_path, filename, source_url):
        path = Path(video_path).resolve()
        metadata = probe_video(path)
        video_id = f"video_{uuid4().hex}"
        job_id = f"job_{uuid4().hex}"
        job = make_job(job_id, video_id)
        chunks = make_chunks(video_id, metadata)
        update_stage(job, "upload_received", "completed", processed=1, total=1, message="Video bytes stored and readable.")
        update_stage(job, "metadata_extracted", "completed", processed=1, total=1, message="Metadata read from the stored video.")
        update_stage(job, "video_normalized", "skipped", message="Normalization is not implemented; the original MP4 is processed unchanged.")
        update_stage(job, "chunks_created", "completed", processed=len(chunks), total=len(chunks), message="Logical 30-second chunks with 5-second overlap created.")
        update_stage(job, "ocr_completed", "skipped", message="OCR is not implemented for this project.")
        update_stage(job, "captions_generated", "skipped", message="Frame caption generation is not implemented for this project.")
        video = {
            "video_id": video_id,
            "job_id": job_id,
            "filename": filename,
            "status": "processing",
            "source_url": source_url or f"/api/videos/{video_id}/content",
            "metadata": metadata,
            "chunks": chunks,
            "frames": [],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "_path": str(path),
        }
        with app.config["STATE_LOCK"]:
            app.config["VIDEOS"][video_id] = video
            app.config["JOBS"][job_id] = job
        return video, job

    def set_job_stage(job, name, status, **values):
        with app.config["STATE_LOCK"]:
            update_stage(job, name, status, **values)

    def process_video(video_id, job_id):
        video = app.config["VIDEOS"][video_id]
        job = app.config["JOBS"][job_id]
        pipe = app.config["PIPELINE"]
        try:
            with app.config["PROCESS_LOCK"]:
                with app.config["STATE_LOCK"]:
                    job["status"] = "running"
                    job["updated_at"] = utc_now()
                for name in ("frames_extracted", "keyframes_selected", "objects_detected"):
                    set_job_stage(job, name, "running", total=video["metadata"]["frame_count"])
                for event in pipe.index_video_iter(video["_path"]):
                    if event.get("kind") == "preview":
                        processed = int(event.get("frame_number", 0)) + 1
                        total = int(video["metadata"]["frame_count"])
                        kept = int(event.get("kept_frames", 0))
                        detected = int(event.get("detections", 0))
                        evidence_frame = frame_from_progress_event(video_id, event)
                        if evidence_frame is not None:
                            with app.config["STATE_LOCK"]:
                                if all(row["frame_id"] != evidence_frame["frame_id"] for row in video["frames"]):
                                    video["frames"].append(evidence_frame)
                                    video["updated_at"] = utc_now()
                        set_job_stage(job, "frames_extracted", "running", processed=processed, total=total, message=event.get("status", ""))
                        set_job_stage(job, "keyframes_selected", "running", processed=kept, total=total, message=f"{kept} useful frames selected so far.")
                        set_job_stage(job, "objects_detected", "running", processed=processed, total=total, message=f"Detector processed frame {processed}; {detected} detections on the latest selected frame.")
                    elif event.get("kind") == "done":
                        meta = event["meta"]
                        frames = materialize_frames(video_id, pipe)
                        if not frames:
                            raise RuntimeError("Processing produced no readable evidence frames.")
                        detection_count = sum(len(frame["detections"]) for frame in frames)
                        vector_count = int(meta.get("nonzero_frame_vectors", 0))
                        if vector_count < len(frames):
                            raise RuntimeError(f"Only {vector_count} of {len(frames)} evidence frames have searchable vectors.")
                        set_job_stage(job, "frames_extracted", "completed", processed=len(frames), total=len(frames), message=f"{len(frames)} real frame images written.")
                        set_job_stage(job, "keyframes_selected", "completed", processed=len(frames), total=len(frames), message="Frames selected using sampling, motion, content, and object-change rules.")
                        set_job_stage(job, "objects_detected", "completed", processed=len(frames), total=len(frames), message=f"YOLO processed {len(frames)} frames and stored {detection_count} detections.")
                        set_job_stage(job, "embeddings_generated", "completed", processed=vector_count, total=len(frames), message=f"{meta.get('embedding_mode', 'unknown')} produced nonzero vectors for every evidence frame.")
                        set_job_stage(job, "vector_index_updated", "completed", processed=len(frames), total=len(frames), message=f"Frame and segment indexes stored with backend {meta.get('retriever', 'unknown')}.")
                        set_job_stage(job, "query_ready", "completed", processed=1, total=1, message="Search is enabled because evidence frames and indexes exist.")
                        with app.config["STATE_LOCK"]:
                            video["frames"] = frames
                            video["status"] = "searchable"
                            video["updated_at"] = utc_now()
                            video["processing"] = {
                                "embedding_mode": meta.get("embedding_mode", "unknown"),
                                "indexed_windows": int(meta.get("segments", 0)),
                                "object_counts": meta.get("object_counts", {}),
                                "scan_timings": meta.get("scan_timings", {}),
                            }
                            job["status"] = "completed"
                            job["updated_at"] = utc_now()
                            app.config["ACTIVE_VIDEO_ID"] = video_id
                        return
                raise RuntimeError("Video pipeline ended without a completion event.")
        except Exception as exc:
            with app.config["STATE_LOCK"]:
                video["status"] = "failed"
                video["updated_at"] = utc_now()
                job["status"] = "failed"
                job["error"] = str(exc)
                job["updated_at"] = utc_now()
                for stage in job["stages"]:
                    if stage["status"] == "running":
                        update_stage(job, stage["name"], "failed", message=str(exc))
                    elif stage["status"] == "waiting":
                        update_stage(job, stage["name"], "skipped", message="Not run because processing failed.")

    def launch_processing(video, job):
        worker = threading.Thread(target=process_video, args=(video["video_id"], job["job_id"]), daemon=True)
        app.config["PROCESS_THREADS"][job["job_id"]] = worker
        worker.start()

    def execute_query(query, video_id=None):
        pipe = app.config["PIPELINE"]
        video = app.config["VIDEOS"].get(video_id) if video_id else None
        if video is not None:
            if video["status"] != "searchable":
                return {"ok": False, "message": "This video is not searchable yet."}, 409
            if app.config["ACTIVE_VIDEO_ID"] != video_id:
                return {"ok": False, "message": "This video is no longer the active in-memory index. Process it again before querying."}, 409
            evidence_frames = video["frames"]
        else:
            evidence_frames = []
        if not pipe.idx:
            return {"ok": False, "message": "Scan a video before searching."}, 400
        if not query:
            return {"ok": False, "message": "Enter a query before searching."}, 400
        with app.config["LOCK"]:
            hits = pipe.search(query, top_k=4)
            prepared = pipe.prepare_hits(hits, query)
            app.config["LAST_QUERY"] = query
            mode = pipe.verification_mode()
            query_plan = pipe.last_query_plan if hasattr(pipe, "last_query_plan") else None
            query_message = getattr(pipe, "last_query_message", "")
            matches = []
            frames = []
            for row in prepared:
                evidence = _evidence_frame_for_hit(evidence_frames, row) if video is not None else None
                if video is not None and evidence is None:
                    continue
                matches.append(_serialize_match(row, evidence))
                if evidence is not None and all(item["frame_id"] != evidence["frame_id"] for item in frames):
                    frames.append(public_frame(evidence))
            insufficient = not frames if video is not None else not matches
            if insufficient:
                matches = []
                frames = []
                answer = "Insufficient evidence found for this query."
                if not query_message:
                    query_message = answer
            else:
                answer = f'Found {len(frames)} evidence frame(s) for "{query}".'
                if not query_message:
                    query_message = answer
            return {
                "ok": True,
                "query": query,
                "answer": answer,
                "video_id": video_id,
                "frames": frames,
                "matches": matches,
                "insufficient_evidence": insufficient,
                "query_plan": query_plan,
                "message": query_message,
                "verification_mode": mode,
                "verification_label": _verification_label(mode),
                "warning": _verification_warning(mode),
            }, 200

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
            "embedding_mode": pipe.embedding_mode() if hasattr(pipe, "embedding_mode") else "unknown",
            "warning": _verification_warning(mode),
        })

    @app.get("/api/assets")
    def api_assets():
        return jsonify({"ok": True, "assets": _asset_rows()})

    @app.get("/api/assets/<asset_name>")
    def api_asset_video(asset_name):
        path = ROOT / "sample_videos" / Path(asset_name).name
        if not path.exists() or path.suffix.lower() != ".mp4":
            abort(404)
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.post("/api/videos/upload")
    def api_video_upload():
        target = None
        try:
            upload = request.files.get("video")
            if upload is not None:
                filename = secure_filename(upload.filename or "upload.mp4")
                if not filename.lower().endswith(".mp4"):
                    return jsonify({"ok": False, "message": "Only MP4 uploads are supported."}), 400
                target = UPLOAD_DIR / f"{uuid4().hex}_{filename}"
                upload.save(target)
                video, job = register_video(target, filename, None)
            else:
                data = request.get_json(silent=True) or {}
                sample_name = Path(str(data.get("sample", ""))).name
                sample_path = ROOT / "sample_videos" / sample_name
                if not sample_name or not sample_path.is_file() or sample_path.suffix.lower() != ".mp4":
                    return jsonify({"ok": False, "message": "Choose a valid sample asset or upload an MP4 video."}), 400
                video, job = register_video(sample_path, sample_name, f"/api/assets/{sample_name}")
        except (OSError, ValueError) as exc:
            if target is not None and target.exists():
                target.unlink()
            return jsonify({"ok": False, "message": str(exc)}), 400
        launch_processing(video, job)
        return jsonify({
            "ok": True,
            "video_id": video["video_id"],
            "job_id": job["job_id"],
            "status": "processing",
            "filename": video["filename"],
            "source_url": video["source_url"],
            **video["metadata"],
        }), 202

    @app.get("/api/videos/<video_id>")
    def api_video(video_id):
        video = app.config["VIDEOS"].get(video_id)
        if video is None:
            abort(404)
        return jsonify({
            "ok": True,
            "video_id": video["video_id"],
            "job_id": video["job_id"],
            "filename": video["filename"],
            "status": video["status"],
            "source_url": video["source_url"],
            **video["metadata"],
            "chunks": video["chunks"],
            "frame_count_indexed": len(video["frames"]),
            "processing": video.get("processing", {}),
        })

    @app.get("/api/videos/<video_id>/content")
    def api_video_content(video_id):
        video = app.config["VIDEOS"].get(video_id)
        if video is None:
            abort(404)
        path = Path(video["_path"])
        if not path.is_file():
            abort(404)
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.get("/api/videos/<video_id>/status")
    def api_video_status(video_id):
        video = app.config["VIDEOS"].get(video_id)
        if video is None:
            abort(404)
        job = app.config["JOBS"][video["job_id"]]
        return jsonify({"ok": True, **public_job(job)})

    @app.get("/api/videos/<video_id>/frames")
    def api_video_frames(video_id):
        video = app.config["VIDEOS"].get(video_id)
        if video is None:
            abort(404)
        return jsonify({
            "ok": True,
            "video_id": video_id,
            "status": video["status"],
            "frames": [public_frame(frame) for frame in video["frames"]],
        })

    @app.get("/api/videos/<video_id>/frames/<frame_id>")
    def api_video_frame(video_id, frame_id):
        video = app.config["VIDEOS"].get(video_id)
        if video is None:
            abort(404)
        frame = next((row for row in video["frames"] if row["frame_id"] == frame_id), None)
        if frame is None:
            abort(404)
        return jsonify({"ok": True, **public_frame(frame)})

    @app.get("/api/videos/<video_id>/frames/<frame_id>/image")
    def api_video_frame_image(video_id, frame_id):
        video = app.config["VIDEOS"].get(video_id)
        if video is None:
            abort(404)
        frame = next((row for row in video["frames"] if row["frame_id"] == frame_id), None)
        if frame is None:
            abort(404)
        image_path = Path(frame["_image_path"])
        if not image_path.is_file():
            abort(404)
        return send_file(image_path, mimetype="image/jpeg", conditional=True)

    @app.get("/api/jobs/<job_id>/events")
    def api_job_events(job_id):
        job = app.config["JOBS"].get(job_id)
        if job is None:
            abort(404)
        try:
            after = max(0, int(request.args.get("after", "0")))
        except ValueError:
            return jsonify({"ok": False, "message": "after must be an integer event ID."}), 400
        events = [dict(event) for event in job["events"] if event["event_id"] > after]
        return jsonify({"ok": True, "job_id": job_id, "status": job["status"], "events": events})

    @app.post("/api/videos/<video_id>/query")
    def api_video_query(video_id):
        if video_id not in app.config["VIDEOS"]:
            abort(404)
        data = request.get_json(silent=True) or {}
        try:
            payload, status = execute_query(str(data.get("query", "")).strip(), video_id)
            return jsonify(payload), status
        except Exception as exc:
            return jsonify({"ok": False, "message": str(exc)}), 500

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
                match = ROOT / "sample_videos" / sample_name
                if not match.exists() or match.suffix.lower() != ".mp4":
                    return jsonify({"ok": False, "message": "Selected sample asset was not found."}), 404
                video_path = str(match)

        if not video_path:
            return jsonify({"ok": False, "message": "Choose a sample asset or upload an MP4 video."}), 400

        try:
            if request.headers.get("Accept") == "application/x-ndjson":
                def stream_scan():
                    try:
                        with app.config["LOCK"]:
                            for event in pipe.index_video_iter(video_path):
                                if event.get("kind") == "preview":
                                    yield json.dumps({"kind": "progress", "status": event.get("status", "Scanning video…")}) + "\n"
                                elif event.get("kind") == "done":
                                    meta = event["meta"]; mode = pipe.verification_mode(); embedding_mode = pipe.embedding_mode(); warnings = [x for x in (_verification_warning(mode), _embedding_warning(embedding_mode)) if x]
                                    yield json.dumps({"kind": "done", "ok": True, "message": "Scan complete.", "video": os.path.basename(meta["video"]), "indexed_windows": int(meta.get("segments", 0)), "object_counts": meta.get("object_counts", {}), "backend": {"verification_mode": mode, "verification_label": _verification_label(mode), "embedding_mode": embedding_mode}, "warnings": warnings}) + "\n"
                    except Exception as exc:
                        yield json.dumps({"kind": "error", "message": str(exc)}) + "\n"
                return Response(stream_scan(), mimetype="application/x-ndjson")
            with app.config["LOCK"]:
                meta = None
                for event in pipe.index_video_iter(video_path):
                    if event.get("kind") == "done":
                        meta = event["meta"]
                if meta is None:
                    return jsonify({"ok": False, "message": "Scan did not complete."}), 500
                mode = pipe.verification_mode()
                embedding_mode = pipe.embedding_mode()
                warning = _verification_warning(mode)
                if warning:
                    warnings.append(warning)
                embedding_warning = _embedding_warning(embedding_mode)
                if embedding_warning:
                    warnings.append(embedding_warning)
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
                        "embedding_mode": embedding_mode,
                    },
                    "warnings": warnings,
                    "zero_query_available": bool(getattr(pipe, "zero_query", None)),
                })
        except Exception as exc:
            return jsonify({"ok": False, "message": str(exc)}), 500

    @app.get("/api/zero_query")
    def api_zero_query():
        pipe = app.config["PIPELINE"]
        if not pipe.idx:
            return jsonify({"ok": False, "message": "Scan a video first."}), 400
        zq = getattr(pipe, "zero_query", None)
        if not zq:
            return jsonify({"ok": False, "message": "Zero-query data not available. Re-scan the video."}), 400
        return jsonify({"ok": True, **zq})

    @app.post("/api/query")
    def api_query():
        data = request.get_json(silent=True) or {}
        query = str(data.get("query", "")).strip()
        try:
            active_video_id = app.config.get("ACTIVE_VIDEO_ID")
            payload, status = execute_query(query, active_video_id)
            return jsonify(payload), status
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
