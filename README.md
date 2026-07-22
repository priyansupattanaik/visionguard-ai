# Vision Guard

Vision Guard is a local, single-process CCTV video search app with **Hierarchical Object-Centric Video RAG (HOC-VideoRAG)** architecture. It scans a video once, runs real multi-object tracking (BoT-SORT), indexes sampled frame, segment, and object-crop embeddings, and lets you search for objects, temporal events, or browse an automatic zero-query analysis.

The main local UI is a Flask app with Jinja templates, static CSS, and vanilla JavaScript. The legacy Gradio app is preserved in `legacy/app_gradio.py`.

## Features

- **Real Multi-Object Tracking**: BoT-SORT with persistent track IDs, trajectory statistics, dwell time, entry/exit detection.
- **Object-Crop Embeddings**: High-confidence detections are cropped and embedded via SigLIP2 for fine-grained retrieval.
- **Zero-Query Mode**: Auto-generated object inventory, event timeline, and anomaly detection after every scan — no query needed.
- **Controlled Temporal Queries**: Supports "loitering", "entering", "exiting", "approaching", "gathering" via track trajectory features.
- **Hierarchical Retrieval**: Tracks → segments → frames → dense reselection.
- **Honest Verification**: Labels clearly whether Qwen VLM actually ran or if results are detector-only.

## Local Setup

```powershell
cd "D:\CDAC PROJECT\visionguard-ai"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run The Flask App

```powershell
.\.venv\Scripts\python.exe flask_app.py
```

Generic Python environments can run the same entry point with `python flask_app.py`.

Open `http://127.0.0.1:7860`.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main UI |
| `/api/status` | GET | System/model status |
| `/api/assets` | GET | List sample videos |
| `/api/scan` | POST | Scan and index a video |
| `/api/zero_query` | GET | **NEW** — Auto-generated analysis (inventory + timeline) |
| `/api/query` | POST | Search indexed video |
| `/api/export` | POST | Export selected matches |
| `/api/download/<id>` | GET | Download exported file |

## Local Workflow

1. Start the Flask app.
2. Choose a sample video or upload an MP4.
3. Click `Scan video`.
4. **NEW**: Click `Show Analysis` in the Auto Analysis panel for zero-query insights (object inventory, event timeline).
5. Enter an object-focused or temporal query (e.g., "person", "white car", "loitering", "person entering").
6. Review result timestamps, windows, objects, tracks, scores, and verification status.
7. Select matches and click `Export selected`.
8. Download generated ZIP, HTML, CSV, or JSON files if export succeeds.

## Tracking & Zero-Query

After scanning, VisionGuard automatically computes:
- **Object Inventory**: Per-class track counts, dwell times, entry/exit timestamps.
- **Event Timeline**: High-motion segments, long-dwell anomalies, sudden appearances.
- **Natural-Language Summary**: Quick overview of the video content.

These are accessible via the UI ("Auto Analysis" panel) or `GET /api/zero_query`.

## Sample Assets

The tracked sample videos are in `assets/`:
- `asset1.mp4`, `asset2.mp4`, `asset3.mp4`, `asset4.mp4`
- `asset5.mp4`, `asset6.mp4`, `asset7.mp4`, `asset8.mp4`

## Verification Honesty

Vision Guard uses Qwen visual verification only when the Qwen backend actually loads and runs. On Windows CPU, `qwen_verifier.py` uses a development passthrough because Qwen is skipped. The Flask UI labels that mode as:

`Detector-only dev passthrough: Qwen skipped on Windows CPU`

Those results are detector/retrieval matches, not real Qwen-verified matches.

## Export Behavior

Export first tries the existing segmentation path. If SAM2 segmentation is unavailable, fails, or times out, the backend creates an honest raw-clip fallback when it can. The UI and API label this as:

`Raw clip export fallback; segmentation unavailable or timed out.`

The app must not silently pretend segmentation worked.

## Runtime Stack

- UI: Flask, Jinja, static CSS, vanilla JavaScript
- Legacy UI: Gradio in `legacy/app_gradio.py`
- Video access: Decord with OpenCV fallback
- Detection & Tracking: Ultralytics YOLO11m + BoT-SORT with `persist=True`
- Retrieval: `google/siglip2-so400m-patch14-384` (frame + crop embeddings)
- Visual verification: `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` when available
- Segmentation: `facebook/sam2.1-hiera-small` during export when available
- Vector index: turbovec (with numpy fallback)
- Reports: JSON, CSV, HTML, ZIP under `output/`

## Tests

```powershell
.\.venv\Scripts\python.exe -m compileall .
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover:
- Tracking produces non-empty track_ids
- Zero-query generates valid inventory + timeline
- Existing object queries still work
- Export still produces clips
- Flask routes and UI integrity

## Outputs

Each scan creates a timestamped run directory under `output/` containing:

- `frames/` — sampled keyframes
- `clips/` — extracted/segmented clips
- `reports/` — index.json, zero_query.json, frame/segment/crop indices
- `segments/` — segmentation masks

Uploads are stored under `output/uploads/`.

## Troubleshooting

- If dependency install cannot reach PyPI, rerun the same pip command with network access.
- If YOLO or Hugging Face model downloads fail, place the required model files in the expected local cache or run with network access.
- If `/api/query` says to scan first, run `/api/scan` or use the UI scan button before searching.
- If export returns raw fallback, segmentation did not complete in the bounded export window.
- If Windows CPU shows detector-only dev passthrough, Qwen did not run.

## Documentation

- Full technical manual: `PROJECT_DOCUMENTATION.md`
- Changelog: `CHANGELOG.md`
- Legacy Gradio app: `legacy/app_gradio.py`
- Colab launcher notebook: `VisionGuard_Colab.ipynb`
