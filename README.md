# Vision Guard

Vision Guard is a local, single-process CCTV video search app. It scans a video once, indexes sampled frame and segment embeddings, and lets you search for object-focused events such as `person`, `white car`, `umbrella`, or `backpack`.

The main local UI is now a Flask app with Jinja templates, static CSS, and vanilla JavaScript. The legacy Gradio app remains in `app.py`, but it is no longer the documented main entry point.

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

The Flask routes are:

- `GET /`
- `GET /api/status`
- `GET /api/assets`
- `POST /api/scan`
- `POST /api/query`
- `POST /api/export`
- `GET /api/download/<id>`

## Local Workflow

1. Start the Flask app.
2. Choose a sample video or upload an MP4.
3. Click `Scan video`.
4. Enter an object-focused query.
5. Review result timestamps, windows, objects, scores, and verification status.
6. Select matches and click `Export selected`.
7. Download generated ZIP, HTML, CSV, or JSON files if export succeeds.

## Sample Assets

The tracked sample videos are:

- `assets/asset1.mp4`
- `assets/asset2.mp4`
- `assets/asset3.mp4`
- `assets/asset4.mp4`
- `assets/asset5.mp4`
- `assets/asset6.mp4`
- `assets/asset7.mp4`
- `assets/asset8.mp4`

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
- Legacy UI: Gradio in `app.py`
- Video access: Decord with OpenCV fallback
- Detection: Ultralytics YOLO, default `yolo11m.pt`
- Retrieval: `google/siglip2-so400m-patch14-384`
- Visual verification: `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` when available
- Segmentation: `facebook/sam2.1-hiera-small` during export when available
- Reports: JSON, CSV, HTML, ZIP under `output/`

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall .
.\.venv\Scripts\python.exe -m pytest -q
```

## Visual QA

Start the Flask app first, then run:

```powershell
.\.venv\Scripts\python.exe scripts\visual_qa.py
```

Screenshots are written under `qa_screenshots/`. The script requires Playwright and a browser available in the local environment.

## Outputs

Each scan creates a timestamped run directory under `output/` containing:

- `frames/`
- `clips/`
- `reports/`
- `segments/`

Uploads are stored under `output/uploads/`.

## Troubleshooting

- If dependency install cannot reach PyPI, rerun the same pip command with network access.
- If YOLO or Hugging Face model downloads fail, place the required model files in the expected local cache or run with network access.
- If `/api/query` says to scan first, run `/api/scan` or use the UI scan button before searching.
- If export returns raw fallback, segmentation did not complete in the bounded export window.
- If Windows CPU shows detector-only dev passthrough, Qwen did not run.

## Documentation

- Full technical manual: `PROJECT_DOCUMENTATION.md`
- Pipeline code explanation: `PIPELINE_CODE_EXPLANATION.md`
- Colab launcher notebook: `VisionGuard_Colab.ipynb`
