# VisionGuard

VisionGuard is an offline-first video intelligence application. It scans a video once, detects and tracks objects, builds searchable indexes, and returns timestamped evidence. When semantic models or hosted verification are unavailable, the application identifies the degraded mode instead of presenting unverified output as confirmed.

## Initialize

Run these commands one at a time from the project root:

```bat
cd /d "D:\CDAC PROJECT\visionguard-ai"
py -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
copy configuration\provider_keys.env.example configuration\provider_keys.env
```

Optional credentials belong in `configuration/provider_keys.env`. NVIDIA visual verification uses `NVIDIA_API_KEY`; gated Hugging Face downloads use `HF_TOKEN`. The credential file is ignored by Git.

## Start

Start the operational video interface:

```bat
.venv\Scripts\python.exe run.py
```

Open `http://127.0.0.1:7860`, select a bundled video or upload an MP4, and start processing. The query input remains disabled until the backend has written real evidence frames and completed the searchable index.

## Query and indexing behavior

The live query path uses runtime capability discovery, hybrid retrieval, and evidence verification. Object names are read from the active detector model; the application does not maintain an object whitelist or object-alias table. Replacing YOLO with a custom detector therefore exposes that detector's class names without editing query code.

When the configured SigLIP model is cached, frames use open visual-semantic queries. When it is not cached, VisionGuard builds nonzero metadata embeddings from every detector class, colors, appearances, and motion instead of zero vectors. If NVIDIA verification is configured, an otherwise open query is checked against a bounded sample of indexed frames. If neither semantic embeddings nor NVIDIA verification is available, only exact runtime detector classes and indexed metadata can be searched; the API reports that limitation explicitly.

## Test

```bat
set PYTHONDONTWRITEBYTECODE=1
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
.venv\Scripts\python.exe -m pip check
```

## Evaluate sample videos

```bat
.venv\Scripts\python.exe evaluation\evaluate_sample_videos.py
.venv\Scripts\python.exe evaluation\verify_e2e_workflow.py
```

`verify_e2e_workflow.py` performs a real multipart upload, waits for the real job stages, checks every frame image route and deterministic timestamp, runs a matching query, and confirms that an absent-object query returns insufficient evidence. `verify_browser_workflow.py` additionally drives installed Microsoft Edge through the complete UI; it requires the verification-only `websocket-client` package.

Detector confidence and frame coverage are not accuracy. Precision, recall, and F1 are calculated only after reviewed labels are added to `evaluation/ground_truth.json`.

See `PROJECT_STRUCTURE.md` for every retained folder and `documentation/` for architecture, API, development, and evaluation details.
