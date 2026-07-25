# VisionGuard

VisionGuard is an offline-first video intelligence application. It scans a video once, detects and tracks objects, builds searchable indexes, and returns timestamped evidence. When semantic models or hosted verification are unavailable, the application identifies the degraded mode instead of presenting unverified output as confirmed.

## Initialize

Run these commands one at a time from the project root:

```powershell
cd "D:\CDAC PROJECT\visionguard-ai"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
copy configuration\provider_keys.env.example configuration\provider_keys.env
```

Optional credentials belong in `configuration/provider_keys.env`. NVIDIA visual verification uses `NVIDIA_API_KEY`; gated Hugging Face downloads use `HF_TOKEN`. The credential file is ignored by Git.

## Start

Start the operational video interface:

```powershell
.\.venv\Scripts\python.exe run.py
```

Open `http://127.0.0.1:7860`, select a video from `sample_videos`, scan it, and enter a query such as `person` or `red car`.

## Query and indexing behavior

The live query path uses a deterministic planner agent, hybrid retrieval, and evidence verification. The planner maps natural aliases to detector classes, routes supported entry/exit/loitering questions to track evidence, requests clarification for incomplete queries, and refuses to claim unsupported action or speech results.

When the configured SigLIP model is cached, frames use visual-semantic embeddings. When it is not cached, VisionGuard builds nonzero metadata embeddings from detected objects, colors, appearances, and motion instead of zero vectors. Metadata mode can search only facts present in detector/tracker output; it is not a replacement for a vision-language model.

To start the evidence API instead:

```powershell
$env:VISIONGUARD_APP="api"
& ".\.venv\Scripts\python.exe" run.py
```

Open `http://127.0.0.1:8000`, `/health`, or `/docs`.

## Test

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
.\.venv\Scripts\python.exe -m pip check
```

## Evaluate sample videos

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate_sample_videos.py
```

Detector confidence and frame coverage are not accuracy. Precision, recall, and F1 are calculated only after reviewed labels are added to `evaluation/ground_truth.json`.

See `PROJECT_STRUCTURE.md` for every retained folder and `documentation/` for architecture, API, data-model, development, deployment, and evaluation details.
