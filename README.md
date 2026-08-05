# VisionGuard

VisionGuard is an evidence-first CCTV semantic retrieval system. It turns a video into timestamped, inspectable event and scene evidence and refuses to claim an event without stored source evidence.

## What it does

The operational path is deterministic:

```text
ingest → decode every frame → exact consecutive deduplication → detect → track → evidence segments
      → embeddings/vector indexes → intent routing → detector/event/vector retrieval → explicit verification
      → timestamped frame/clip export
```

Each decoded frame is examined in source order. A frame is removed only when every pixel matches the immediately preceding decoded frame; no fixed-second sampling, motion threshold, empty-frame suppression, or forced keyframe interval is used. Each retained frame keeps its decoder timestamp, source frame number, detector boxes, confidences, track IDs, and appearance tags.

## Required semantic path

NVIDIA multimodal analysis is the required semantic stage, and readiness requires an authenticated live provider probe. Every source-time-bounded segment receives a structured description, scene tags, event tags, provider confidence, and source-frame reference. These fields remain `semantic_description`, not verified facts. Invalid or unavailable NVIDIA responses fail indexing explicitly.

YOLO and BoT-SORT provide `detector_fact` evidence. The event graph emits first-observed state, measured movement, observed dwell, and configured zone transitions as `event_fact`. Last observation at video end is not disappearance, stationary tracks are not movement, and zone transitions require observed membership changes.

## Setup

PowerShell:

```powershell
Set-Location "D:\CDAC PROJECT\visionguard-ai"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env  # Set NVIDIA_API_KEY; keep SEMANTIC_PROVIDER=nvidia
.\.venv\Scripts\python.exe scripts\bootstrap_models.py --yolo yolo11m.pt
.\.venv\Scripts\python.exe run.py
```

Open `http://127.0.0.1:7860`. Upload an MP4, review its metadata, start indexing, then search after the backend marks the evidence index query-ready.

The bootstrap command downloads a model into ignored `.models/`; it is intentionally not source-controlled. Use `yolo11n.pt` if CPU throughput matters more than small-object recall, or `yolo11m.pt` for the balanced local profile.

## Google Colab (GPU, fast profile)

For a Colab T4/A100 session with evidence-first answers (detector object hits + mandatory abstention on absent objects):

1. Runtime → **GPU**
2. Open `notebooks/visionguard_colab.ipynb` or follow `documentation/COLAB.md`
3. Store `NVIDIA_API_KEY` in Colab Secrets
4. Run:

```bash
python scripts/run_colab_e2e.py --video sample_videos/asset3.mp4 --query "find the person" --absent-query "find the elephant"
```

The Colab profile uses YOLO nano, larger `WIN_SEC`, parallel `SEMANTIC_WORKERS`, and `MODEL_PROVIDER=none`. Object answers remain detector-grounded; NVIDIA segment text stays unlabeled as verified fact. Perfect zero ML error is not possible—correct abstention on missing evidence is the anti-hallucination contract.

## Query contract

The LangGraph query brain routes object, count, numeric temporal, event, zone, semantic-scene, and explicit-verification requests. Object routes use detector observations, semantic-scene routes use the stored SigLIP segment vector index, counts use distinct supported track IDs, and numeric `before`, `after`, or `between` requests filter stored timestamps. Unsupported events abstain with a precise limitation.

Detector confidence, provider confidence, and vector similarity are different quantities and none is accuracy. Semantic descriptions remain visibly unverified. Verification is explicit: rejection or unavailability forces abstention, and ordinary searches do not invoke the external verifier.

## Evaluation

Run the unit and contract suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Run the real local workflow check:

```powershell
.\.venv\Scripts\python.exe evaluation\verify_e2e_workflow.py --live-provider
```

`--live-provider` is required because the workflow makes real NVIDIA API calls. `evaluation/ground_truth.json` remains review-gated; add human-reviewed annotations before reporting retrieval precision, temporal overlap, verification quality, or accuracy.

## Repository hygiene

Tracked content is limited to source, tests, documentation, configuration templates, evaluation schemas, and bundled sample videos. `.models/`, `.cache/`, `output/`, virtual environments, frame dumps, indexes, logs, and test caches are intentionally ignored and can be recreated.

See `documentation/PROJECT_MAP.md` for the codebase map, `documentation/PROJECT_FLOW.md` for the canonical operational flow, `documentation/ARCHITECTURE.md` for boundaries, and `documentation/API.md` for the HTTP contract.
