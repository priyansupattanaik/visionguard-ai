# VisionGuard

VisionGuard is a local-first CCTV evidence retrieval system. It turns a video into timestamped, inspectable evidence and refuses to present a retrieval candidate as a verified fact.

## What it does

The operational path is deterministic:

```text
ingest → decode/sample → deduplicate → detect → track → evidence segments
      → embeddings/indexes → intent routing → retrieval → optional verification
      → timestamped frame/clip export
```

Each indexed frame retains its decoder timestamp, source frame number, detector boxes, confidences, track IDs, appearance tags, and selection reason. Retrieval produces an evidence interval with a representative source frame. An answer is generated only from these stored records.

## Modes

The default local mode uses YOLO for object evidence, tracking, calibrated detector-segment retrieval, and deterministic query aliases such as `vehicles` and `pedestrians`. It works without a language model.

If SigLIP weights are installed locally, VisionGuard also performs text-to-frame semantic retrieval. If a separately configured verifier is reachable, it can check bounded visual candidates. Those external/optional modes are clearly labelled in the UI. Without them, unsupported open-ended requests abstain instead of claiming a result.

## Setup

PowerShell:

```powershell
Set-Location "D:\CDAC PROJECT\visionguard-ai"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe scripts\bootstrap_models.py --yolo yolo11m.pt
.\.venv\Scripts\python.exe run.py
```

Open `http://127.0.0.1:7860`. Upload an MP4, review its metadata, start indexing, then search after the backend marks the evidence index query-ready.

The bootstrap command downloads a model into ignored `.models/`; it is intentionally not source-controlled. Use `yolo11n.pt` if CPU throughput matters more than small-object recall, or `yolo11m.pt` for the balanced local profile.

## Query contract

Exact detector classes and documented aliases use detector evidence. A returned result contains the evidence interval, timestamp, stored frame, and detected classes. `answer` mode returns text with exact timestamps; `frames` mode returns only evidence frames; `both` returns both.

Detector confidence is not accuracy. Evidence below `MIN_EVIDENCE_CONFIDENCE` is excluded from detector-segment retrieval. Semantic or verification candidates remain unverified until an available verifier confirms them.

## Evaluation

Run the unit and contract suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Run the real local workflow check:

```powershell
.\.venv\Scripts\python.exe evaluation\verify_e2e_workflow.py
```

`evaluation/ground_truth.json` is a review-gated schema. Add human-reviewed video, segment, and query annotations before reporting retrieval precision, temporal overlap, verification quality, or accuracy. Generated reports belong in `output/evaluation/` and are ignored by Git.

## Repository hygiene

Tracked content is limited to source, tests, documentation, configuration templates, evaluation schemas, and bundled sample videos. `.models/`, `.cache/`, `output/`, virtual environments, frame dumps, indexes, logs, and test caches are intentionally ignored and can be recreated.

See `documentation/ARCHITECTURE.md`, `documentation/API.md`, and `evaluation/README.md` for the detailed contracts and limitations.
