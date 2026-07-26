# Project Map

VisionGuard is organized by dependency direction. Browser and HTTP code call application orchestration; orchestration calls deterministic pipeline and search components; those components call model adapters and runtime utilities. Model adapters never import web routes.

| Area | Responsibility | Key files |
|---|---|---|
| Entry point | Load local configuration, configure logging, start the Flask app | `run.py` |
| Web/API | HTTP routes, upload/job state, frame serialization, download registration | `visionguard/web_app/server.py`, `visionguard/web_app/video_jobs.py` |
| Pipeline | Decode every frame, exact deduplication, indexing, evidence lifecycle, export coordination | `visionguard/video_pipeline/video_pipeline.py` |
| Pipeline helpers | Video decoding, exact deduplication, vector index, detector-segment retrieval | `visionguard/video_pipeline/*.py` |
| Model adapters | YOLO/BoT-SORT, SigLIP, optional verification, segmentation, clip/report output | `visionguard/model_services/*.py` |
| Search | Deterministic query normalization, runtime label resolution, alias/event rules | `visionguard/search/query_planner.py`, `visionguard/search/query_rules.json` |
| Runtime | `.env` loading, validated settings, cache locations, device selection, logging | `visionguard/runtime/*.py` |
| UI | HTML template and browser JavaScript/CSS served by Flask | `web_interface/` |
| Evaluation | Ground-truth schema and repeatable evaluator/workflow checks | `evaluation/` |
| Tests | Unit and HTTP contract coverage for the current architecture | `tests/` |
| Local setup | Explicit local model bootstrap only | `scripts/bootstrap_models.py` |

## Generated paths

The application may create `output/`, `.cache/`, `.models/`, `.pytest_cache/`, Python bytecode, and local logs. They are intentionally ignored and are never source-of-truth inputs. A new index run recreates its evidence output from the original video.

## Debug order

1. Confirm configuration and model readiness with `GET /api/status` or `GET /api/model/health`.
2. Confirm upload metadata and job stages with `GET /api/videos/<video_id>/status`.
3. Inspect indexed source frames and timestamps through the frame API.
4. Inspect query plan, retrieval mode, verification mode, and citations in the query response.
5. Inspect generated export files only after their evidence rows are confirmed.
