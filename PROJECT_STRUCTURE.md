# Project structure

| Path | Contents |
|---|---|
| `run.py` | Single application entry point |
| `visionguard/video_pipeline/` | Video reading, scan orchestration, vector indexing, and retrieval |
| `visionguard/model_services/` | Detection, tracking, visual embeddings, meaningful metadata fallback embeddings, segmentation, verification, and clip/report services |
| `visionguard/runtime/` | Environment loading, device selection, and cache configuration |
| `visionguard/search/` | Detector-aware query planning and language/event rules; no object catalog |
| `visionguard/web_app/` | Flask routes plus real video/job/chunk/frame state contracts |
| `web_interface/templates/` | Operational video-interface HTML |
| `web_interface/static/` | Operational CSS and JavaScript |
| `sample_videos/` | Bundled MP4 files and their detector-derived catalog |
| `configuration/` | Provider-key template, ignored local key file, and configuration guidance |
| `evaluation/` | Accuracy evaluator plus repeatable backend and Edge browser workflow verifiers |
| `documentation/` | Architecture, API, development, and evaluation documents |
| `tests/` | Automated tests for the operational scan, search, model, and web paths |
| `.models/` | Ignored local model weights retained to prevent repeat downloads |
| `.venv/` | Ignored project-local Python environment |

Generated directories such as `output/`, `.cache/`, `.yolo/`, pytest caches, and Python bytecode are ignored and can be removed safely when the application is stopped. `.models/` is also generated, but retaining it prevents model downloads from repeating.

Searchable object classes are discovered from the active detector at runtime. `query_rules.json` contains language and supported event-rule terms only; it contains no application object catalog or object-alias table.
