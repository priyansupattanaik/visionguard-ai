# Project structure

| Path | Contents |
|---|---|
| `run.py` | Single application entry point |
| `visionguard/video_pipeline/` | Video reading, scan orchestration, vector indexing, and model-free detector evidence segmentation |
| `visionguard/model_services/` | Model adapters for detection/tracking, embeddings, verification, segmentation, clips, and reports |
| `visionguard/runtime/` | Environment loading, validated pipeline settings, logging, device selection, and cache configuration |
| `visionguard/search/` | LangGraph query orchestration, deterministic intent planning, and query rules |
| `visionguard/semantic/` | Required NVIDIA semantic segment adapter and detector-grounded event graph |
| `visionguard/web_app/` | Flask routes plus real video/job/chunk/frame state contracts |
| `web_interface/templates/` | Operational video-interface HTML |
| `web_interface/static/` | Operational CSS and JavaScript |
| `sample_videos/` | Bundled MP4 files and their detector-derived catalog |
| `configuration/` | Provider-key template, ignored local key file, and configuration guidance |
| `evaluation/` | Accuracy evaluator plus repeatable backend and Edge browser workflow verifiers |
| `documentation/` | Architecture, API, development, and evaluation documents |
| `tests/` | Automated tests for the operational scan, search, model, and web paths |
| `scripts/bootstrap_models.py` | Explicit model-cache bootstrap command; downloaded weights stay ignored |
| `.models/` | Ignored, disposable local model cache; never commit it |
| `.venv/` | Ignored project-local Python environment |

`documentation/PROJECT_FLOW.md` is the canonical end-to-end behavior contract. Update it whenever ingest, indexing, retrieval, verification, or export behavior changes.

Generated directories such as `output/`, `.cache/`, pytest caches, model caches, and Python bytecode are ignored and can be removed safely when the application is stopped.

Searchable object classes are discovered from the active detector at runtime. `query_rules.json` contains event rules and a small documented alias map. Aliases are applied only when their concrete target classes exist in the active detector's runtime labels.
