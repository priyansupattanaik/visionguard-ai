# Project structure

| Path | Contents |
|---|---|
| `run.py` | Single application entry point |
| `visionguard/video_pipeline/` | Video reading, scan orchestration, vector indexing, and retrieval |
| `visionguard/model_services/` | Detection, tracking, visual embeddings, meaningful metadata fallback embeddings, segmentation, verification, and clip/report services |
| `visionguard/runtime/` | Environment loading, device selection, and cache configuration |
| `visionguard/web_app/` | Operational Flask API and UI server |
| `visionguard/evidence_api/agents/` | Query planner, retriever, verifier, reasoner, and grounded responder agents |
| `visionguard/evidence_api/` | Typed evidence schemas, database, hybrid retrieval, workers, and FastAPI application |
| `web_interface/templates/` | Operational video-interface HTML |
| `web_interface/static/` | Operational CSS and JavaScript |
| `web_interface/evidence_console.html` | Evidence API console |
| `sample_videos/` | Bundled MP4 files and their detector-derived catalog |
| `configuration/` | Provider-key template, ignored local key file, and configuration guidance |
| `evaluation/` | Sample-video evaluator, ground-truth format, and latest measured report |
| `documentation/` | Architecture, API, data model, development, deployment, and evaluation documents |
| `tests/` | Operational and evidence-backend automated tests |
| `.models/` | Ignored local model weights retained to prevent repeat downloads |
| `.venv/` | Ignored project-local Python environment |

Generated directories such as `output/`, `.visionguard/`, `.cache/`, `.yolo/`, pytest caches, and Python bytecode are ignored and can be removed safely when the application is stopped.

The operational Flask application reuses the evidence query planner. Detector vocabulary and aliases live in `visionguard/evidence_api/resources/query_vocabulary.json`; they are not duplicated inside the indexing loop.
