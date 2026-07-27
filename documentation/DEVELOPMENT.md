# Development and deployment

Use Python 3.11 or newer. Create `.venv` with `py -m venv .venv`, install `requirements.txt`, copy `.env.example` to `.env`, configure the mandatory NVIDIA semantic key/model, and run `.venv\Scripts\python.exe run.py`. Use `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests` for local regression verification. Run `.venv\Scripts\python.exe evaluation\verify_e2e_workflow.py --live-provider` only when real provider calls are intended.

Use `PROJECT_MAP.md` before adding or moving code. Keep dependency direction one-way: `web_app` calls `video_pipeline` and `search`; those layers call `model_services` and `runtime`; model adapters never import routes. Query rules may contain language, attributes, and implemented event terms, but never a fixed catalog of searchable objects; object names must come from the active detector or indexed metadata.

Update `PROJECT_FLOW.md` in the same change whenever ingest, indexing, retrieval, verification, or export behavior changes. Do not commit `output/`, `.cache/`, `.models/`, virtual environments, frame dumps, indexes, reports, or logs.

Model output must keep timestamps, score type, evidence state, claim provenance, and verification mode. Never convert model prose directly into a verified event. Provider confidence and vector ranking scores are not accuracy. NVIDIA semantic indexing is required; explicit verification remains separate and cannot create evidence.
