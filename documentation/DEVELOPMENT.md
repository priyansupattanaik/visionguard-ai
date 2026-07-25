# Development and deployment

Use Python 3.11 or newer. Create `.venv` with `py -m venv .venv`, install `requirements.txt` with `.venv\Scripts\python.exe -m pip install -r requirements.txt`, copy `.env.example` to `.env`, and run `.venv\Scripts\python.exe run.py`. Use `.venv\Scripts\python.exe -m pytest -q tests` for verification. The evidence console is served by the API from `web_interface/evidence_console.html`.

Model adapters implement `analyze(video, artifacts)` and return `Evidence` dictionaries. Record a stable producer version, calibrated confidence, exact timestamps, and entity identifiers. Never convert model prose directly into a verified event; add deterministic validation or an independent verifier.

Production deployments should separate API and workers, use a durable queue and storage, enable backups, and emit structured logs with job, video, stage, evidence, and correlation IDs. Hosted inference providers remain optional.
