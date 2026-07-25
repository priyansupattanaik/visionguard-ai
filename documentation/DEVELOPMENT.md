# Development and deployment

Use Python 3.11 or newer. Create `.venv` with `py -m venv .venv`, install `requirements.txt` with `.venv\Scripts\python.exe -m pip install -r requirements.txt`, copy `.env.example` to `.env`, and run `.venv\Scripts\python.exe run.py`. Use `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests` for verification.

Keep runtime code inside the four responsibility-focused packages: `video_pipeline`, `model_services`, `search`, and `web_app`. Query rules may contain language, attributes, and implemented event terms, but never a fixed catalog of searchable objects; object names must come from the active detector or indexed metadata.

Model output must keep timestamps, confidence, and its verification mode. Never convert model prose directly into a verified event. Hosted inference remains optional, and failure must produce an explicit limitation instead of fabricated evidence.
