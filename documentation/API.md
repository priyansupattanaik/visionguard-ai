# API

Run `python -m backend.main`. `GET /health` returns service state. `POST /v1/videos` accepts `{"path":"video-path"}` and returns the immutable video plus a processing job. `GET /v1/jobs/{job_id}` reports real worker stage and progress. `POST /v1/videos/{video_id}/query` accepts `{"query":"..."}` and returns confidence, citations, uncertainty, and verification state.

Path-based ingestion is for trusted local deployments. Public deployments must use bounded uploads, MIME sniffing, quarantine storage, authentication, and authorization.
