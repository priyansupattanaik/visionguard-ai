# API

Run `.\.venv\Scripts\python.exe run.py`. The same Flask application serves the browser interface and these endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | Model readiness and current index status |
| GET | `/api/assets` | Bundled sample videos |
| GET | `/api/assets/<name>` | Stream a bundled sample video |
| GET | `/api/model/health` | Selected provider plus text, vision, NVIDIA, and Groq capability state |
| POST | `/api/videos/upload` | Store and validate an MP4, then create an uploaded `video_id` and `job_id` |
| POST | `/api/videos/<video_id>/index` | Start the explicit decode-to-index job for an uploaded video |
| GET | `/api/videos/<video_id>` | Real video metadata, logical chunks, and processing summary |
| DELETE | `/api/videos/<video_id>` | Remove a non-running video's retained upload, job state, and generated run directory |
| GET | `/api/videos/<video_id>/content` | Stream the immutable stored upload |
| GET | `/api/videos/<video_id>/status` | Truthful job and stage states |
| GET | `/api/videos/<video_id>/frames` | Extracted frame records with deterministic timestamps |
| GET | `/api/videos/<video_id>/frames/<frame_id>` | One evidence-frame record and real detections |
| GET | `/api/videos/<video_id>/frames/<frame_id>/image` | The stored JPEG evidence image |
| POST | `/api/videos/<video_id>/query` | Evidence-gated search for one query-ready video |
| GET | `/api/jobs/<job_id>/events` | Incremental backend job events; use `?after=<event_id>` for polling |
| POST | `/api/scan` | Scan a path or bundled asset; supports NDJSON progress streaming |
| GET | `/api/zero_query` | Detector-derived index summary |
| POST | `/api/query` | Search the active index with a natural-language query |
| POST | `/api/export` | Create a report or video clip from selected evidence |
| GET | `/api/download/<id>` | Download a registered export |

The API enforces byte, duration, resolution, registered-video, mutation-rate, and download-token lifetime limits. Mutating requests from non-loopback clients require `Authorization: Bearer <VISION_GUARD_API_TOKEN>`. This remains a local service; durable multi-user job persistence is not implemented.

OCR, per-frame captioning, and video normalization are not implemented and are reported as `skipped`. Required segment semantic analysis has a separate live stage. Query responses contain frames only when the referenced JPEG exists; unsupported or insufficient requests abstain.
