# Changelog

> Path names below are historical. See `PROJECT_STRUCTURE.md` for the current repository layout.

All notable changes to VisionGuard are documented in this file.

## [0.2.0] - 2026-07-22 — HOC-VideoRAG Upgrade

### Added
- **Real multi-object tracking**: BoT-SORT tracking activated with persistent track IDs across frames. Each detection now carries a `track_id`.
- **Track statistics**: Per-track metrics computed automatically: trajectory length, dwell time, entry/exit timestamps, average confidence.
- **Object-crop embeddings**: High-confidence detections (≥0.35) are cropped and embedded via SigLIP2 for fine-grained retrieval.
- **Zero-query mode**: Automatic analysis generated after every scan:
  - Object inventory (unique tracks per class with counts and dwell stats)
  - Event timeline (high-motion windows, long-dwell anomalies, sudden appearances)
  - Natural-language summary
  - New API endpoint `GET /api/zero_query`
  - UI panel with "Show Analysis" button
- **Controlled temporal queries**: Queries using terms like "loitering", "enter", "exit", "approach", "gather" now work via track trajectory features instead of being rejected.
- **New `_temporal_track_hits` method**: Routes temporal queries through track data with VLM verification on candidate windows.
- **New `_is_temporal_query` method**: Distinguishes soft temporal terms from hard-rejected abstract events.
- **Crop vector index**: `crop_idx` stores object-crop embeddings alongside frame and segment indices.
- **Tests**: `tests/test_hoc_features.py` with 11 focused tests for tracking, zero-query, and backward compatibility.

### Changed
- `tracker.py`: Added `track_frame()` for per-frame tracking and `compute_track_stats()` for aggregation.
- `pipeline.py`: Indexing loop now uses `track_frame()` instead of `detect_batch()`. Track IDs flow into frame metadata, segment metadata, and index JSON.
- `flask_app.py`: Added `/api/zero_query` endpoint, `zero_query_available` in scan response, `tracks` in serialized matches.
- `_is_event_query`: Now only rejects hard abstract events (fight, accident, violence). Temporal terms moved to `_is_temporal_query`.
- `requirements.txt`: Removed `gradio` from required deps (moved to optional comment), removed `playwright`.
- `.gitignore`: Added `.pytest_cache/`, `*.pyc`, `.agents/`.

### Moved
- `app.py` → `legacy/app_gradio.py` (legacy Gradio interface preserved for reference).

### UI
- Added zero-query analysis panel to Flask UI (auto-shown after scan).
- Added CSS styles for inventory tables, event timeline, and type badges.
- Updated `app.js` with zero-query fetch and render logic.
