# VisionGuard – Project Documentation

## Architecture Overview

VisionGuard implements the **Hierarchical Object-Centric Video RAG (HOC-VideoRAG)** architecture for CCTV video search and analysis.

### Data Flow
```
Video → Decord Reader → Frame Sampling → BoT-SORT Tracking → SigLIP2 Embedding → Vector Indexing
                                              ↓                      ↓
                                         Track Stats           Crop Embeddings
                                              ↓                      ↓
                                        Zero-Query              Crop Index
                                        Analysis
```

### Retrieval Hierarchy
```
User Query → Temporal Track Hits (if temporal terms)
           → Detector Hits (YOLO class matching)
           → Semantic Frame Hits (SigLIP2 cosine similarity)
           → Object Fallback Hits (metadata matching)
           → Weak Semantic Hits
           → Segment Hits
           → VLM Verification (Qwen2.5-VL)
```

## Environment Variables (.env Support)

VisionGuard automatically loads environment variables from a `.env` file in the project root directory via `cache_utils.setup_cache()`.

| Variable | Default | Description |
|---|---|---|
| `VISION_GUARD_HOST` | `127.0.0.1` | Host address for Flask web application |
| `VISION_GUARD_PORT` | `7860` | Listening port for Flask web application |
| `VISION_GUARD_SKIP_WARMUP` | `0` | Set to `1` to disable async model pre-warming |
| `VISION_GUARD_OUT_DIR` | `output` | Output directory for indexed frames, clips, and reports |
| `VISION_GUARD_DEVICE` | *(auto)* | Force target PyTorch device (`cuda` or `cpu`) |
| `VISION_GUARD_VERIFIER_BACKEND` | *(auto)* | Set to `dev_passthrough` or `real` to override Qwen mode |
| `YOLO_MODEL` | `yolo11m.pt` | Path or name of Ultralytics YOLO model weight file |
| `YOLO_CONF` | `0.22` | YOLO detection confidence threshold |
| `YOLO_IMGSZ` | `640` | YOLO image input inference size |
| `YOLO_TRACKER` | `botsort.yaml` | Ultralytics tracking YAML configuration file |
| `CLIP_MODEL` | `google/siglip2-so400m-patch14-384` | Hugging Face model identifier for search encoder |
| `VERIFIER_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | Hugging Face model identifier for visual verifier |
| `SAM_MODEL` | `facebook/sam2.1-hiera-small` | Hugging Face model identifier for SAM2 segmenter |
| `SAMPLE_SEC` | `0.75` | Keyframe sampling step interval in seconds |
| `WIN_SEC` | `4.5` | Temporal window segment length in seconds |
| `IMAGE_BATCH_SIZE` | *(auto)* | Override SigLIP2 image batch inference size |
| `HF_TOKEN` | *(none)* | Hugging Face access token for gated models |

## File Structure

```
visionguard-ai/
├── flask_app.py          # Main entry point (Flask web app)
├── pipeline.py           # Core orchestrator: indexing, search, export
├── tracker.py            # YOLO11m + BoT-SORT tracking
├── vlm.py                # SigLIP2 search encoder
├── qwen_verifier.py      # Qwen2.5-VL visual verification
├── vector_index.py       # Vector index (turbovec / numpy fallback)
├── video_reader.py       # Decord video reader
├── clip_generator.py     # Clip extraction
├── segmenter.py          # SAM2 grounded segmentation
├── report_generator.py   # JSON, CSV, HTML, ZIP report generation
├── cache_utils.py        # Cache directory setup
├── templates/
│   └── index.html        # Flask UI template
├── static/
│   ├── css/app.css       # UI styles
│   └── js/app.js         # Frontend JavaScript
├── tests/
│   ├── test_web_app.py           # Flask route tests
│   ├── test_pipeline_export.py   # Export fallback tests
│   └── test_hoc_features.py      # HOC-VideoRAG feature tests
├── assets/               # Sample CCTV videos
├── legacy/
│   └── app_gradio.py     # Legacy Gradio UI (preserved)
├── optional_integrations/ # Experimental integrations
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## Data Contracts

### Frame Row Schema
```json
{
  "frame_id": 0,
  "frame": 0,
  "ts": 1.5,
  "frame_path": "output/.../frames/f_000025.jpg",
  "representative_frame_path": "output/.../frames/f_000025.jpg",
  "objects": ["person", "car"],
  "appearances": ["white car"],
  "tracks": [1, 3, 5],
  "detections": [
    {
      "box": [100.0, 200.0, 300.0, 400.0],
      "conf": 0.85,
      "cls": 0,
      "name": "person",
      "color": null,
      "track_id": 1
    }
  ],
  "motion_score": 0.045,
  "keep_reason": "motion",
  "still_people": 0,
  "object_delta": 1
}
```

### Track Stats Schema
```json
{
  "track_id": 1,
  "class_name": "person",
  "class_id": 0,
  "trajectory_length": 12,
  "dwell_time": 8.5,
  "entry_frame": 25,
  "exit_frame": 237,
  "entry_ts": 1.0,
  "exit_ts": 9.5,
  "avg_confidence": 0.83,
  "boxes": [[100, 200, 300, 400], ...]
}
```

### Zero-Query Response Schema
```json
{
  "ok": true,
  "object_inventory": {
    "person": {
      "count": 3,
      "total_dwell_time": 18.5,
      "avg_dwell_time": 6.17,
      "tracks": [...]
    }
  },
  "event_timeline": [
    {
      "type": "high_motion",
      "start": 2.0,
      "end": 5.5,
      "max_motion": 0.12,
      "objects": ["person", "car"],
      "tracks": [1, 2]
    },
    {
      "type": "long_dwell",
      "track_id": 1,
      "class": "person",
      "dwell_time": 8.5,
      "entry_ts": 1.0,
      "exit_ts": 9.5
    }
  ],
  "summary": "10.0s video with 5 tracked objects | detected classes: car, person | 3 notable events detected",
  "meta": {
    "total_tracks": 5,
    "classes": ["car", "person"],
    "duration": 10.0,
    "fps": 25.0
  }
}
```

## Query Categories

| Category | Example | Handling |
|---|---|---|
| Object queries | "person", "white car" | Detector + semantic + VLM verification |
| Temporal queries | "loitering", "person entering" | Track trajectory features + VLM verification |
| Hard-rejected events | "fight", "accident", "violence" | Rejected (no reliable detection method) |
| Color queries | "red car", "blue truck" | HSV color estimation + detector matching |

## API Endpoints

### GET /api/zero_query
Returns automatic analysis computed during the last scan.

**Response**: Object inventory, event timeline, summary (see schema above).

**Error**: 400 if no video has been scanned.

### POST /api/scan
Scans and indexes a video. Now includes `zero_query_available: true` in response.

### POST /api/query
Searches indexed video. Response now includes `tracks` field in each match.

## Vector Indices

Three vector indices are built during indexing:
1. **Frame index** (`frame_index.tvim`): Full-frame SigLIP2 embeddings
2. **Segment index** (`segment_index.tvim`): Averaged segment embeddings
3. **Crop index** (`crop_index.tvim`): Object-crop SigLIP2 embeddings (for high-confidence detections ≥ 0.35)

## Future TODOs

- **Qdrant integration**: Replace in-memory numpy/turbovec with Qdrant for production-scale vector search.
- **Full event detection**: Train or integrate a dedicated event detection model for fight/accident/violence.
- **Multi-camera support**: Correlate tracks across multiple camera feeds with re-identification.
- **Streaming mode**: Real-time tracking and indexing for live CCTV feeds.
- **Crop-level retrieval**: Use crop embeddings in the primary search path (currently indexed but retrieval path prioritizes frame-level).
- **Track re-identification**: Cross-segment track merging for long videos where BoT-SORT IDs may reset.
