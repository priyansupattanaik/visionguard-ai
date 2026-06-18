# Vision Guard Context Snapshot

This file is a handoff/context snapshot for optional Headroom-based compression or downstream agent consumption.
It is not imported by the Vision Guard runtime.
It exists only to capture what has been done in the repository up to this point.

## Purpose

Use this file when you want to:

- compress project context with Headroom
- hand off the current repository state to another agent
- quickly review what has already been changed
- understand the current Colab run path

## Project Identity

- Project name in UI and docs: `Vision Guard`
- Repository path audited here: `D:\CDAC_PROJECT\CV_Project`
- Runtime style: Python inference app
- UI: Gradio
- Training loop: none

## Current Tracked Runtime Files

- `app.py`
- `cache_utils.py`
- `clip_generator.py`
- `pipeline.py`
- `qwen_verifier.py`
- `report_generator.py`
- `segmenter.py`
- `tracker.py`
- `vector_index.py`
- `video_reader.py`
- `vlm.py`
- `requirements.txt`
- `VisionGuard_Colab.ipynb`
- `README.md`
- `PROJECT_DOCUMENTATION.md`

## Current Runtime Stack

- UI: Gradio
- Video reader: Decord with OpenCV fallback
- Detector/tracker wrapper: Ultralytics YOLO + BoT-SORT
- Embedding model: `google/siglip2-so400m-patch14-384`
- Verifier/grounder: `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`
- Segmenter: `facebook/sam2.1-hiera-small`
- Vector index: turbovec `IdMapIndex` with NumPy fallback

## Completed Phase Work

### Phase 1

Requested change:

- change only the default `sample_sec` in `pipeline.py:index_video_iter`

Completed result:

- `sample_sec=0.75`
- `sample_sec=1.25` removed from that signature
- `win_sec=4.5` unchanged

### Phase 2

Requested change:

- update detector defaults in `tracker.py`
- update `.gitignore`

Completed result:

- `tracker.py:ObjectTracker.__init__` now defaults to:
  - `model="yolo11m.pt"`
  - `imgsz=640`
- `.gitignore` contains both:
  - `yolo11n.pt`
  - `yolo11m.pt`
- `pipeline.py:VisionGuardPipeline.__init__` now also defaults to `yolo="yolo11m.pt"`

### Phase 3

Requested change:

- reduce verifier threshold for abstract/event-style queries

Completed result in `qwen_verifier.py`:

- added class-level `_ABSTRACT_TERMS`
- added `_confidence_threshold(query)`
- replaced direct `0.45` comparison inside `verify_query()` with `self._confidence_threshold(query)`

### Phase 4

Requested change:

- make local Windows CPU development avoid loading Qwen

Completed result in `qwen_verifier.py`:

- added module-level `_DEV_MODE`
- `load()` enters `dev_passthrough` on Windows without CUDA
- `verify_query()` returns a synthetic success payload in `dev_passthrough`

Important effect:

- Windows CPU local verification is not a faithful accuracy environment
- it is a developer bypass only

### Phase 5

Requested change:

- expose model warmup status instead of silently swallowing failures

Completed result:

- `pipeline.py`
  - added `self._warmup_failures = {}`
  - added `self._warmup_done = False`
  - rewrote `warmup_models()`
  - added `warmup_status()`
- `app.py`
  - added `get_system_status()`
  - wired `demo.load(fn=get_system_status, inputs=None, outputs=status)`

## Additional Fix Completed After Phase 5

Colab error observed:

- `AttributeError: Cannot call load outside of a gradio.Blocks context.`

Cause:

- `demo.load(...)` had been placed outside the `with gr.Blocks(...)` context

Fix applied:

- moved `demo.load(...)` inside the `with gr.Blocks(...) as demo:` block in `app.py`

## Current User-Facing Flow

1. Launch app
2. Upload a video or choose a sample video
3. Click `step 1: scan video`
4. Wait for scan completion
5. Enter natural-language query
6. Click `step 2: find matches`
7. Review returned matched frames and timestamps
8. Optionally export selected outputs

## Current Scan Pipeline Summary

From `pipeline.py:index_video_iter(...)`:

1. Create run directory
2. Reset tracker state
3. Read video through `DecordVideoReader`
4. Sample frames every `0.75s`
5. Apply cheap duplicate/motion filter
6. Run batch detection on interesting frames
7. Drop non-content frames
8. Save sampled frames to disk
9. Embed sampled frames with SigLIP2
10. Build frame-level and segment-level indexes
11. Write `index.json`
12. Yield preview updates during scan

## Current Query Pipeline Summary

From `pipeline.py`:

1. Normalize query
2. Reject event-style queries and unsupported simple exact-object labels
3. Embed normalized query with SigLIP2
4. Try detector-first retrieval
5. Try frame ANN retrieval
6. Try object fallback retrieval
7. Try segment ANN retrieval
8. Reselect dense best frame inside candidate time window
9. Verify top candidates with Qwen using exact-label constraints
10. Prepare gallery/result rows

## Current Export Pipeline Summary

From `pipeline.py:export_selected(...)`:

1. Resolve selected hit rows
2. Ensure raw clips exist
3. Run segmentation job for selected matches
4. Write JSON, CSV, HTML reports
5. Write ZIP archive

## Current Colab Run Path

The simplest non-persistent Colab path currently expected is:

```python
!git clone https://github.com/priyansupattanaik/visionguard-ai.git || true
%cd visionguard-ai
!git pull
!python -m pip install --upgrade pip
!pip install -r requirements.txt
import os
os.environ["GRADIO_SHARE"] = "1"
os.environ["VISION_GUARD_HOST"] = "0.0.0.0"
!python -u app.py
```

Expected usage:

- ignore `0.0.0.0`
- open only the printed `gradio.live` URL

## Current Known Code-Level Caveats

These are evidence-based and visible in current source.

### 1. Windows CPU verifier bypass

- `qwen_verifier.py` bypasses real Qwen inference on Windows without CUDA

### 2. Warmup is now visible, not silent

- failures are exposed through `warmup_status()`

### 3. `search_stream()` only emits confirmed rows

- low-confidence rows are not streamed unless they become verified matches

### 4. Unsupported simple exact-object labels return no matches

- `taxi`-style unsupported exact labels are rejected early instead of loosely matching `car`

### 5. Event-style queries are intentionally disabled

- collision, fight, fall, crowd, and loitering are currently out of scope

### 6. Current documentation source of truth

- full audited repo documentation lives in `PROJECT_DOCUMENTATION.md`

## Files Most Relevant For Future Work

- `app.py`
- `pipeline.py`
- `tracker.py`
- `vlm.py`
- `qwen_verifier.py`
- `segmenter.py`
- `video_reader.py`
- `clip_generator.py`
- `report_generator.py`
- `vector_index.py`
- `VisionGuard_Colab.ipynb`

## Recommended Headroom Usage For This Snapshot

If you want to compress current project state with Headroom later, this file is the best single input to start from.

Then optionally add:

- `PROJECT_DOCUMENTATION.md`
- `README.md`
- selected diff summaries from current modified files

## Removal

This file is optional and isolated.
If you do not want Headroom-related project context stored in the repo, delete:

- `optional_integrations/headroom/VISION_GUARD_CONTEXT.md`
