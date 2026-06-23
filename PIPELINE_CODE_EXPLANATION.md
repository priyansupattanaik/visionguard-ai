# Pipeline Code Explanation

This document explains the project by following `pipeline.py`, because that file is the real coordinator of the application. If someone wants to understand how Vision Guard works without reading the whole repository first, this is the best place to start.

The key idea is simple:

1. scan a video once
2. keep only useful sampled frames
3. describe those frames with detector metadata and SigLIP embeddings
4. build searchable frame and segment indexes
5. answer user queries by retrieving likely candidates
6. verify the best candidates with Qwen
7. optionally export clips and segmented evidence

## Why `pipeline.py` matters

`VisionGuardPipeline` is the top-level orchestrator. It does not implement every low-level model itself, but it decides:

- what happens during scan time
- what happens during query time
- how helper modules are called
- how results are ranked
- how output files are generated

In other words, the helper files are the workers, but `pipeline.py` is the manager.

## The helper modules that `pipeline.py` controls

At initialization time, the pipeline creates one object for each major responsibility:

- `ObjectTracker` from `tracker.py`
  Used mainly for YOLO detection in the active scan path.
- `SearchEncoder` from `vlm.py`
  Uses SigLIP2 to embed images and text into the same vector space.
- `QwenFrameVerifier` from `qwen_verifier.py`
  Verifies whether a candidate frame truly satisfies the exact user query.
- `GroundedSegmenter` from `segmenter.py`
  Produces segmented visual outputs during export.
- `SegmentVectorIndex` from `vector_index.py`
  Stores and searches frame and segment embeddings.
- `DecordVideoReader` from `video_reader.py`
  Reads frames from the video efficiently.
- `ClipGenerator` from `clip_generator.py`
  Cuts raw result clips from the source video.
- `ReportGenerator` from `report_generator.py`
  Writes JSON, CSV, HTML, and ZIP outputs.

So the class is not a model by itself. It is the runtime controller for all models and outputs.

## The lifecycle of the pipeline

The pipeline has three main phases:

1. setup and warmup
2. video indexing
3. query and export

## Phase 1: setup and warmup

The constructor `__init__()` wires together all dependencies and prepares shared state.

Important state:

- `self.idx`
  In-memory representation of the indexed video.
- `self.frame_idx`
  Vector index for frame embeddings.
- `self.search_idx`
  Vector index for segment embeddings.
- `self.last_hits`
  The latest prepared query results, used by export.
- `self.pool`
  A thread pool used for overlapping tasks like file writes, verification jobs, and clip/segment generation.

The method `warmup_models()` loads the heavy components early:

- tracker
- encoder
- verifier

This is why the UI can start loading models in the background before the user searches.

## Phase 2: video indexing

The main indexing function is:

`index_video_iter(video, sample_sec=0.75, win_sec=4.5)`

This is the most important method in the file.

### What it is trying to achieve

The system does not want to store every frame of the video. That would be too slow and too redundant. Instead, it tries to create a compact searchable summary of the video.

It does this by:

- sampling frames every `0.75` seconds
- rejecting near-duplicates
- rejecting empty frames
- extracting object metadata
- generating SigLIP embeddings
- grouping nearby frames into short segments
- building vector indexes for both frames and segments

### Step 2.1: create a fresh run folder

`_new_run(video)` creates a timestamped output folder with:

- `frames/`
- `clips/`
- `reports/`
- `segments/`

This isolates one scan from another.

### Step 2.2: open the video and decide sample points

The pipeline opens the video through `DecordVideoReader`.

Then it calculates:

- `fps`
- total frame count
- duration
- the frame step size derived from `sample_sec`

If `sample_sec=0.75`, then the code samples roughly one frame every 0.75 seconds, not every frame of the original video.

### Step 2.3: read sampled frames in batches

The pipeline builds a list of sample frame indices and reads them in small chunks with `get_batch(...)`.

This is an important performance choice:

- it avoids full-frame-by-frame decoding logic in Python
- it allows batched downstream processing

This is batching, but it is not video sharding. The code does not split the original video into many subvideos and process them as independent jobs.

### Step 2.4: reject unhelpful frames early

Before running expensive embedding work, the pipeline filters frames aggressively.

It uses:

- `_cheap_signature(frame)`
  Builds a tiny blurred grayscale signature.
- `_frame_diff_score(sig_a, sig_b)`
  Measures visual change.
- `_is_interesting_frame(...)`
  Keeps the frame if it is the first sample, has enough motion, or enough time has passed since the last kept frame.
- `_is_non_content_frame(frame, tracks)`
  Rejects very dark, flat, edge-poor frames when no detections are present.

This is one of the main reasons indexing stays practical on long CCTV videos. A lot of nearly identical or empty frames never reach the expensive model stages.

### Step 2.5: run YOLO detection on the kept candidates

For each batch of interesting frames, the pipeline runs:

`self.trk.detect_batch(...)`

This gives per-frame detections such as:

- class name
- confidence
- box coordinates

Even though `tracker.py` contains tracking support, the active indexing path currently uses batched detection and stores empty `tracks` lists.

### Step 2.6: derive metadata for each kept frame

For each detection batch result, the pipeline builds frame metadata:

- `objects`
  A dictionary of detected classes and counts for that frame.
- `detections`
  Raw detection rows with boxes, confidence, class, and optional estimated color.
- `appearances`
  Tags such as `white car` or `blue truck`.
- `motion_score`
  How different the frame was from the previous sampled signature.
- `keep_reason`
  Why the frame was kept: `first`, `motion`, or `forced_gap`.
- `object_delta`
  How much the set of detected object classes changed compared with the previous kept frame.
- `still_people`
  A small heuristic for low-motion person frames.

This stage matters because the system is not relying only on embeddings. It also stores explicit object evidence that can later power detector-first retrieval.

### Step 2.7: write JPEGs and embed frames

Each kept frame is added to a `pending` queue.

When enough frames accumulate, `flush_pending()` runs:

- JPEG writes are submitted to the thread pool
- SigLIP embeds the pending frames in batch with `self.enc.embed_frames(...)`
- the code waits for the writes to finish
- frame rows and frame vectors are finalized

This overlap is intentional. The code tries to hide some disk-write cost behind embedding work.

### Step 2.8: build segment-level representations

After frame rows are complete, the pipeline groups nearby kept frames into fixed-width windows.

The segment width is controlled by:

- `win_sec=4.5`
- `sample_sec=0.75`

So each segment usually represents a small local time window made from several sampled frames.

For each segment, the pipeline:

- averages the frame embeddings
- normalizes the result
- aggregates object labels
- aggregates lightweight temporal stats

This produces a segment row with:

- `start`
- `end`
- `mid`
- `emb`
- `objects`
- `temporal_stats`

The segment index is useful when one frame alone is too narrow, but a short moment of the video better represents the query.

### Step 2.9: finalize the searchable index

At the end of indexing, the pipeline builds two vector indexes:

- `self.frame_idx`
  Searchable index of frame embeddings.
- `self.search_idx`
  Searchable index of segment embeddings.

If `turbovec` is available, it is used as the ANN backend. Otherwise, the code falls back to NumPy similarity search.

Then the pipeline writes `reports/index.json` and stores the in-memory structure in `self.idx`.

### What `self.idx` contains

At a high level, `self.idx` contains:

- video-level metadata
- indexed frame rows
- indexed segment rows

This is the in-memory snapshot that later query methods use.

## Phase 3: query understanding and candidate retrieval

When the user searches, the most important internal method is:

`_candidate_hits(raw_q, top_k=4)`

This method does not return final answers immediately. It builds and ranks candidate matches first.

### Step 3.1: normalize the query

The code starts with `_normalize_query(q)`.

Examples of what this does:

- `cars` becomes `car`
- `buses` becomes `bus`
- `peoples` becomes `people`

The goal is not full language understanding. The goal is controlled normalization for the supported query style.

### Step 3.2: understand object and color hints

The pipeline extracts:

- object hints with `_q_objs(q)`
- color hints with `_query_colors(q)`

This helps queries like:

- `white car`
- `blue truck`
- `person`

### Step 3.3: reject unsupported query families early

Two conservative checks can stop the search before expensive retrieval:

- `_is_event_query(q)`
  Rejects event-style queries such as `fight`, `accident`, or `crowd`.
- `_is_simple_unsupported_object_query(q)`
  Rejects unsupported exact object labels instead of silently widening them.

This is a deliberate design choice. The system prefers returning nothing over pretending it understood a query that the current code does not safely support.

### Step 3.4: detector-first retrieval

The first retrieval mode is `_refine_detector_hits(q, top_k)`.

This is the best path for supported object queries.

It searches stored frame detections for exact object evidence and optionally matches color-aware appearances. If the query is `white car`, this path can often answer the query before semantic search needs to do much.

This mode is strong because it uses explicit detector evidence instead of only vector similarity.

### Step 3.5: frame ANN retrieval

If detector-first retrieval does not produce hits, the pipeline moves to semantic frame retrieval.

It does this by:

- embedding the query with `_embed_query(q)`
- searching `self.frame_idx.search(qv, fetch_k)`

This is SigLIP-based text-to-frame retrieval.

The returned scores are then adjusted using lightweight logic such as:

- object overlap boosts
- object mismatch penalties
- color match boosts
- small phrase-specific boosts like `sitting` with `person`

Then nearby frame hits are clustered into cleaner candidate windows with `_cluster_frame_hits(...)`.

### Step 3.6: object fallback retrieval

If strong semantic frame hits are not found, the pipeline tries `_fallback_object_hits(q, top_k)`.

This mode is less strict than detector-first matching. It uses stored frame-level object and appearance metadata to offer trusted fallback results for supported object queries.

These hits are usually marked low-confidence, but they are still useful because explicit object evidence exists.

### Step 3.7: weak semantic retrieval

If ranked semantic rows exist but not strongly enough, the code can still produce weak semantic candidates.

These are the nearest available visual matches rather than strong confirmed ones.

This helps the system degrade gracefully instead of failing silently.

### Step 3.8: segment ANN retrieval

If the frame-level path still does not produce final candidates, the pipeline searches the segment index.

This asks:

"Which short time window best matches the query?"

It is useful because some queries are represented better by a small scene interval than by one sampled frame.

## Dense frame reselection

After any retrieval mode produces candidate windows, the pipeline often calls:

`_apply_reselection(...)`

This is a precision-improvement step.

The logic is:

1. take a candidate clip window
2. re-read denser frames inside that window, every `0.1s`
3. embed those denser frames with SigLIP
4. choose the best-scoring exact frame

This matters because the indexing pass only sampled every `0.75s`. The best true visual match may be between those sampled timestamps.

So dense reselection is how the system moves from:

- "this area of the video is promising"

to:

- "this exact frame is the strongest representative"

## Verification with Qwen

After candidate generation and reselection, the pipeline verifies only the top few rows with Qwen.

This happens in:

- `_verify_rows(...)`
- `_verify_rows_stream(...)`

### Why Qwen is not used on every frame

Qwen is expensive compared with ANN retrieval.

So the architecture is intentionally staged:

1. cheap narrowing with detector rules and embeddings
2. expensive exact verification on a tiny shortlist

This is the classic retrieve-then-verify pattern.

### What verification changes

For each top candidate, Qwen returns:

- `matched`
- `confidence`
- `caption`
- `boxes`

The pipeline then:

- stores `verified_caption`
- stores `verify_score`
- stores grounded boxes if available
- boosts the score for confirmed matches
- demotes low-confidence or unverified rows
- rewrites the human-facing `summary`

If Qwen confirms a result, that row becomes a strong final answer.

If Qwen does not confirm anything, the system may still return detector/object-fallback hits for supported object queries.

## Streaming search versus one-shot search

The pipeline exposes two user-facing search styles:

- `search_stream(...)`
- `search(...)`

`search_stream(...)` is used when the UI wants progressive feedback. It emits confirmed hits as they become available.

`search(...)` does the same overall logic in one pass and returns the final list.

Both rely on `_candidate_hits(...)` and the same verification rules.

## Preparing results for the UI

After search, the pipeline converts raw candidate rows into prepared display rows with:

`prepare_hits(hits, query)`

This adds UI-friendly fields such as:

- `match_id`
- `label`
- `gallery_frame`
- `raw_clip`
- `clip`
- `frames`
- `segmented`

The first hit may also receive a gallery overlay image via `_attach_gallery_frame(...)`, which tries to draw Qwen grounding boxes or falls back to detector boxes.

## Export flow

When the user chooses results to export, the entry point is:

`export_selected(picks, query)`

This method:

1. finds the chosen prepared rows from `self.last_hits`
2. ensures each result has a clip
3. optionally ensures segmentation output exists
4. writes final report artifacts

### Raw clip creation

`_build_raw_clip(row)` uses `ClipGenerator` to cut a local MP4 around the matched time window.

`_ensure_raw_clip(...)` handles caching and optional async execution.

### Segmented clip creation

`_segment_payload(row, query)` calls `GroundedSegmenter.segment_clip(...)`.

This stage uses grounding and SAM-based masking to produce a more visually focused export when possible.

If segmentation cannot ground anything useful, the raw clip is kept as fallback.

### Final output files

The export method writes:

- selected JSON
- selected CSV
- selected HTML
- selected ZIP

The ZIP contains the clip files for the selected matches.

## What the code is really optimizing for

Reading `pipeline.py` carefully shows the project is trying to balance:

- speed
- practical CCTV usefulness
- conservative correctness

The code makes that tradeoff in several ways:

- sample sparsely first
- filter duplicates early
- use detector evidence when available
- use embeddings for semantic flexibility
- use segment search when one frame is not enough
- reselect a denser best frame before final judgment
- let Qwen confirm only a small shortlist
- preserve trusted fallback results for supported object queries

So the design is not "caption everything and search captions."

It is closer to:

- build a compact visual memory of the video
- search that memory quickly
- verify exact claims only on the most promising candidates

## A plain-English summary of the full flow

If we explain the project only through `pipeline.py`, the flow is:

1. the video is sampled at fixed time intervals
2. duplicate or unhelpful frames are discarded
3. YOLO extracts object evidence from the kept frames
4. SigLIP turns kept frames into searchable vectors
5. nearby kept frames are merged into segment-level vectors
6. both frame and segment vectors are indexed
7. a user query is normalized and turned into a text vector
8. the pipeline retrieves candidate matches using detector rules, frame ANN, fallback logic, weak semantic retrieval, and segment ANN
9. the pipeline reselects a better exact frame within promising windows
10. Qwen verifies whether the top candidates truly satisfy the exact query
11. the best confirmed rows are prepared for display
12. selected rows can be exported as raw clips, segmented clips, and reports

## Recommended reading order in the code

If someone wants to study the source after reading this document, the best reading order is:

1. `pipeline.py`
2. `vlm.py`
3. `qwen_verifier.py`
4. `vector_index.py`
5. `tracker.py`
6. `segmenter.py`
7. `app.py`

That order keeps the mental model clean:

- orchestration first
- embeddings second
- verification third
- retrieval backend fourth
- detector and export helpers after that

## Final takeaway

Yes, explaining the project through `pipeline.py` is the right choice.

It is the best single-file summary because it shows:

- what is indexed
- what is searched
- how results are ranked
- where SigLIP is used
- where Qwen is used
- why turbovec exists
- how exports are produced

The rest of the repository makes the pipeline possible, but `pipeline.py` explains the real flow of the project end to end.
