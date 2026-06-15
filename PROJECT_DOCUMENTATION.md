# VisionGuard AI Project Documentation

## 1. Project Summary

VisionGuard AI is a scan-first CCTV video search system.

The goal is:

- scan a CCTV or general surveillance-style video
- index the video once
- ask a natural-language query after scanning
- return the most relevant time windows
- show the matched clip
- localize the queried object or region in matched clips
- generate exportable clips and reports

This is not a training pipeline. It is an inference pipeline.

The project does not fine-tune models when a video is uploaded.

## 2. Problem Statement

A person cannot manually watch long CCTV footage every time they need to find:

- a person sitting
- a white car entering
- a fight
- a fall
- a possible collision
- a crowd
- loitering

So the project converts a raw video into a searchable indexed representation, then uses vision-language retrieval and grounding to find likely matching parts.

## 3. Current High-Level Flow

The current app flow is:

1. Upload or choose a video.
2. Click `step 1: scan video`.
3. The system samples the video, tracks objects, and builds retrieval embeddings.
4. After scan completes, enter a natural-language query.
5. Click `step 2: find matches`.
6. The system searches indexed windows and returns top time ranges.
7. The first clip is prepared immediately.
8. Other clips trim in the background.
9. When a match is opened, localization and segmentation are prepared for that clip.
10. Selected clips and reports can be exported.

## 4. Core Design Decision

The project is scan-first, not query-first.

Why:

- scanning once is cheaper than rescanning the whole video for every query
- it makes repeated queries practical
- it separates indexing cost from query cost
- it fits CCTV usage better because one video can be queried many times

## 5. Repo Map

Current main files:

- [app.py](/D:/CDAC_PROJECT/CV_Project/app.py:1): Gradio UI and user flow
- [pipeline.py](/D:/CDAC_PROJECT/CV_Project/pipeline.py:1): end-to-end video indexing, search, background clip prep, export
- [tracker.py](/D:/CDAC_PROJECT/CV_Project/tracker.py:1): YOLO + ByteTrack object tracking
- [vlm.py](/D:/CDAC_PROJECT/CV_Project/vlm.py:1): text-frame embedding search
- [vector_index.py](/D:/CDAC_PROJECT/CV_Project/vector_index.py:1): `turbovec`-backed segment vector index with NumPy fallback
- [locate_anything.py](/D:/CDAC_PROJECT/CV_Project/locate_anything.py:1): NVIDIA LocateAnything wrapper
- [segmenter.py](/D:/CDAC_PROJECT/CV_Project/segmenter.py:1): LocateAnything grounding + SAM2 segmentation + segmented clip render
- [clip_generator.py](/D:/CDAC_PROJECT/CV_Project/clip_generator.py:1): clip extraction and browser-ready finalize
- [report_generator.py](/D:/CDAC_PROJECT/CV_Project/report_generator.py:1): JSON/CSV/HTML/ZIP outputs
- [cache_utils.py](/D:/CDAC_PROJECT/CV_Project/cache_utils.py:1): Colab Drive-backed cache setup
- [requirements.txt](/D:/CDAC_PROJECT/CV_Project/requirements.txt:1): Python deps
- [VisionGuard_Colab.ipynb](/D:/CDAC_PROJECT/CV_Project/VisionGuard_Colab.ipynb:1): Colab run notebook
- [README.md](/D:/CDAC_PROJECT/CV_Project/README.md:1): quick start

Runtime folders:

- `assets/`: sample videos
- `output/`: generated run outputs
- `.yolo/`: Ultralytics settings/model metadata

## 5A. Architecture Overview

The project follows a modular pipeline architecture.

Main layers:

1. Presentation layer
2. Orchestration layer
3. Vision analysis layer
4. Search and ranking layer
5. Localization and segmentation layer
6. Export and reporting layer
7. Runtime/cache layer

### 5A.1 Presentation Layer

The presentation layer is the Gradio app in [app.py](/D:/CDAC_PROJECT/CV_Project/app.py:1).

It handles:

- video upload
- scan button
- query input
- result display
- clip selection
- export actions

It does not implement the vision logic itself.

Its job is:

- gather user input
- call the pipeline
- render outputs returned from the pipeline

### 5A.2 Orchestration Layer

The orchestration layer is [pipeline.py](/D:/CDAC_PROJECT/CV_Project/pipeline.py:1).

This is the backend coordinator.

It decides:

- when to scan
- when to search
- when to trim clips
- when to start background jobs
- when to run localization and segmentation
- when to export files

This file is the real backend spine of the project.

### 5A.3 Vision Analysis Layer

The vision analysis layer is split across:

- [tracker.py](/D:/CDAC_PROJECT/CV_Project/tracker.py:1)
- [vlm.py](/D:/CDAC_PROJECT/CV_Project/vlm.py:1)
- [locate_anything.py](/D:/CDAC_PROJECT/CV_Project/locate_anything.py:1)
- [segmenter.py](/D:/CDAC_PROJECT/CV_Project/segmenter.py:1)

Each module owns one kind of model behavior instead of mixing everything in one file.

### 5A.4 Search and Ranking Layer

This is conceptually inside [pipeline.py](/D:/CDAC_PROJECT/CV_Project/pipeline.py:179).

It uses:

- SigLIP2 embedding similarity
- `turbovec` `IdMapIndex` for the primary dense vector lookup
- query-object overlap boosts
- deduplication of near-identical windows

This layer turns indexed windows into ranked matches.

### 5A.5 Localization and Segmentation Layer

This layer is handled by [segmenter.py](/D:/CDAC_PROJECT/CV_Project/segmenter.py:12).

Its job starts only after a match is already found.

It uses:

- LocateAnything-3B for primary grounding
- SAM2 for masks

### 5A.6 Export and Reporting Layer

This layer is split across:

- [clip_generator.py](/D:/CDAC_PROJECT/CV_Project/clip_generator.py:1)
- [report_generator.py](/D:/CDAC_PROJECT/CV_Project/report_generator.py:1)

It handles:

- raw clip generation
- segmented clip generation
- browser-ready video finalize
- JSON/CSV/HTML/ZIP outputs

### 5A.7 Runtime and Cache Layer

This layer is [cache_utils.py](/D:/CDAC_PROJECT/CV_Project/cache_utils.py:1).

It configures cache paths in Colab so:

- Hugging Face files persist
- Torch caches persist
- Ultralytics settings persist

This is important because model download cost is large in Colab.

## 5B. Backend Working Explained

This section describes how the backend actually processes a request.

### 5B.1 App Startup

When `python app.py` starts:

1. `setup_cache()` runs first.
2. The `VisionGuardPipeline` object is created.
3. The Gradio UI tree is created.
4. No heavy model is loaded immediately just because the app started.

The models use lazy loading.

That means:

- the app object is created first
- models load when their first real task is called

Why:

- faster startup
- avoids loading every model if the user never reaches a later stage

### 5B.2 What Happens During Scan

When the user clicks `step 1: scan video`:

`app.py` calls `scan_only(video)`, which then iterates over `pipe.index_video_iter(video)`.

Inside `index_video_iter(...)`:

1. A new run folder is created under `output/`.
2. The tracker state is reset.
3. OpenCV opens the source video.
4. Video metadata is read:
   - fps
   - frame count
   - duration
5. The sampling stride is computed from `sample_sec`.
6. The video is read frame by frame.
7. Only sampled frames are sent into the heavy semantic indexing path.

For each sampled frame:

1. YOLO + ByteTrack produce tracked objects.
2. Object counts and ids are collected.
3. A live preview overlay is drawn.
4. SigLIP2 creates a frame embedding.
5. The raw frame is saved to `output/.../frames`.
6. A preview event is yielded back to Gradio.

That yield behavior is important.

It means the backend is streaming progress to the UI while scanning, not waiting until the entire scan finishes.

### 5B.3 What Happens After Frame Scan Completes

After all sampled frames are collected:

1. Nearby sampled frames are grouped into windows.
2. Window embeddings are averaged.
3. Object names and track ids are aggregated.
4. The final search index is stored in memory in `self.idx`.
5. A `turbovec` segment index is also written to disk.
6. A JSON index report is also written to disk.

At this point the video is searchable.

### 5B.4 What Is Stored in Memory

The most important backend state after scan is `self.idx` in [pipeline.py](/D:/CDAC_PROJECT/CV_Project/pipeline.py:174).

It contains:

- source video path
- video metadata
- indexed windows

Each indexed window stores:

- stable segment id
- start time
- end time
- midpoint
- embedding vector
- representative frame path
- object names
- track ids

This in-memory index is the reason repeated queries are possible without rescanning.

### 5B.5 What Happens During Search

When the user clicks `step 2: find matches`:

1. `app.py` calls `find_query(q)`.
2. `find_query(q)` calls `pipe.search(q, top_k=4)`.
3. The query is embedded with SigLIP2.
4. `turbovec` searches the persisted segment embedding index.
5. Query-object overlap adds a small boost or penalty.
6. The candidate list is reranked.
7. Nearby duplicate windows are filtered.

This stage is search only.

It does not scan the video from scratch again.

### 5B.6 What Happens Right After Search

Search results are converted into hit rows in `prepare_hits(...)`.

That method:

- stores hit metadata in `self.last_hits`
- prepares the first raw clip immediately
- starts background raw-clip jobs for the remaining hits
- starts background segmentation for the first hit

Why this matters:

- the first result becomes watchable sooner
- the user does not wait for every clip before interacting

### 5B.7 Background Jobs

The backend uses `ThreadPoolExecutor` in [pipeline.py](/D:/CDAC_PROJECT/CV_Project/pipeline.py:29).

Background jobs are used for:

- raw clip trimming
- segmented clip generation

This is backend optimization, not just UI behavior.

The important idea is:

- the pipeline returns control early
- clip processing continues in worker threads

### 5B.8 What Happens When a Match Is Opened

When the user selects a specific match:

1. `app.py` calls `show_match(...)`.
2. `show_match(...)` calls `pipe.pick_match(label, q)`.
3. The backend checks whether the raw clip is already ready.
4. The backend checks whether segmented output is already finished.
5. If not ready, it returns the raw clip first and keeps segmentation in background.
6. If ready, it returns the segmented clip and preview frames.

This is why the current backend tries not to block clip viewing.

### 5B.9 How Localization Works Internally

For matched frames:

1. `LocateAnythingGrounder.detect(...)` is called.
2. It sends image + prompt to `nvidia/LocateAnything-3B`.
3. It parses `<box><...></box>` style outputs into pixel boxes.
4. Those boxes are passed into SAM2.

This is done in [segmenter.py](/D:/CDAC_PROJECT/CV_Project/segmenter.py:32).

### 5B.10 How Segmentation Works Internally

Once boxes are available:

1. SAM2 receives the frame and boxes.
2. SAM2 returns masks.
3. The backend overlays masks on the frame.
4. Preview images are saved to the run folder.
5. A segmented video is rendered.

This segmented output is post-match refinement, not part of the primary retrieval stage.

### 5B.11 How Clip Writing Is Protected

The backend uses atomic file handling for clip outputs.

Process:

1. write to a temporary `.part.mp4`
2. finish the writer
3. optionally re-encode to browser-friendly H.264 with ffmpeg
4. rename to final `.mp4`

Why:

- prevents half-written clips from being shown
- fixes damaged header problems when background tasks and UI access overlap

### 5B.12 How Export Works Internally

When export is triggered:

1. selected hit rows are resolved
2. segmentation is ensured for those rows
3. JSON report is written
4. CSV report is written
5. HTML report is written
6. ZIP archive is written

The backend therefore treats export as a finalization stage.

## 5C. Backend Data Flow Summary

Conceptual backend data flow:

`video path -> sampled frames -> tracking/meta -> embeddings -> indexed windows -> query embedding -> ranked hits -> raw clips -> grounded boxes -> masks -> exports`

This is the shortest accurate summary of the backend processing chain.

## 6. End-to-End Technical Flow

### 6.1 Video Input

The user uploads a video through Gradio in [app.py](/D:/CDAC_PROJECT/CV_Project/app.py:46).

The project then passes the video path into `pipe.index_video_iter(...)` in [pipeline.py](/D:/CDAC_PROJECT/CV_Project/pipeline.py:91).

### 6.2 Video Sampling

The pipeline does not semantically embed every frame.

It samples frames every `sample_sec` seconds. Current default:

- `sample_sec = 1.5`

Why:

- much lower indexing cost
- practical for Colab
- still enough for broad CCTV search

Tradeoff:

- very short events between sampled frames may be missed or blurred into nearby windows

### 6.3 Object Tracking

For each sampled frame:

- YOLO detects objects
- ByteTrack keeps track ids across time

This is handled by [tracker.py](/D:/CDAC_PROJECT/CV_Project/tracker.py:11).

Current tracker stack:

- model: `yolo11s.pt`
- tracker: `bytetrack.yaml`

Why used:

- fast
- practical in Colab
- stable object boxes for people and vehicles
- good enough for preview and object presence signals

What it contributes:

- object names like `person`, `car`, `truck`
- boxes
- track ids
- live preview overlays during scanning

What it does not solve:

- event understanding
- semantic retrieval
- natural-language grounding

### 6.4 Retrieval Embeddings

Each sampled frame is converted into an embedding with SigLIP2 in [vlm.py](/D:/CDAC_PROJECT/CV_Project/vlm.py:9).

Current retrieval model:

- `google/siglip2-base-patch16-224`
- `turbovec` stores and searches the resulting window vectors

Why used:

- text-image similarity
- smaller and lighter than the heavier earlier retrieval stack
- reasonable balance for Colab and repeated search

How it works:

- frame -> image embedding
- query -> text embedding
- cosine similarity between query and indexed windows

### 6.5 Window Indexing

After frame sampling, the project groups neighboring sampled frames into windows in [pipeline.py](/D:/CDAC_PROJECT/CV_Project/pipeline.py:147).

For each window it stores:

- `start`
- `end`
- `mid`
- averaged embedding
- representative frame path
- detected object names
- tracked ids

This becomes the searchable video index.

### 6.6 Query Search

When the user enters a query:

1. The query text is embedded with SigLIP2.
2. `turbovec` returns the top candidate windows by vector similarity.
3. Query-object overlap adjusts the score.
4. Top windows are returned.

Search code is in [pipeline.py](/D:/CDAC_PROJECT/CV_Project/pipeline.py:179).

Why this design:

- keeps search fast after indexing
- supports multiple queries on one scanned video
- avoids rerunning full video analysis for every query
- reduces obvious false-positive labels when the detected objects do not support the query

### 6.7 Clip Preparation

The project does not trim every clip synchronously during search anymore.

Current behavior:

- first matched clip is prepared immediately
- remaining raw clips start trimming in background threads
- segmentation is also background-friendly

This is handled in [pipeline.py](/D:/CDAC_PROJECT/CV_Project/pipeline.py:212) onward.

Why:

- reduces visible latency after `find matches`
- lets the user start watching earlier
- avoids blocking the UI on full post-processing

### 6.8 Localization and Segmentation

Matched clips use a layered localization design in [segmenter.py](/D:/CDAC_PROJECT/CV_Project/segmenter.py:12):

1. `LocateAnything-3B` grounds the natural-language phrase.
2. SAM2 segments the grounded boxes.
3. The project renders segmented preview frames and a segmented clip.

Why this hybrid design:

- `LocateAnything-3B` is strong for free-form grounding and dense cluttered localization
- SAM2 gives region masks, not just boxes

Important limitation:

- LocateAnything improves localization, not temporal event recognition
- SAM2 segments what is localized, but does not understand whether a collision or fight really happened

### 6.9 Export

The project can export:

- selected clips
- CSV
- JSON
- HTML
- ZIP

Handled by [report_generator.py](/D:/CDAC_PROJECT/CV_Project/report_generator.py:1) and [pipeline.py](/D:/CDAC_PROJECT/CV_Project/pipeline.py:308).

## 7. Why These Libraries and Models Were Used

### 7.1 Gradio

Used for:

- quick browser UI
- Colab-compatible public link
- Hugging Face Spaces deployment

Why chosen:

- lowest friction for notebooks and demos
- easy local + Colab + Spaces flow

Alternatives:

- Streamlit
- FastAPI + custom frontend
- Dash

Why not those by default:

- more setup for the same demo goal
- Gradio is better aligned with Colab and Spaces

### 7.2 OpenCV

Used for:

- reading video
- writing clips
- frame operations
- preview overlays

Alternatives:

- PyAV
- decord
- ffmpeg-python

Why OpenCV stayed:

- simple
- already integrated
- practical for clip generation and frame drawing

### 7.3 Ultralytics YOLO

Used for:

- fast object detection
- boxes for people/vehicles
- live scan preview

Alternatives:

- RT-DETR
- Grounding DINO only
- Detectron2

Why YOLO stayed:

- faster and simpler for broad scan-time object presence
- useful even when event detection is weak

### 7.4 ByteTrack

Used for:

- persistent ids across sampled frames

Alternatives:

- BoT-SORT
- DeepSORT

Why ByteTrack:

- stable default from Ultralytics flow
- good speed/quality tradeoff

### 7.5 SigLIP2

Used for:

- retrieval
- text-frame semantic matching

Alternatives:

- CLIP
- EVA-CLIP
- BLIP2 embeddings
- Qwen-VL style reranking

Why SigLIP2:

- strong vision-text retrieval
- lighter than large VLM rerankers
- practical for indexing many windows

### 7.6 turbovec

Used for:

- indexing segment embeddings after scan
- fast top-k vector retrieval during search
- writing a reusable `.tvim` vector index for each scan

Why chosen:

- purpose-built Python vector index
- supports stable external ids through `IdMapIndex`
- lower search overhead than rescoring every segment in Python

Alternatives:

- plain NumPy cosine search
- FAISS
- LanceDB

Why it fits this repo:

- the pipeline already creates one embedding per searchable segment
- segment ids need to stay stable across retrieval and export
- it improves query latency without changing the grounding stage

### 7.7 LocateAnything-3B

Used for:

- matched-frame visual grounding from natural-language phrases

Alternatives:

- Grounding DINO
- OWL-ViT
- Florence-2 style grounding workflows

Why it was added:

- better open-vocabulary grounding intent
- designed for dense and cluttered grounding
- strong fit for `locate the person`, `locate the white car`, `locate all the instances`

Important note:

- it helps localization
- it does not replace a temporal incident model

### 7.8 Grounding DINO

Grounding DINO was used earlier as a grounding fallback.

Current status:

- removed from the active runtime path to reduce model downloads and simplify the stack
- kept here only as a conceptual alternative, not as part of the current backend

### 7.9 SAM2

Used for:

- segmentation masks on matched clips

Alternatives:

- SAM 1
- FastSAM
- MobileSAM

Why SAM2:

- stronger modern segmentation path
- better fit for matched-clip post-processing than for full-video indexing

## 8. Current Model Stack

Current defaults from code:

- tracker: `yolo11s.pt`
- retrieval: `google/siglip2-base-patch16-224`
- vector retriever: `turbovec` `IdMapIndex`
- grounding primary: `nvidia/LocateAnything-3B`
- segmentation: `facebook/sam2.1-hiera-small`

## 9. Why the Project Does Not Process Every Frame for Everything

Full dense processing of every frame for:

- detection
- retrieval
- event reasoning
- grounding
- segmentation

would be too expensive for Colab and casual deployment.

So the pipeline is staged:

- sample and index broadly
- search cheaply
- localize and segment only on matched clips

This is the main optimization concept of the project.

## 10. Performance Optimizations Already Present

- scan-first indexing
- frame sampling instead of full semantic embedding on every frame
- averaged window embeddings
- `turbovec` top-k vector retrieval instead of full Python rescoring over every segment
- background clip trimming
- background segmentation start for the first result
- atomic clip writes to avoid broken partially-written MP4s
- optional ffmpeg finalize to browser-friendly H.264 output
- Colab Drive cache for Hugging Face and Torch assets

## 11. What the Project Is Good At

- natural-language search over one scanned video
- repeated queries after one scan
- object-aware retrieval
- representative-frame retrieval for top matches
- localization and segmentation on matched clips
- selective export
- Colab demo workflow

## 12. What the Project Is Not Guaranteed To Do

The project does not guarantee:

- perfect accuracy
- zero hallucination
- flawless incident detection on all CCTV scenes
- exact event understanding in every dense or occluded video
- unlimited runtime scalability on Colab

This must be stated clearly in any submission or presentation.

## 13. Current Limitations

### 13.1 Collision / Fight / Fall Are Still Hard

These are temporal events.

The current pipeline improves retrieval and localization, but event correctness can still fail because:

- sampling may miss fine-grained motion
- grounding localizes objects, not event truth
- the runtime avoids injecting event labels by default because those labels produced false positives

### 13.2 Query Quality Depends on Prompt Style

Short, concrete prompts work better than vague prompts.

Better:

- `white car near gate`
- `person sitting by wall`
- `person near road`

Worse:

- `show suspicious thing`
- `anything unusual`

### 13.3 Long Videos Still Cost Time

Even with optimization, very long videos still require:

- reading frames
- tracking
- embedding
- retrieval scoring

The project is more scalable than before, but not unbounded.

### 13.4 Segmentation Is Post-Match Only

This is intentional for speed.

The project does not segment the full video during scanning.

## 14. Known Warnings and Their Meaning

### 14.1 HF Token Warning

Meaning:

- unauthenticated Hugging Face downloads

Impact:

- lower rate limits
- slower download startup

### 14.2 Source Video Browser-Codec Warning

Meaning:

- the uploaded source video may not use a browser-friendly codec

Impact:

- Gradio may re-encode it for display

### 14.3 SAM2 Checkpoint Warning

Meaning:

- Transformers warns about checkpoint/model type relation

Impact:

- warning only unless runtime actually fails

### 14.4 LocateAnything Load Risk

Meaning:

- `trust_remote_code=True` research model path may change upstream

Impact:

- if it fails, matched-clip grounding fails because the current runtime stack no longer keeps a grounding fallback model

## 14A. Terms and Meanings

### Grounding

Grounding means connecting a natural-language phrase to a specific spatial region in an image or frame.

Example:

- query: `white car near gate`
- grounding result: box around the white car

Grounding is not the same as segmentation.

### Segmentation

Segmentation means identifying the exact pixels of an object or region, not just a rectangle.

Example:

- box around car = detection or grounding box
- exact car-shaped mask = segmentation

### Detection

Detection means finding objects and returning boxes plus class labels.

Example:

- `person`
- `car`
- `truck`

### Tracking

Tracking means keeping object identity over time across frames.

Example:

- the same person in multiple frames gets one track id

### Retrieval

Retrieval means searching indexed visual content using semantic similarity.

Example:

- query text embedding is compared with video-window embeddings

### Embedding

An embedding is a numeric vector representing semantic content.

The project uses embeddings so text and frames can be compared mathematically.

### Indexing

Indexing means preprocessing the video into searchable units.

In this project, indexing stores:

- timestamps
- frame paths
- embeddings
- object names

### Scan-First

Scan-first means the video is analyzed before the user starts querying it.

This is why repeated queries are faster later.

## 14B. Pretrained Model Assessment

This section records the current model-selection conclusion based on official sources.

### 14B.1 LocateAnything-3B

Officially, LocateAnything-3B is presented as a visual grounding model for:

- phrase grounding
- dense object detection
- open-set localization
- cluttered-scene localization

Why it helps this project:

- better natural-language localization of people, cars, trucks, and other queried regions

Why it does not solve everything:

- it localizes regions in frames
- it does not by itself recognize temporal events like collisions across time

### 14B.2 VideoMAE and TimeSformer

Officially, VideoMAE and TimeSformer are pretrained video-classification models in Transformers.

Why they were considered:

- they are stronger closed-label video classifiers than raw retrieval-only tagging

Why they were not made the default here:

- they are still Kinetics-style action classifiers
- they are not natural-language query models
- they do not directly solve free-form CCTV query search
- they still do not provide a verified single answer for all of:
  - collision
  - fight
  - fall
  - crowd
  - loitering

Current conclusion:

- there is no verified single pretrained model from the checked official sources that cleanly replaces retrieval, event detection, and grounding together for this project
- the current modular stack remains the most practical architecture for this repo

## 15. Output Structure

Per run, the project creates a timestamped folder under `output/` with:

- `frames/`
- `clips/`
- `segments/`
- `reports/`

Expected report artifacts:

- `index.json`
- selected `json`
- selected `csv`
- selected `html`
- selected `zip`

## 16. Colab and Deployment Concept

### 16.1 Why Colab

- free or low-cost GPU access
- easy notebook workflow
- simple demo environment

### 16.2 Why Drive Cache

- avoid redownloading models each session
- faster restart loop

### 16.3 Why Hugging Face Spaces

- easy public deployment
- aligns with Gradio
- easy to demo

### 16.4 Deployment Tradeoff

- CPU deployment is possible
- GPU deployment is strongly preferred
- segmentation and grounding become slow on CPU

## 17. How to Keep This Documentation Updated

Whenever the project changes, update this file if any of these change:

- model names
- app flow
- query flow
- scan flow
- output files
- dependency list
- deployment steps
- limitations
- fallback behavior

Minimum update checklist:

1. Update `Current Model Stack`.
2. Update `Core Design Decision` if flow changes.
3. Update `Repo Map` if files are added or removed.
4. Update `Known Warnings` if runtime behavior changes.
5. Update `Questions You May Be Asked`.

## 18. Questions You May Be Asked

### 18.1 Problem and Goal

Q: What problem does this project solve?  
A: It reduces manual CCTV review by turning one video into a searchable indexed representation and retrieving relevant clips from natural-language queries.

Q: Why is scan-first better than searching the raw video every time?  
A: It pays the analysis cost once and allows repeated queries cheaply afterward.

### 18.2 Architecture

Q: What are the major stages of the pipeline?  
A: Video scan, frame sampling, tracking, retrieval embedding, window indexing, query search, clip prep, grounding, segmentation, export.

Q: Why use multiple models instead of one large model?  
A: Each part solves a different subproblem more efficiently: tracking, retrieval, grounding, segmentation.

Q: Why not process every frame densely?  
A: It is too expensive for Colab and would make indexing much slower.

### 18.3 Model Choice

Q: Why use YOLO?  
A: Fast object detection and useful live preview overlays.

Q: Why use SigLIP2?  
A: It provides text-image retrieval embeddings for semantic search.

Q: Why use LocateAnything-3B?  
A: It improves open-vocabulary grounding on matched frames from natural-language phrases.

Q: Why was Grounding DINO removed from the active runtime path?  
A: To reduce model downloads and simplify the runtime stack after moving to LocateAnything as the primary grounding model.

Q: Why use SAM2?  
A: To convert grounded boxes into region masks and segmented previews.

### 18.4 Accuracy

Q: Is the system 100 percent accurate?  
A: No. It is a practical retrieval-and-localization system, not a guaranteed perfect forensic system.

Q: Why can collision detection still fail?  
A: Collision is a temporal event. Localization alone is not enough, and the current runtime does not claim reliable event truth from object boxes alone.

Q: Does the system hallucinate?  
A: It can produce incorrect retrievals or weak event matches. No honest open-world system can promise zero hallucination.

### 18.5 Optimization

Q: What optimizations are used?  
A: Frame sampling, scan-first indexing, averaged window embeddings, background clip trimming, on-demand segmentation, caching, atomic video writes.

Q: Why is the first clip shown faster now?  
A: Because only the first result is prepared immediately and the others trim in background.

Q: Why are atomic writes used for clips?  
A: To prevent broken MP4 files from being exposed before writing finishes.

### 18.6 Deployment

Q: Why is Colab suitable?  
A: It offers GPU access and a fast demo workflow for research-style projects.

Q: Why is Hugging Face Spaces suitable?  
A: It matches the Gradio app structure and is easy to publish.

Q: What happens on CPU-only deployment?  
A: The app may run, but grounding and segmentation become slow.

### 18.7 Limitations and Future Work

Q: What is the main current weakness?  
A: Reliable event understanding for collisions, fights, and falls in dense CCTV scenes.

Q: What would improve the project next?  
A: A stronger dedicated temporal incident-recognition stage and better domain-specific evaluation data.

Q: Why not fine-tune everything now?  
A: That requires a labeled CCTV dataset and a different training pipeline, which this repo does not currently implement.

## 19. Conceptual Alternatives

If the project were redesigned, possible substitutions include:

- retrieval:
  - CLIP
  - EVA-CLIP
  - larger VLM reranker
- event models:
  - VideoMAE
  - TimeSformer
  - dedicated action-recognition model
- grounding:
  - Grounding DINO only
  - OWL-ViT
  - Florence-2 style grounding
- segmentation:
  - FastSAM
  - MobileSAM

Why the current stack remains:

- it already balances Colab feasibility, modularity, and functionality
- replacing every stage at once would increase break risk

## 19A. Can LangChain Be Used Here?

Yes, but only in a supporting role.

LangChain can help with:

- metadata indexing
- report generation
- query routing
- retrieval orchestration
- agent-style workflows
- combining video results with text summaries

LangChain does not solve the core vision problems:

- missed bikes
- weak object detection
- weak event classification
- inaccurate grounding
- poor segmentation

So LangChain should not be treated as the main video-understanding engine.

Best use of LangChain in this project:

- store per-window metadata
- store timestamps
- store object tags
- store generated summaries
- query those records using natural language
- call the vision stack only for top candidates

Bad use of LangChain here:

- forcing every frame into a document chain and expecting that alone to make video understanding accurate

Conclusion:

- LangChain can be added around the project
- it should orchestrate search and metadata
- the actual understanding still depends on detector, retrieval, grounding, and event models

## 19B. Why Frame-by-Frame Indexing Is Not the Default

Frame-by-frame indexing sounds attractive, but it is usually the wrong default for surveillance search.

Why it is avoided:

- adjacent frames are often redundant
- storage grows too fast
- embedding cost grows too fast
- search cost grows too fast
- event understanding does not automatically improve
- Colab runtime becomes much heavier

What happens if every frame is indexed densely:

- more GPU time
- more RAM pressure
- larger output artifacts
- slower query ranking
- more duplicated near-identical results

Why sampled windows are better as a default:

- they compress nearby visual information
- they keep timestamps meaningful
- they reduce repeated content
- they are much more practical for repeated querying

When frame-level processing is still useful:

- for the currently selected match
- for final localization
- for final segmentation
- for export refinement

So the project uses:

- broad window indexing first
- frame-level refinement later

This is the main reason the app can stay responsive enough for Colab-style use.

## 19C. Best Architecture for Natural-Language Timestamped Video Search

The best practical architecture is a staged pipeline.

### Stage 1. Video scan

Read the video once and collect:

- timestamps
- object tracks
- sampled frames
- preview frames

### Stage 2. Semantic indexing

Convert sampled windows into searchable units with:

- frame embeddings
- object tags
- representative frames

### Stage 3. Natural-language search

Convert the query into a text embedding and rank indexed windows by:

- semantic similarity
- object presence clues

### Stage 4. Candidate refinement

For only the top matches:

- trim raw clips
- prepare localization
- prepare segmentation

### Stage 5. Spatial localization

Use a grounding model to answer:

- where is the queried object or phrase in the frame

### Stage 6. Pixel refinement

Use segmentation to answer:

- which exact region should be highlighted

### Stage 7. Delivery

Return:

- timestamps
- matched clip
- segmented preview
- exportable reports

Why this architecture is preferred:

- fast enough to use
- modular
- easier to debug
- easier to upgrade stage by stage
- avoids wasting heavy compute on the entire video

## 19D. Optimization Strategy for This Project

The project should be optimized in a realistic way, not by making false promises.

Correct optimization goal:

- make scan time practical
- make query time short
- make first result appear fast
- make refinement selective

Wrong optimization goal:

- run every heavy model on every frame and still expect lightning-speed output

Current optimization principle:

- cheap broad pass first
- expensive precise pass only on shortlisted matches

This is the only practical way to balance:

- speed
- accuracy
- Colab feasibility
- deployment simplicity

Important truth:

- very fast output and very high accuracy are always a tradeoff
- the project can be improved, but no honest system can guarantee perfect accuracy with instant output on arbitrary videos

## 20. Final Conceptual Summary

This project is best understood as a layered surveillance search system:

- YOLO + ByteTrack answers: what objects are present
- SigLIP2 answers: which windows are semantically similar to the query
- turbovec answers: which indexed segment embeddings should be searched first
- LocateAnything answers: where is the queried thing in the matched frame
- SAM2 answers: what exact region should be highlighted
- Clip/report generation answers: how to deliver the result to the user

That layered design is the reason the system is usable in Colab and still supports natural-language video search with localization and exports.
