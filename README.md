---
title: VisionGuard AI
sdk: gradio
app_file: app.py
pinned: false
---

# VisionGuard AI

Natural-language CCTV search with a scan-first workflow, live indexing preview, matched-frame results, on-demand export, and timestamp reports.

Full project documentation:

- [PROJECT_DOCUMENTATION.md](/D:/CDAC_PROJECT/CV_Project/PROJECT_DOCUMENTATION.md)

## Stack

- `ultralytics` `yolo11m.pt` + BoT-SORT for live indexing preview and object tracking
- `transformers` `google/siglip2-so400m-patch14-384` for text-frame retrieval
- `turbovec` `IdMapIndex` for persisted frame and segment embedding search
- `transformers` `microsoft/Florence-2-large` for top-k frame verification, description, and grounding
- `transformers` `facebook/sam2.1-hiera-small` for matched clip segmentation
- `gradio` for the UI
- OpenCV for video read/write

## What changed

- Added live indexing preview during scan
- Upgraded scan-time detector/tracker to `yolo11m.pt` + BoT-SORT
- Upgraded retrieval model to `google/siglip2-so400m-patch14-384`
- Switched grounding and verification to `microsoft/Florence-2-large`
- Added `turbovec` as the primary frame/segment retriever with NumPy fallback
- Added frame-first retrieval plus `Florence-2` top-k verification for better timestamp precision
- Added per-match export with CSV/JSON/HTML/ZIP outputs
- Reduced repeated downloads by supporting Drive-backed cache in Colab
- Removed clip extraction from the initial query path so search returns matched frames and timestamps first and generates clips only when you export selected matches
- Removed noisy event-tag injection from runtime search to reduce false positives
- Added query normalization plus color-aware vehicle matching for queries such as `yellow car` or `white bus`
- Added detector-backed query refinement for detectable objects such as `person`, `car`, and `umbrella`, so matched frames prefer real object hits over weak semantic fallbacks
- Added boxed matched-frame previews in the gallery and skipped obvious low-content intro/title-card frames during scan

## Local run

```bash
pip install -r requirements.txt
python app.py
```

## Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
import os
if not os.path.exists("/content/visionguard-ai"):
    !git clone https://github.com/priyansupattanaik/visionguard-ai.git /content/visionguard-ai
%cd /content/visionguard-ai
!git pull
!pip install -r requirements.txt
!python app.py
```

Colab will expose a Gradio share link automatically in the notebook output.
Use a GPU runtime in Colab for the current default stack.
The first run downloads the models once. If Drive is mounted, the project cache helper keeps Hugging Face and Torch caches in Drive so later Colab sessions reuse them.

You can also open the ready notebook:

- `VisionGuard_Colab.ipynb`

## Open In Colab

After pushing the notebook to GitHub, open it with:

`https://colab.research.google.com/github/priyansupattanaik/visionguard-ai/blob/main/VisionGuard_Colab.ipynb`

## Faster Update Flow In Colab

Do not delete the repo folder every time.

```python
import os
if not os.path.exists("visionguard-ai"):
    !git clone https://github.com/priyansupattanaik/visionguard-ai.git
%cd visionguard-ai
!git pull
!pip install -r requirements.txt
!python app.py
```

This keeps the repo folder and the Drive-backed model cache, so only your code updates are pulled.

## Hugging Face Spaces

Push this repo to a Gradio Space. The YAML block at the top of this `README.md` is the Spaces config. Spaces installs `requirements.txt` and starts `app.py`.

### Space steps

1. Create a new Space on Hugging Face.
2. Choose `Gradio` as the SDK.
3. Connect the GitHub repo or upload the repo files.
4. Keep `app.py` as the entry file.
5. Let Spaces install `requirements.txt`.
6. Open the Space once the build finishes.

### Notes

- CPU Spaces will work but will be slow for segmentation.
- GPU Spaces are strongly recommended for Florence-2-large + SAM2.
- If you want tighter version control, add `sdk_version` in the YAML block at the top of this file.

## App Flow

1. Upload or pick a video.
2. Click `step 1: scan video`.
3. Watch live indexing preview until scan completes.
4. Enter a natural-language query.
5. Click `step 2: find matches`.
6. Review the matched-frame gallery and timestamp table.
7. Export only the clips and reports you want.

## Notes

- Mount Drive in Colab before running if you want model downloads cached between sessions.
- Best experience comes from GPU-backed Colab or GPU-enabled Spaces.
- Search now uses frame-first retrieval with a higher-capacity SigLIP2 encoder, object-aware reranking, and `Florence-2-large` verification on top candidates.
- Vehicle queries with color words now use coarse per-frame appearance tags, so `yellow car` is not treated the same as any generic `car`.
- Frame and segment embeddings are persisted to local `turbovec` indexes per scan, so repeated queries use the vector index instead of rescoring everything in Python.
- The UI writes a short answer block below the query and shows matched sampled frames immediately.
- The matched-frame gallery now uses boxed preview frames when detector-backed matches are available.
- If no strong semantic match is found, the app now falls back to nearest object-backed or low-confidence frame matches instead of showing a blank result area.
