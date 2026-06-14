---
title: VisionGuard AI
sdk: gradio
app_file: app.py
pinned: false
---

# VisionGuard AI

Natural-language CCTV search with a scan-first workflow, live indexing preview, matched clip segmentation, per-match export, and timestamp reports.

## Stack

- `ultralytics` `yolo11s.pt` + ByteTrack for live indexing preview and object tracking
- `transformers` `google/siglip2-base-patch16-224` for text-frame retrieval
- `transformers` `microsoft/xclip-base-patch32` for pretrained event tagging on indexed windows
- `transformers` `IDEA-Research/grounding-dino-base` for query grounding
- `transformers` `facebook/sam2.1-hiera-small` for matched clip segmentation
- `gradio` for the UI
- OpenCV for video read/write

## What changed

- Added live indexing preview during scan
- Added Grounding DINO + SAM2 segmentation on matched clips
- Added per-match export with CSV/JSON/HTML/ZIP outputs
- Reduced repeated downloads by supporting Drive-backed cache in Colab
- Removed clip extraction from the initial query path so search returns timestamps first and generates clips only when you open or export a match
- Replaced overlap/motion incident heuristics with pretrained X-CLIP event tagging

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
- GPU Spaces are strongly recommended for Grounding DINO + SAM2.
- If you want tighter version control, add `sdk_version` in the YAML block at the top of this file.

## App Flow

1. Upload or pick a video.
2. Click `step 1: scan video`.
3. Watch live indexing preview until scan completes.
4. Enter a natural-language query.
5. Click `step 2: find matches`.
6. Use `view one match` to switch clips.
7. Export only the clips and reports you want.

## Notes

- Mount Drive in Colab before running if you want model downloads cached between sessions.
- Best experience comes from GPU-backed Colab or GPU-enabled Spaces.
- Event-like queries such as `car accident`, `collision`, `fight`, or `incident` are boosted by pretrained X-CLIP event tags computed during indexing.
- The UI now also writes a short answer block below the query with timestamps and what was detected in each returned match.
