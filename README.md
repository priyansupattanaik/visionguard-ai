---
title: VisionGuard AI
sdk: gradio
app_file: app.py
pinned: false
---

# VisionGuard AI

Natural-language CCTV search with live indexing preview, matched clip segmentation, per-match export, and timestamp reports.

## Stack

- `ultralytics` YOLO + ByteTrack for live indexing preview
- `transformers` SigLIP2 for retrieval
- `transformers` Grounding DINO for query grounding
- `transformers` SAM2 for matched clip segmentation
- `gradio` for the UI
- OpenCV for video read/write

## What changed

- Added live indexing preview during scan
- Added Grounding DINO + SAM2 segmentation on matched clips
- Added per-match export with CSV/JSON/HTML/ZIP outputs
- Reduced repeated downloads by supporting Drive-backed cache in Colab

## Local run

```bash
pip install -r requirements.txt
python app.py
```

## Google Colab

```python
!git clone https://github.com/priyansupattanaik/visionguard-ai.git
%cd visionguard-ai
from google.colab import drive
drive.mount('/content/drive')
!pip install -r requirements.txt
!python app.py
```

Colab will expose a Gradio share link automatically in the notebook output.
Use a GPU runtime in Colab for the Qwen verification stage.

You can also open the ready notebook:

- `VisionGuard_Colab.ipynb`

## Open In Colab

After pushing the notebook to GitHub, open it with:

`https://colab.research.google.com/github/priyansupattanaik/visionguard-ai/blob/main/VisionGuard_Colab.ipynb`

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

## Search flow

1. Upload or pick a video.
2. Index it once.
3. Enter a natural-language query.
4. Download the top clips and the timestamp record files.

## Notes

- Mount Drive in Colab before running if you want model downloads cached between sessions.
- Best experience comes from GPU-backed Colab or GPU-enabled Spaces.
