---
title: VisionGuard AI
sdk: gradio
app_file: app.py
pinned: false
---

# VisionGuard AI

Natural-language CCTV search with fast retrieval, Qwen2.5-VL-7B verification, matched clip export, and timestamp records.

## Stack

- `ultralytics` YOLO + ByteTrack for object tracking
- `transformers` SigLIP2 for fast retrieval
- `transformers` Qwen2.5-VL-7B-Instruct for top-k verification
- `gradio` for the UI
- OpenCV for video read/write

## What changed

- Removed the old placeholder search path
- Added Qwen2.5-VL-7B verification on the top candidate clips
- Added multi-clip export, CSV/JSON/HTML reports, and zip download
- Kept indexing fast enough for Colab while using a stronger verifier

## Local run

```bash
pip install -r requirements.txt
python app.py
```

## Google Colab

```python
!git clone https://github.com/priyansupattanaik/visionguard-ai.git
%cd visionguard-ai
!pip install -r requirements.txt
!python app.py
```

Colab will expose a Gradio share link automatically in the notebook output.

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

- CPU Spaces will work but will be slow for long videos.
- Qwen2.5-VL-7B verification is practical on GPU Spaces, not ideal on free CPU Spaces.
- If you want tighter version control, add `sdk_version` in the YAML block at the top of this file.

## Search flow

1. Upload or pick a video.
2. Index it once.
3. Enter a natural-language query.
4. Download the top clips and the timestamp record files.

## Notes

- Default tracking focus can be `all`, `person`, or `vehicle`.
- For long videos, increase `sample every (sec)` to speed up indexing.
- Best accuracy comes from GPU-backed Colab or GPU-enabled Spaces because Qwen2.5-VL-7B is used in the verification stage.
