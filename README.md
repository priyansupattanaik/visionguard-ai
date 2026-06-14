---
title: VisionGuard AI
sdk: gradio
app_file: app.py
pinned: false
---

# VisionGuard AI

Natural-language search for CCTV video with matched clip export, timestamp records, and a Gradio UI that can run in Google Colab or Hugging Face Spaces.

## Stack

- `ultralytics` YOLO + ByteTrack for object tracking
- `transformers` CLIP for text-image retrieval
- `gradio` for the UI
- OpenCV for video read/write

## What changed

- Removed the per-frame 3B VLM loop
- Replaced placeholder string search with embedding search
- Added multi-clip export, CSV/JSON/HTML reports, and zip download
- Kept the code light enough for Colab and realistic for Spaces

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

## Hugging Face Spaces

Push this repo to a Gradio Space. The YAML block at the top of this `README.md` is the Spaces config. Spaces installs `requirements.txt` and starts `app.py`.

## Search flow

1. Upload or pick a video.
2. Index it once.
3. Enter a natural-language query.
4. Download the top clips and the timestamp record files.

## Notes

- Default tracking focus can be `all`, `person`, or `vehicle`.
- For long videos on CPU, increase `sample every (sec)` to speed up indexing.
- Best accuracy comes from GPU-backed Colab or GPU-enabled Spaces.
