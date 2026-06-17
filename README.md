# Vision Guard

Vision Guard is a scan-first CCTV video search app with a Gradio UI.
It scans a video once, builds a searchable index, and then lets you ask natural-language questions about that video.

For the full architecture, runtime flow, model stack, limitations, and Colab instructions, read:

- [PROJECT_DOCUMENTATION.md](D:/CDAC_PROJECT/CV_Project/PROJECT_DOCUMENTATION.md)

Optional integrations are kept separate from the core runtime:

- [optional_integrations/headroom/README.md](D:/CDAC_PROJECT/CV_Project/optional_integrations/headroom/README.md)

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open:

- local: `http://127.0.0.1:7860`
- Colab: the `gradio.live` URL printed by `python -u app.py`

## What The App Does

1. Upload a video or choose a sample clip.
2. Click `step 1: scan video`.
3. Wait for the scan to finish.
4. Enter a natural-language query.
5. Click `step 2: find matches`.
6. Review the matched frames and timestamps.
7. Export only the clips and reports you want.

## Current Stack

- UI: Gradio
- Video reading: Decord
- Detection + tracking: YOLO11n + BoT-SORT
- Retrieval: SigLIP2
- Verification + grounding: Qwen2.5-VL-7B-Instruct-AWQ
- Segmentation: SAM2.1-hiera-small
- Vector search: turbovec

## Colab Run

The notebook is available at [VisionGuard_Colab.ipynb](D:/CDAC_PROJECT/CV_Project/VisionGuard_Colab.ipynb).

The intended Colab flow is:

1. mount Google Drive
2. clone or pull the repo
3. install dependencies
4. set `GRADIO_SHARE=1`
5. run `python -u app.py`
6. open the printed `gradio.live` URL

## Notes

- The first run is slow because model weights need to load.
- Drive-backed caching is used so later Colab sessions can reuse downloads.
- Temporal events like collision, fight, and fall are not guaranteed to be perfect.
- Headroom is documented as an optional external context-compression layer only. It is not active in the default app runtime.
