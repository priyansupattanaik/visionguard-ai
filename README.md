# Vision Guard

A scan-first CCTV video search system.
Scan a video once. Search it many times using natural-language queries.
Returns matched frames with timestamps, bounding boxes on top results, and exportable clips.

## How It Works

1. Upload a video and click Scan.
2. The system samples frames, detects objects, and builds a searchable index.
3. Enter a natural-language query and click Find Matches.
4. The top matched frame is shown with a grounding box if the queried phrase can be located.
5. Export selected clips, CSV, JSON, or HTML reports.

## Model Stack

| Role | Model |
|---|---|
| Detection + Tracking | YOLO11m + BoT-SORT |
| Retrieval | SigLIP2 So400m/14 384 |
| Verification + Grounding | LocateAnything-3B |
| Segmentation | SAM2.1-hiera-small |
| Vector Index | turbovec IdMapIndex |

## What Works Well

- Natural-language search over a scanned video
- Repeated queries without rescanning
- Object-aware retrieval (person, car, truck, bus, motorcycle, bicycle, umbrella)
- Color-object queries for supported vehicle classes (yellow car, white bus)
- Grounded query verification on top matches at search time
- Bounding boxes drawn on matched frames when the query can be grounded
- Exact frame re-selection within matched windows
- Segmented clip and region mask on export
- Fast shortlist retrieval with turbovec after scan-time indexing is complete

## Known Limitations

- Temporal events (collisions, fights, falls) are not guaranteed to be correctly identified.
  The system retrieves visually similar frames, not verified event classifications.
- Queries shorter than 3 words or very abstract queries (unusual activity, suspicious thing)
  will return weaker results.
- Very long videos increase scan time linearly. This is expected behavior.
- First-run startup is slower because YOLO11m, SigLIP2, LocateAnything-3B, and SAM2 checkpoints must load before the full stack becomes responsive.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Running on Colab

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

Use a GPU runtime in Colab for the current default stack.
The first run downloads the models once. If Drive is mounted, the project cache helper keeps Hugging Face and Torch caches in Drive so later Colab sessions reuse them.
