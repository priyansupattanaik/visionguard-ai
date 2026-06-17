# Vision Guard

A scan-first CCTV video search system with a Gradio interface.
Scan a video once. Search it many times using natural-language queries.
Returns matched frames with timestamps, bounding boxes on top results, and exportable clips.

## How It Works

1. Upload a video or scan a sample clip.
2. The system samples frames, prunes near-duplicate/static frames, detects objects, and builds a searchable index.
3. Enter a natural-language query and click Find Matches.
4. The top matched frames are shown in the Gradio UI with timestamps and export selectors.
5. Export selected clips, CSV, JSON, or HTML reports.

## Model Stack

| Role | Model |
|---|---|
| Detection + Tracking | YOLO11n + BoT-SORT |
| Retrieval | SigLIP2 So400m/14 384 |
| Verification + Grounding | Qwen2.5-VL-7B-Instruct-AWQ |
| Segmentation | SAM2.1-hiera-small |
| Vector Index | turbovec IdMapIndex |

## What Works Well

- Natural-language search over a scanned video
- Repeated queries without rescanning
- Object-aware retrieval (person, car, truck, bus, motorcycle, bicycle, umbrella)
- Open-world query verification for objects and visible events outside fixed detector labels
- Color-object queries through the same open-language verification path
- Open-world query verification on top matches at search time
- Bounding boxes drawn on matched frames when the query can be grounded
- Exact frame re-selection within matched windows
- Segmented clip and region mask on export
- Fast shortlist retrieval with turbovec after scan-time indexing is complete
- Lower default scan density with cheap redundancy pruning before heavy models
- SigLIP2 frame embedding uses larger CUDA batches and guarded `torch.compile` on the vision tower
- Long scans accumulate vectors in chunks before the final turbovec build
- Background warm-up loads the heavy models as the app starts
- Query verification is cached by stable frame key plus normalized query
- Confirmed matches stream into the UI as they arrive
- Qwen2.5-VL uses an AWQ model by default and can use `vllm` automatically when available

## Known Limitations

- Temporal events (collisions, fights, falls) are not guaranteed to be correctly identified.
  The system verifies shortlisted frames with Qwen2.5-VL, but difficult event understanding is still model-limited.
- The default scan interval is now coarser than before. This is faster, but ultra-short events have a slightly higher chance of being missed.
- Queries shorter than 3 words or very abstract queries (unusual activity, suspicious thing)
  will return weaker results.
- Very long videos increase scan time linearly. This is expected behavior.
- First-run startup is slower because YOLO11n, SigLIP2, Qwen2.5-VL-7B, and SAM2 checkpoints must load before the full stack becomes responsive.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Open:

- local Windows/Linux/macOS: [http://127.0.0.1:7860](http://127.0.0.1:7860)
- Colab: use the notebook port window opened by `output.serve_kernel_port_as_window(7860)`

## Running on Colab

```python
from google.colab import drive, output, userdata
drive.mount('/content/drive')

import os
import subprocess
import time
import urllib.request

base = "/content/drive/MyDrive/visionguard_cache"
paths = {
    "HF_HOME": f"{base}/hf",
    "TRANSFORMERS_CACHE": f"{base}/hf/transformers",
    "HUGGINGFACE_HUB_CACHE": f"{base}/hf/hub",
    "TORCH_HOME": f"{base}/torch",
    "YOLO_CONFIG_DIR": f"{base}/ultralytics",
    "ULTRALYTICS_SETTINGS": f"{base}/ultralytics/settings.json",
}
for key, value in paths.items():
    os.environ[key] = value
for key in ["HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "YOLO_CONFIG_DIR"]:
    os.makedirs(os.environ[key], exist_ok=True)

hf_token = userdata.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

if not os.path.exists("/content/visionguard-ai"):
    !git clone https://github.com/priyansupattanaik/visionguard-ai.git /content/visionguard-ai
%cd /content/visionguard-ai
!git pull
!pip install -r requirements.txt

os.environ["VISION_GUARD_HOST"] = "0.0.0.0"
os.environ["GRADIO_SHARE"] = "0"
proc = subprocess.Popen(
    ["python", "app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

deadline = time.time() + 300
ready = False
while time.time() < deadline:
    line = proc.stdout.readline()
    if line:
        print(line, end="")
    try:
        with urllib.request.urlopen("http://127.0.0.1:7860/", timeout=3) as r:
            if r.status == 200:
                ready = True
                break
    except Exception:
        time.sleep(1)

if not ready:
    raise RuntimeError("Vision Guard did not become reachable on port 7860.")

print("App Ready")
output.serve_kernel_port_as_window(7860)
```

Use a GPU runtime in Colab for the current default stack.
The first run downloads the models once. If Drive is mounted, the project cache helper keeps Hugging Face and Torch caches in Drive so later Colab sessions reuse them.

If the old `proxyPort(...)` URL shows a 404 page, use `output.serve_kernel_port_as_window(7860)` instead. That is the correct Colab launch method for this notebook flow.

To remove unauthenticated Hugging Face Hub warnings and get better rate limits, set a valid `HF_TOKEN` before launch.

Example in Colab:

```python
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

Or set it manually:

```python
os.environ["HF_TOKEN"] = "hf_your_token_here"
```
