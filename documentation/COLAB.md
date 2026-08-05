# Run VisionGuard on Google Colab (GPU, fast, evidence-first)

This is the supported path for using a Colab GPU without inventing unsupported shortcuts.

## What “fast” and “no hallucinations” mean here

| Goal | How this profile achieves it | Honest limit |
|------|------------------------------|--------------|
| Fast | GPU YOLO nano, larger `WIN_SEC`, parallel NVIDIA semantic workers, no crop embeddings, short video | Long CCTV still costs full-frame decode |
| No invented objects | Object queries use detector evidence only; absent queries must abstain | YOLO can still false-positive |
| No free-form LLM answers | `MODEL_PROVIDER=none` | NVIDIA captions still exist as **unverified** `semantic_description` |
| Inspectable | Every hit has timestamps + stored JPEG frames | Not the same as human-labeled accuracy |

**There is no configuration that guarantees zero ML error.** VisionGuard’s contract is: *evidence-linked answers or explicit abstention*, never silent verified fiction.

## Prerequisites

1. Colab runtime: **Runtime → Change runtime type → GPU** (T4 is enough for short samples).
2. [NVIDIA API key](https://build.nvidia.com/) for the mandatory semantic index stage.
3. Project source on the machine (clone or upload zip). Keep `sample_videos/`.

## One-command E2E (recommended)

In a Colab notebook:

```python
# 1) Mount/upload repo, then:
%cd /content/visionguard-ai

# 2) Secrets → NVIDIA_API_KEY
from google.colab import userdata
import os
os.environ["NVIDIA_API_KEY"] = userdata.get("NVIDIA_API_KEY")

# 3) Install (removes Colab's preinstalled Gradio, which conflicts with transformers 4.x)
!apt-get -qq update && apt-get -qq install -y libgl1 libglib2.0-0 > /dev/null
!python scripts/colab_install.py

# 4) Fast GPU end-to-end (asset3 ~12s is the speed demo)
!python scripts/run_colab_e2e.py --video sample_videos/asset3.mp4 --query "find the person" --absent-query "find the elephant"
```

### Gradio / huggingface-hub conflict

Colab ships Gradio 6.x (`huggingface-hub>=1.2`). VisionGuard uses **transformers 4.x** (`huggingface-hub<1`). VisionGuard does **not** use Gradio (Flask UI only). Always install with:

```bash
python scripts/colab_install.py
```

Do **not** use bare `pip install -r requirements.txt` on Colab without uninstalling Gradio first.


Expected result:

- `all_passed: true`
- present query returns frames
- absent query has `insufficient_evidence: true`
- report at `output/colab_e2e_report.json`

## Fast profile knobs (`configuration/colab_fast.env.example`)

| Variable | Fast default | Why |
|----------|--------------|-----|
| `VISION_GUARD_DEVICE` | `cuda` | YOLO + SigLIP on GPU |
| `YOLO_MODEL` | `yolo11n.pt` | Fastest Ultralytics profile |
| `YOLO_IMGSZ` | `512` | Fewer pixels / frame |
| `YOLO_CONF` / `MIN_EVIDENCE_CONFIDENCE` | `0.28` / `0.35` | Fewer weak detections as “evidence” |
| `IMAGE_BATCH_SIZE` | `16` | GPU embedding throughput |
| `ENABLE_CROP_EMBEDDINGS` | `0` | Avoids per-box SigLIP cost |
| `WIN_SEC` | `8.0` | Fewer segments ⇒ fewer NVIDIA calls |
| `SEMANTIC_WORKERS` | `6` | Parallel I/O-bound semantic API calls |
| `MODEL_PROVIDER` | `none` | No optional reasoning LLM |
| `VISION_GUARD_MAX_DURATION_SECONDS` | `90` | Hard cap for Colab sessions |

Apply only:

```bash
python scripts/apply_colab_fast_env.py --from-env --write-dotenv --device cuda
python scripts/bootstrap_models.py --yolo yolo11n.pt
```

## Grounded queries (use these for anti-hallucination)

**Safe (detector-fact routes)**

- `find the person`
- `find the car`
- `how many people`
- time filters the planner supports (`before` / `after` / `between` when expressed as in the query rules)

**Avoid if you need strict grounding**

- open scene poetry (“describe the mood”)
- attributes without detector support (“red helmet”, “suspicious”)
- speech / OCR / identity (“what did they say”, “who is that”)
- events the graph does not emit (`fall`, `fight`, …) — system should **abstain**

NVIDIA captions are always `claim_provenance=nvidia_semantic` / `evidence_state=semantic_description`. Do not treat them as verified facts.

## Optional browser UI

After the E2E script passes:

```python
import os, threading
from pathlib import Path
from visionguard.runtime.env import load_project_env
load_project_env(Path("/content/visionguard-ai"))
os.environ["VISION_GUARD_HOST"] = "0.0.0.0"

from visionguard.web_app.server import app
threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=7860, debug=False, threaded=True, load_dotenv=False),
    daemon=True,
).start()

# Expose with ngrok (optional)
# !pip -q install pyngrok
# from pyngrok import ngrok
# from google.colab import userdata
# ngrok.set_auth_token(userdata.get("NGROK_AUTHTOKEN"))
# print(ngrok.connect(7860).public_url)
```

UI flow: upload or pick asset → **start indexing** (upload alone does not index) → wait until searchable → query.

## Speed expectations (order of magnitude)

On Colab T4 with `asset3.mp4` (~12s, small resolution), after models are cached:

- first run: slower (YOLO + SigLIP download)
- later runs: typically a few minutes total depending on NVIDIA API latency

`asset1.mp4` (~5 minutes) is **not** a “very fast” demo; use `asset2` / `asset3` first.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| No GPU | Runtime → GPU; re-check `torch.cuda.is_available()` |
| `decord` install fails | Retry `pip install decord`; keep Linux Colab (not local Windows copy of the issue) |
| SigLIP missing | Ensure `VISION_GUARD_LOCAL_MODELS_ONLY=0` on first run |
| YOLO missing | `python scripts/bootstrap_models.py --yolo yolo11n.pt` |
| Semantic stage fails | Valid `NVIDIA_API_KEY`, model name, and network; raise `NVIDIA_API_TIMEOUT` |
| Present query empty | Try another sample (`asset2`); raise nothing—may be true miss at higher conf |
| Absent query returns frames | Bug / too-loose retrieval; fail the run (script does) |
| Disk full | Delete `output/*/frames` or restart runtime |

## Security

Store `NVIDIA_API_KEY` in **Colab Secrets**, not in the notebook body or git. Rotate any key that was ever committed or shared.
