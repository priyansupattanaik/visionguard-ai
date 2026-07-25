# VisionGuard

VisionGuard is an offline-first video intelligence application. It scans a video once, detects and tracks objects, builds searchable indexes, and returns timestamped evidence. When semantic models or hosted verification are unavailable, the application identifies the degraded mode instead of presenting unverified output as confirmed.

## Initialize

Use the block that matches the shell shown in your terminal. A prompt beginning with `PS` is PowerShell; do not use the Command Prompt-only `cd /d` syntax there.

PowerShell commands:

```powershell
Set-Location "D:\CDAC PROJECT\visionguard-ai"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
Copy-Item configuration\provider_keys.env.example configuration\provider_keys.env
```

Command Prompt commands:

```bat
cd /d "D:\CDAC PROJECT\visionguard-ai"
py -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
copy configuration\provider_keys.env.example configuration\provider_keys.env
```

The two copy commands are first-time initialization commands. Skip either one when its destination file already exists so that local credentials are not overwritten.

Optional credentials belong in `configuration/provider_keys.env`. NVIDIA visual verification uses `NVIDIA_API_KEY`; gated Hugging Face downloads use `HF_TOKEN`. The credential file is ignored by Git.

## Local llama.cpp provider

VisionGuard defaults to `MODEL_PROVIDER=llama_cpp`. Video upload, metadata, frame extraction, YOLO detection, timestamp mapping, and detector-backed search continue to work when llama.cpp is stopped. The text endpoint adds intent normalization; the optional vision endpoint can verify open visual descriptions.

Start a text server from your llama.cpp directory using the actual GGUF path on your machine:

```bat
llama-server.exe -m models\qwen2.5-7b-instruct-q4_k_m.gguf -c 8192 -ngl 99 --host 127.0.0.1 --port 8080
```

Optional vision server:

```bat
llama-server.exe -m models\vision-model.gguf --mmproj models\vision-mmproj.gguf -c 8192 -ngl 99 --host 127.0.0.1 --port 8081
```

CPU-only text server:

```bat
llama-server.exe -m models\qwen2.5-7b-instruct-q4_k_m.gguf -c 4096 --host 127.0.0.1 --port 8080
```

GGUF and multimodal projector filenames differ by download. Use the real compatible model paths on your machine. Set `MODEL_PROVIDER=none` to disable model-assisted reasoning completely, or explicitly select `nvidia` or `groq` after configuring that provider's URL and key.

## Start

Start the operational video interface:

```text
.\.venv\Scripts\python.exe run.py
```

Open `http://127.0.0.1:7860`, select a bundled video or upload an MP4, and start processing. The query input remains disabled until the backend has written real evidence frames and completed the searchable index.

## Query and indexing behavior

The live query path uses runtime capability discovery, hybrid retrieval, and evidence verification. Object names are read from the active detector model; the application does not maintain an object whitelist or object-alias table. Replacing YOLO with a custom detector therefore exposes that detector's class names without editing query code.

When the configured SigLIP model is cached, frames use open visual-semantic queries. When it is not cached, VisionGuard builds nonzero metadata embeddings from every detector class, colors, appearances, and motion instead of zero vectors. A reachable llama.cpp vision endpoint or selected NVIDIA verifier can check an open query against a bounded sample of indexed frames. Without semantic or vision verification, exact runtime detector classes and indexed metadata remain searchable and the API reports the limitation explicitly.

## Test

```bat
set PYTHONDONTWRITEBYTECODE=1
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
.venv\Scripts\python.exe -m pip check
```

## Evaluate sample videos

```bat
.venv\Scripts\python.exe evaluation\evaluate_sample_videos.py
.venv\Scripts\python.exe evaluation\verify_e2e_workflow.py
```

`verify_e2e_workflow.py` performs a real multipart upload, waits for the real job stages, checks every frame image route and deterministic timestamp, runs a matching query, and confirms that an absent-object query returns insufficient evidence. `verify_browser_workflow.py` additionally drives installed Microsoft Edge through the complete UI; it requires the verification-only `websocket-client` package.

Detector confidence and frame coverage are not accuracy. Precision, recall, and F1 are calculated only after reviewed labels are added to `evaluation/ground_truth.json`.

See `PROJECT_STRUCTURE.md` for every retained folder and `documentation/` for architecture, API, development, and evaluation details.
