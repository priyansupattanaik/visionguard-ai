# Headroom Integration Notes

This folder is intentionally isolated from the main Vision Guard runtime.

If you do not want Headroom later, you can remove this entire folder without affecting:

- the Gradio app
- the Colab notebook
- the scan pipeline
- the retrieval pipeline
- clip export
- reports

## Status

Headroom is **not active by default** in this repository.

Current project context snapshot for optional Headroom compression or agent handoff:

- [VISION_GUARD_CONTEXT.md](D:/CDAC_PROJECT/CV_Project/optional_integrations/headroom/VISION_GUARD_CONTEXT.md)

The current app does **not** import Headroom anywhere in:

- `app.py`
- `pipeline.py`
- `qwen_verifier.py`
- `segmenter.py`
- `tracker.py`
- `vlm.py`

That means the current project behavior is unchanged.

## Why Keep It Separate

Headroom is an external context-compression layer.
It is useful for agent workflows, long logs, tool outputs, and multi-agent context passing.

It is **not** part of the video-search runtime itself.

Keeping it separate avoids:

- breaking Colab execution
- changing model behavior silently
- adding hidden runtime dependencies
- mixing optional agent tooling with the core app

## What Headroom Could Be Used For Here

Possible future uses:

1. Compress long generated `index.json` or report payloads before sending them to an external agent.
2. Compress large tool outputs during debugging sessions.
3. Compress multi-run metadata when building a future RAG or agent layer on top of Vision Guard.
4. Share compressed context across external coding-agent workflows.

## What It Should Not Be Used For

Do not treat Headroom as:

- a replacement for SigLIP2 retrieval
- a replacement for YOLO detection
- a replacement for Qwen verification
- a replacement for SAM2 segmentation
- a direct accuracy improvement for frame matching

Headroom helps with context size and token efficiency.
It does not perform the actual CCTV understanding pipeline.

## Safe Integration Modes

Headroom supports multiple modes according to its public README:

- library mode
- proxy mode
- MCP mode
- agent wrapper mode

For this project, the safest future integration path is:

1. keep Vision Guard unchanged
2. run Headroom outside the app
3. use it only when sending large outputs to another agent or LLM layer

## Recommended Future Path

If you decide to use Headroom later, prefer one of these:

### Option A. External agent workflow only

Use Headroom for:

- documentation compression
- report compression
- debugging/log compression

Do not modify the app runtime.

### Option B. Optional post-processing helper

Create a separate helper script later that:

1. reads a generated report
2. compresses it with Headroom
3. sends it to an external LLM workflow

This keeps the main app stable.

## Minimal Future Installation Example

This is intentionally documentation only.
It is not wired into the current project.

```bash
pip install "headroom-ai[all]"
```

Possible external usage model:

```python
from headroom import compress

messages = [
    {"role": "user", "content": "Summarize this Vision Guard report."},
    {"role": "user", "content": open("output/example/reports/index.json", "r", encoding="utf-8").read()},
]

compressed = compress(messages)
```

## Removal

To remove this optional integration scaffold later, delete:

- `optional_integrations/headroom/`

No other project file needs to be changed unless you later wire it into runtime code.
