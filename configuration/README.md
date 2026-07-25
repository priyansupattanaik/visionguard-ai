# Configuration and API keys

General runtime settings live in the project-root `.env`. Optional provider credentials live in `configuration/provider_keys.env`, which is ignored by Git and loaded automatically.

Add an NVIDIA key as `NVIDIA_API_KEY=...` to enable the currently implemented hosted visual verifier. The system runs without it and labels results as detector/retrieval evidence rather than verified evidence. `HF_TOKEN` is needed only for gated Hugging Face downloads. No other hosted provider is currently wired, so adding unrelated keys has no effect.
