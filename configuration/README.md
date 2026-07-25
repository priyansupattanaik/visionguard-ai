# Configuration and API keys

General runtime settings live in the project-root `.env`. Optional provider credentials live in `configuration/provider_keys.env`, which is ignored by Git and loaded automatically.

Local llama.cpp is the default reasoning provider and requires no API key. `NVIDIA_API_KEY` and `GROQ_API_KEY` are optional and are read only when their provider is selected. The system continues to upload, extract frames, detect objects, and perform detector-backed retrieval when every language-model provider is unavailable. `HF_TOKEN` is needed only for gated Hugging Face downloads.
