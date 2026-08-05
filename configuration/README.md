# Configuration and API keys

General runtime settings live in the project-root `.env`. Optional provider credentials live in `configuration/provider_keys.env`, which is ignored by Git and loaded automatically.

**Semantic indexing** requires `SEMANTIC_PROVIDER=nvidia` and a live `NVIDIA_API_KEY`. Invalid or unavailable NVIDIA responses fail the index job; there is no silent semantic fallback.

**Optional** `MODEL_PROVIDER` (`none` | `llama_cpp` | `nvidia` | `groq`) controls reasoning / visual verification only. With `MODEL_PROVIDER=none`, detector-backed object retrieval still works after a successful index. `HF_TOKEN` is needed only for gated Hugging Face downloads.

For Google Colab GPU, start from `configuration/colab_fast.env.example` and `documentation/COLAB.md`.
