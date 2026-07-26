import logging
import os
from pathlib import Path

from visionguard.runtime.env import load_project_env


logger = logging.getLogger(__name__)


def _setup_hf_token():
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if not token:
        return
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_TOKEN", token)
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        logger.warning("Hugging Face token setup failed: %s", exc)


def setup_cache():
    load_project_env(Path.cwd())
    base = os.path.abspath(os.getenv("VISION_GUARD_CACHE_DIR", ".cache/visionguard"))
    _setup_hf_token()
    paths = {
        "HF_HOME": os.path.join(base, "hf"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(base, "hf", "hub"),
        "TORCH_HOME": os.path.join(base, "torch"),
        "YOLO_CONFIG_DIR": os.path.join(base, "ultralytics"),
        "ULTRALYTICS_SETTINGS": os.path.join(base, "ultralytics", "settings.json"),
    }
    for key, path in paths.items():
        os.environ.setdefault(key, path)
    for k in ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "YOLO_CONFIG_DIR"]:
        os.makedirs(os.environ[k], exist_ok=True)


def resolve_device(requested=None):
    """Resolve a safe Torch device, never passing an unavailable CUDA target onward."""
    target = (requested or os.getenv("VISION_GUARD_DEVICE") or "auto").strip().lower()
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if target in {"", "auto"}:
        return "cuda" if cuda_available else "cpu"
    if target == "cuda" and not cuda_available:
        logger.warning("CUDA was requested but unavailable; using CPU.")
        return "cpu"
    return target
