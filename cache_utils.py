import os


def setup_cache():
    if "COLAB_RELEASE_TAG" not in os.environ:
        return
    base = "/content/drive/MyDrive/visionguard_cache"
    if not os.path.exists("/content/drive/MyDrive"):
        return
    os.environ.setdefault("HF_HOME", os.path.join(base, "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(base, "hf", "transformers"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(base, "hf", "hub"))
    os.environ.setdefault("TORCH_HOME", os.path.join(base, "torch"))
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(base, "ultralytics"))
    for k in ["HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "YOLO_CONFIG_DIR"]:
        os.makedirs(os.environ[k], exist_ok=True)
