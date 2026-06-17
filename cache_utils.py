import os


def setup_cache():
    base = "/content/drive/MyDrive/visionguard_cache"
    if not os.path.exists("/content/drive/MyDrive"):
        return
    paths = {
        "HF_HOME": os.path.join(base, "hf"),
        "TRANSFORMERS_CACHE": os.path.join(base, "hf", "transformers"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(base, "hf", "hub"),
        "TORCH_HOME": os.path.join(base, "torch"),
        "YOLO_CONFIG_DIR": os.path.join(base, "ultralytics"),
        "ULTRALYTICS_SETTINGS": os.path.join(base, "ultralytics", "settings.json"),
    }
    for key, path in paths.items():
        os.environ.setdefault(key, path)
    for k in ["HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "YOLO_CONFIG_DIR"]:
        os.makedirs(os.environ[k], exist_ok=True)
