"""Download a requested model into VisionGuard's ignored local model cache."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo", default="yolo11m.pt", help="Ultralytics YOLO checkpoint to cache")
    parser.add_argument("--model-dir", type=Path, default=Path(".models"))
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    destination = args.model_dir / Path(args.yolo).name
    if destination.is_file():
        print(destination)
        return
    settings_dir = Path(".cache") / "visionguard" / "ultralytics"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir.resolve()))
    from ultralytics import YOLO

    model = YOLO(args.yolo)
    source = Path(getattr(model, "ckpt_path", args.yolo) or args.yolo)
    if not source.is_file():
        raise RuntimeError(f"Ultralytics did not create the requested checkpoint: {args.yolo}")
    shutil.copy2(source, destination)
    print(destination)


if __name__ == "__main__":
    main()
