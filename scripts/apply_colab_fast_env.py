"""Apply the Colab fast-GPU environment profile for VisionGuard.

Usage (Colab or local):
  python scripts/apply_colab_fast_env.py --nvidia-key "$NVIDIA_API_KEY"
  python scripts/apply_colab_fast_env.py --from-env   # read NVIDIA_API_KEY already set
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configuration" / "colab_fast.env.example"


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvidia-key", default="", help="NVIDIA API key (nvapi-...)")
    parser.add_argument(
        "--from-env",
        action="store_true",
        help="Use NVIDIA_API_KEY already present in the process environment",
    )
    parser.add_argument(
        "--write-dotenv",
        action="store_true",
        help="Write project .env (overwrites existing .env)",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    args = parser.parse_args()

    if not EXAMPLE.is_file():
        raise SystemExit(f"Missing profile: {EXAMPLE}")

    profile = parse_env_file(EXAMPLE)
    profile["VISION_GUARD_DEVICE"] = args.device

    key = (args.nvidia_key or "").strip()
    if not key and args.from_env:
        key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "NVIDIA_API_KEY is required. Pass --nvidia-key or set the env var with --from-env."
        )
    profile["NVIDIA_API_KEY"] = key

    for name, value in profile.items():
        os.environ[name] = value

    if args.write_dotenv:
        lines = [f"{k}={v}" for k, v in profile.items()]
        (ROOT / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {ROOT / '.env'}")

    # Ensure model/cache dirs exist under the project root.
    for rel in (profile.get("VISION_GUARD_MODEL_DIR", ".models"), profile.get("VISION_GUARD_CACHE_DIR", ".cache/visionguard")):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)

    print("Colab fast-GPU profile applied:")
    for name in (
        "VISION_GUARD_DEVICE",
        "YOLO_MODEL",
        "YOLO_IMGSZ",
        "IMAGE_BATCH_SIZE",
        "WIN_SEC",
        "SEMANTIC_WORKERS",
        "MIN_EVIDENCE_CONFIDENCE",
        "MODEL_PROVIDER",
        "ENABLE_CROP_EMBEDDINGS",
    ):
        print(f"  {name}={os.environ.get(name)}")
    print("  NVIDIA_API_KEY=***set***")


if __name__ == "__main__":
    main()
