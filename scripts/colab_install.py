"""Install VisionGuard deps on Google Colab without Gradio/huggingface-hub conflicts.

Colab preinstalls Gradio 6.x, which wants huggingface-hub>=1.2.
VisionGuard uses transformers 4.x (huggingface-hub 0.36.x) and does not use Gradio.
This script removes the conflicting Colab packages, then installs requirements.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, check: bool = True) -> int:
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd)
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.returncode


def main() -> None:
    # Drop Colab UI packages that force an incompatible huggingface-hub range.
    # VisionGuard serves a Flask UI; Gradio is not a runtime dependency.
    # Ignore missing packages (local machines may not have Gradio at all).
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "gradio",
            "gradio-client",
            "safehttpx",
            "groovy",
        ],
        check=False,
    )

    # Keep transformers 4.x on huggingface-hub 0.36.x (explicit pin avoids re-pull of hub 1.x).
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str(ROOT / "requirements.txt"),
            "huggingface-hub>=0.34.0,<1.0",
        ]
    )

    # Prove the stack imports.
    code = (
        "import flask, torch, transformers, ultralytics, cv2, numpy; "
        "import huggingface_hub; "
        "print('flask', flask.__version__); "
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); "
        "print('transformers', transformers.__version__); "
        "print('huggingface_hub', huggingface_hub.__version__); "
        "print('ultralytics', ultralytics.__version__)"
    )
    run([sys.executable, "-c", code])
    print("Colab install OK — Gradio removed; VisionGuard deps installed.")


if __name__ == "__main__":
    main()
