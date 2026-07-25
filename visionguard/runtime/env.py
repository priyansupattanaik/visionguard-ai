"""Minimal .env loader shared by launchers without adding an import-time dependency."""
from __future__ import annotations

import os
from pathlib import Path


def load_project_env(root: Path) -> None:
    paths = (root / ".env", root / "configuration" / "provider_keys.env")
    for path in paths:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (key not in os.environ or not os.environ[key]):
                os.environ[key] = value
