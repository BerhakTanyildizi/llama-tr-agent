# -*- coding: utf-8 -*-
"""Reads .env from the repo root.

No python-dotenv dependency on purpose: the whole inference layer runs on the
standard library, and requirements.txt covers training only.

A real environment variable always wins over the file, so a key can be
overridden for one run without editing anything:

    TAVILY_API_KEY=... python inference/orchestrator.py
"""
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # setdefault: the process environment takes precedence over the file.
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load()


def get(name: str, default: str | None = None) -> str | None:
    """Empty strings count as unset - an empty placeholder must not look like a key."""
    return os.environ.get(name) or default
