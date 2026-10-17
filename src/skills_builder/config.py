"""Settings loaded from .env.

Load .env before importing tau2, because tau2 reads API keys from the
environment at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Load the repo's .env into os.environ (existing variables win)."""
    load_dotenv(REPO_ROOT / ".env", override=False)


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default
