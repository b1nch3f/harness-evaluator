"""Settings loaded from .env.

Load .env before importing tau2, because tau2 reads API keys from the
environment at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

# tau2's data ships in its repo, not its wheel. The repo is a git submodule
# pinned to the same commit as the installed tau2 package.
TAU2_COMMIT = "96a3ef92329baf908ffe9dfe188cea34e8701d2a"
TAU2_REPO_DIR = REPO_ROOT / "third_party" / "tau2-bench"


def load_env() -> None:
    """Load the repo's .env into os.environ (existing variables win)."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    os.environ.setdefault("TAU2_DATA_DIR", str(TAU2_REPO_DIR / "data"))


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default
