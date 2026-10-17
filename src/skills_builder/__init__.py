"""Measure how agent skills improve an open-weights agent on tau2-bench retail."""

# tau2 resolves its data directory once, on first import. Load .env and set
# TAU2_DATA_DIR here so any import of this package runs before tau2's.
from skills_builder.config import load_env as _load_env

_load_env()
