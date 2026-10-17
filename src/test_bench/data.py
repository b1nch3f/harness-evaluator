"""tau2-bench data files, read straight from the pinned submodule (no tau2 code)."""

from __future__ import annotations

import json
from functools import lru_cache

from test_bench.config import TAU2_REPO_DIR

DATA_DIR = TAU2_REPO_DIR / "data" / "tau2"
RETAIL_DIR = DATA_DIR / "domains" / "retail"
DB_PATH = RETAIL_DIR / "db.json"
TASKS_PATH = RETAIL_DIR / "tasks.json"
SPLITS_PATH = RETAIL_DIR / "split_tasks.json"
POLICY_PATH = RETAIL_DIR / "policy.md"
USER_GUIDELINES_PATH = DATA_DIR / "user_simulator" / "simulation_guidelines.md"


@lru_cache(maxsize=1)
def _db_text() -> str:
    return DB_PATH.read_text()


def db_json() -> dict:
    """A fresh, independent copy of the retail database as plain JSON."""
    return json.loads(_db_text())


@lru_cache(maxsize=1)
def tasks() -> list[dict]:
    return json.loads(TASKS_PATH.read_text())


def tasks_by_id() -> dict[str, dict]:
    return {t["id"]: t for t in tasks()}


@lru_cache(maxsize=1)
def splits() -> dict[str, list[str]]:
    return json.loads(SPLITS_PATH.read_text())


@lru_cache(maxsize=1)
def policy() -> str:
    return POLICY_PATH.read_text()


@lru_cache(maxsize=1)
def user_guidelines() -> str:
    return USER_GUIDELINES_PATH.read_text()
