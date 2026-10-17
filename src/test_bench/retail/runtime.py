"""How tool results are turned into text."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


def to_plain(obj: Any) -> Any:
    """Records become JSON-ready dicts; everything else is left as is."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    return obj


def result_text(obj: Any) -> str:
    """A tool result as text: strings as-is, records as compact JSON (as tau2 does)."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    return json.dumps(to_plain(obj), default=str)
