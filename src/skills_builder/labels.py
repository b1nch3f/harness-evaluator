"""Intent labels, width and depth for tau2 retail tasks.

Definitions are in docs/01-experiment-protocol.md:
- one label per expected write action type, plus `transfer` and `answer`
- width = number of distinct write intents (tau2 ToolType.WRITE tools)
- depth = number of expected actions (reads and writes)
"""

from __future__ import annotations

import json
from pathlib import Path

from skills_builder import data
from skills_builder.config import REPO_ROOT

LABELS_PATH = REPO_ROOT / "data" / "labels" / "retail.json"

# tau2 retail tool name -> intent label. The first seven are WRITE tools in
# tau2.domains.retail.tools; transfer_to_human_agents is GENERIC.
WRITE_INTENTS = {
    "cancel_pending_order": "cancel",
    "modify_pending_order_items": "modify_items",
    "modify_pending_order_address": "modify_order_address",
    "modify_pending_order_payment": "modify_payment",
    "return_delivered_order_items": "return",
    "exchange_delivered_order_items": "exchange",
    "modify_user_address": "modify_user_address",
}
TRANSFER_TOOL = "transfer_to_human_agents"

ALL_LABELS = [*WRITE_INTENTS.values(), "answer", "transfer"]


def label_task(task: dict) -> dict:
    """Labels for one task (a tasks.json entry)."""
    criteria = task.get("evaluation_criteria") or {}
    actions = criteria.get("actions") or []
    names = [a["name"] for a in actions]
    write_labels = sorted({WRITE_INTENTS[n] for n in names if n in WRITE_INTENTS})
    labels = list(write_labels)
    if TRANSFER_TOOL in names:
        labels.append("transfer")
    if criteria.get("communicate_info"):
        labels.append("answer")
    return {
        "labels": sorted(labels),
        "width": len(write_labels),
        "depth": len(actions),
    }


def build_labels() -> dict:
    """Labels for every retail task, from tau2's data files."""
    split_of = {}
    for name in ("train", "test"):
        for tid in data.splits()[name]:
            split_of[tid] = name
    out = {}
    for task in sorted(data.tasks(), key=lambda t: int(t["id"]) if t["id"].isdigit() else t["id"]):
        out[task["id"]] = {"split": split_of[task["id"]], **label_task(task)}
    return out


def write_labels(path: Path = LABELS_PATH) -> dict:
    labels = build_labels()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2) + "\n")
    return labels


def load_labels(path: Path = LABELS_PATH) -> dict:
    return json.loads(path.read_text())


def label_counts(labels: dict, split: str) -> dict[str, int]:
    counts = {name: 0 for name in ALL_LABELS}
    for row in labels.values():
        if row["split"] == split:
            for name in row["labels"]:
                counts[name] += 1
    return counts
