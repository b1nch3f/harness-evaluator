"""Push one finished eval run to Weights & Biases: overall pass@k / pass^k plus its configuration.

Only run-level numbers go to W&B; per-task results and transcripts stay local.
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse


def _endpoint_label(api_base: str | None) -> str | None:
    """Host only (e.g. api.runpod.ai); never the full URL."""
    if not api_base:
        return None
    url = api_base if "://" in api_base else "https://" + api_base
    return urlparse(url).hostname


def _models(record: dict) -> dict:
    return {
        role: {
            "model": m.get("model"),
            "endpoint": _endpoint_label(m.get("api_base")),
            **{k: m.get(k) for k in ("temperature", "top_p", "reasoning_effort", "max_tokens", "timeout") if k in m},
        }
        for role, m in record["models"].items()
    }


def wandb_config(record: dict) -> dict:
    h = record["harness"]
    eps = record.get("episodes", {})
    return {
        "harness": h["name"],
        "harness_version": h["version"][:12],
        "harness_commit": h.get("git_commit"),
        "prompt_sha256": h.get("prompt_sha256"),
        "skills_sha256": h.get("skills_sha256"),
        "uses_policy": h.get("uses_policy"),
        "skills": h.get("skills", []),
        "test_bench_sha": record["scaffolding"]["git_sha"],
        "tau2_data_commit": record["tau2_data_commit"],
        "settings": record["settings"],
        "models": _models(record),
        "tasks": len(record["task_ids"]),
        "episodes": eps,
        "seconds": record.get("seconds"),
        "started_at": record["started_at"],
        "local_run_id": record["run_id"],
    }


def wandb_rows(record: dict) -> tuple[list[dict], dict]:
    """Per-k rows (for charts with k on the x axis) and summary values, overall only."""
    rows, summary = [], {}
    for split, s in record["scores"].items():
        o = s["overall"]
        ks = sorted(int(m.group(1)) for key in o if (m := re.fullmatch(r"pass\^(\d+)", key)))
        for k in ks:
            row = {"k": k, f"{split}/pass^k": o[f"pass^{k}"], f"{split}/pass@k": o[f"pass@{k}"]}
            rows.append(row)
            for name in (f"pass^{k}", f"pass@{k}"):
                summary[f"{split}/{name}"] = o[name]
                lo, hi = o.get(f"{name}_ci") or (None, None)
                summary[f"{split}/{name}_ci_low"], summary[f"{split}/{name}_ci_high"] = lo, hi
        summary[f"{split}/tasks"] = o["tasks"]
    return rows, summary


X_AXIS = "k (trials)"


def labelled_charts(rows: list[dict], wandb) -> dict:
    """Line charts whose axis titles are the table's column names (W&B's default panels have none)."""
    charts = {}
    for key in sorted({k for row in rows for k in row if k != "k"}):
        split, metric = key.split("/")
        table = wandb.Table(columns=[X_AXIS, metric], data=[[r["k"], r[key]] for r in rows if key in r])
        charts[f"charts/{split}_{metric}"] = wandb.plot.line(
            table, X_AXIS, metric, title=f"{metric} vs k ({split})")
    return charts


def push(record: dict, echo=print) -> str:
    """Create (or update) the W&B run for this eval run; returns its URL."""
    if record.get("status") != "complete":
        raise ValueError(f"run {record['run_id']} is {record.get('status')}, not complete")
    import wandb

    project = os.environ.get("WANDB_PROJECT") or "harness-evaluator"
    entity = os.environ.get("WANDB_ENTITY") or None
    agent_model = record["models"]["agent"]["model"].rsplit("/", 1)[-1]
    split = record["settings"]["split"]
    base_id = re.sub(r"[^A-Za-z0-9_-]", "-", record["run_id"])
    run = None
    # W&B never reuses a deleted run's id; if this run was deleted there, use base_id_2, _3, ...
    for attempt in range(1, 10):
        try:
            run = wandb.init(
                project=project,
                entity=entity,
                id=base_id if attempt == 1 else f"{base_id}_{attempt}",
                name=f"{record['harness']['name']} {agent_model} {split} n{record['settings']['trials']}",
                group=record["harness"]["name"],
                tags=[agent_model, split, record["harness"]["name"]],
                config=wandb_config(record),
                resume="allow",
                reinit=True,
                settings=wandb.Settings(x_disable_stats=True, console="off", save_code=False),
            )
            break
        except wandb.errors.CommError as e:
            if "previously created and deleted" not in str(e):
                raise
    if run is None:
        raise RuntimeError(f"no usable W&B run id for {record['run_id']}")
    try:
        rows, summary = wandb_rows(record)
        # Only the two labelled charts are logged; the numbers go in the run summary (no auto panels).
        run.log(labelled_charts(rows, wandb))
        run.summary.update(summary)
        url = run.url
    finally:
        run.finish(quiet=True)
    echo(f"pushed {record['run_id']} -> {url}")
    return url
