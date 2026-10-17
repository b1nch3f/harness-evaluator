"""Run one experiment and evaluate it: `uv run sb eval <experiment>`.

Settings come from experiments/<name>/config.yaml. Conversations run one at
a time unless run.concurrency > 1. Each run writes runs/<run_id>/ and adds
one entry to results.json (both git-ignored):

    runs/<run_id>/config.json                 everything in CLAUDE.md rule 7
    runs/<run_id>/summary.md                  scores, then each task's rewards
    runs/<run_id>/conversations/<task>_t<n>.json   full episode log (train tasks only)
    runs/<run_id>/conversations/<task>_t<n>.md     the same, readable
"""

from __future__ import annotations

import fcntl
import json
import os
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from skills_builder import data, metrics, transcript
from skills_builder.config import REPO_ROOT, TAU2_COMMIT
from skills_builder.experiment import Experiment
from skills_builder.labels import load_labels
from skills_builder.model import ModelSettings, judge_settings, user_settings
from skills_builder.retail.episode import EpisodeLog, EpisodeSpec, default_spec, run_episode

RESULTS_PATH = REPO_ROOT / "results.json"
RUNS_DIR = REPO_ROOT / "runs"
MAX_ATTEMPTS = 3  # per episode; only exceptions (transport, judge parsing) are retried
MAX_CONSECUTIVE_FAILURES = 5  # stop the run: the endpoint is probably down


class RunRefused(RuntimeError):
    pass


# --- task selection ----------------------------------------------------------


def halves(seed: int) -> dict[str, list[str]]:
    """Random halves of train for skill iterations (authoring vs scoring)."""
    train = sorted(data.splits()["train"], key=int)
    random.Random(seed).shuffle(train)
    mid = len(train) // 2
    return {"half:authoring": sorted(train[:mid], key=int), "half:scoring": sorted(train[mid:], key=int)}


def select_tasks(split: str, tasks: str | list[str], seed: int | None, final: bool) -> list[str]:
    splits = data.splits()
    test = set(splits["test"])
    if split in ("test", "all") and not final:
        raise RunRefused(f"split {split!r} includes test tasks; it runs only with --final (CLAUDE.md rule 1)")
    pool = {"train": splits["train"], "test": splits["test"], "all": splits["train"] + splits["test"]}[split]

    if tasks == "all":
        chosen = list(pool)
    elif isinstance(tasks, str) and tasks.startswith("half:"):
        if split != "train":
            raise RunRefused("half:authoring / half:scoring select from train; set run.split: train")
        chosen = halves(seed)[tasks]
    else:
        unknown = [t for t in tasks if t not in pool]
        if unknown:
            raise RunRefused(f"tasks {unknown} are not in split {split!r}")
        chosen = list(tasks)
    if not final and test.intersection(chosen):
        raise RunRefused("selection includes test tasks; it runs only with --final")
    return sorted(set(chosen), key=int)


# --- results.json --------------------------------------------------------------


def append_result(entry: dict, path: Path = RESULTS_PATH) -> None:
    """Add one run to results.json, under a file lock (safe across terminals)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        doc = json.loads(path.read_text()) if path.exists() else {"runs": []}
        doc["runs"].append(entry)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n")
        os.replace(tmp, path)


def load_results(path: Path = RESULTS_PATH) -> list[dict]:
    return json.loads(path.read_text())["runs"] if path.exists() else []


# --- running -----------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", text).strip("-")


@dataclass
class EpisodeResult:
    task_id: str
    trial: int
    log: EpisodeLog | None
    attempts: list[str] = field(default_factory=list)  # error text of each failed attempt

    @property
    def ok(self) -> bool:
        return self.log is not None


def run_with_retries(make_spec: Callable[[str, int], EpisodeSpec], task_id: str, trial: int,
                     runner: Callable[[EpisodeSpec], EpisodeLog] = run_episode) -> EpisodeResult:
    """Retry on exceptions (transport errors, unparseable judge replies), never on a low reward."""
    errors = []
    for _ in range(MAX_ATTEMPTS):
        try:
            return EpisodeResult(task_id, trial, runner(make_spec(task_id, trial)), errors)
        except Exception as e:  # noqa: BLE001 - every exception here is retried, then recorded
            errors.append(f"{type(e).__name__}: {e}"[:1000])
    return EpisodeResult(task_id, trial, None, errors)


@dataclass
class EvalSettings:
    final: bool = False
    allow_dirty: bool = False
    agent_profile: str | None = None  # overrides config.yaml agent.profile
    trials: int | None = None  # overrides config.yaml run.trials
    tasks: str | list[str] | None = None  # overrides config.yaml run.tasks
    split: str | None = None  # overrides config.yaml run.split


def run_eval(
    exp: Experiment,
    settings: EvalSettings = EvalSettings(),
    *,
    make_spec: Callable[[str, int], EpisodeSpec] | None = None,
    models: dict[str, ModelSettings | None] | None = None,
    results_path: Path = RESULTS_PATH,
    runs_dir: Path = RUNS_DIR,
    echo: Callable[[str], None] = print,
) -> dict:
    """Run every selected task `trials` times, score the run, write it out. Returns the results entry."""
    cfg = exp.config
    run_cfg = cfg.run
    agent_cfg = cfg.agent
    if settings.agent_profile:
        agent_cfg = type(agent_cfg)(**{**agent_cfg.__dict__, "profile": settings.agent_profile})
    trials = settings.trials or run_cfg.trials
    tasks_sel = settings.tasks or run_cfg.tasks
    split = settings.split or run_cfg.split

    if exp.dirty and not settings.allow_dirty:
        raise RunRefused(f"experiments/{exp.name} has uncommitted changes; commit them so the result "
                         "maps to a version (or pass --allow-dirty for a throwaway run)")
    if exp.skills_dir is not None and not exp.skill_names:
        raise RunRefused(f"experiments/{exp.name}/skills has no skills yet (no */SKILL.md)")
    task_ids = select_tasks(split, tasks_sel, run_cfg.seed, settings.final)

    if make_spec is None:
        from skills_builder.model import agent_settings

        agent = agent_settings(agent_cfg.profile, agent_cfg.model, agent_cfg.temperature, agent_cfg.reasoning_effort)
        user, judge = user_settings(), judge_settings()
        models = {"agent": agent, "user": user, "judge": judge}

        def make_spec(task_id: str, trial: int) -> EpisodeSpec:
            spec = default_spec(task_id, exp, trial)
            spec.agent, spec.user, spec.judge = agent, user, judge
            return spec
    models = models or {}
    public = {role: (s.public() if s else None) for role, s in models.items()}
    model_label = _slug((models.get("agent").model.split("/")[-1]) if models.get("agent") else "injected")

    started = datetime.now(timezone.utc)
    run_id = (f"{started:%Y%m%d-%H%M%S}_{exp.name}_{exp.version[:8]}_{model_label}"
              f"_{split}_n{trials}")
    run_dir = runs_dir / run_id
    conv_dir = run_dir / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    test_ids = set(data.splits()["test"])
    labels = load_labels()
    tasks_by_id = data.tasks_by_id()

    config = {
        "run_id": run_id,
        "started_at": started.isoformat(timespec="seconds"),
        "experiment": exp.public(),
        "scaffolding": {"git_sha": _git("rev-parse", "HEAD"),
                        "dirty": bool(_git("status", "--porcelain", "--", "src"))},
        "agent_profile": agent_cfg.profile,
        "models": public,  # agent, user simulator, judge: model, endpoint label, temperature, reasoning effort
        "split": split,
        "tasks": tasks_sel,
        "seed": run_cfg.seed,
        "task_ids": task_ids,
        "trials": trials,
        "max_steps": run_cfg.max_steps,
        "concurrency": run_cfg.concurrency,
        "final": settings.final,
        "tau2_data_commit": TAU2_COMMIT,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    jobs = [(tid, trial) for tid in task_ids for trial in range(trials)]
    echo(f"run {run_id}: {len(task_ids)} tasks x {trials} trials = {len(jobs)} conversations, "
         f"{run_cfg.concurrency} at a time")

    results: list[EpisodeResult] = []
    lock = threading.Lock()
    stop = threading.Event()
    consecutive_failures = 0
    t0 = time.time()

    def job(tid: str, trial: int) -> None:
        nonlocal consecutive_failures
        if stop.is_set():
            return
        res = run_with_retries(make_spec, tid, trial)
        with lock:
            results.append(res)
            consecutive_failures = 0 if res.ok else consecutive_failures + 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                stop.set()
            if res.ok and tid not in test_ids:  # never write test conversations (CLAUDE.md rule 1)
                d = res.log.to_dict()
                d["attempt_errors"] = res.attempts
                stem = f"{tid}_t{trial}"
                (conv_dir / f"{stem}.json").write_text(json.dumps(d, indent=2, default=str) + "\n")
                (conv_dir / f"{stem}.md").write_text(transcript.render(d, tasks_by_id[tid]))
            n = len(results)
            if res.ok:
                log = res.log
                echo(f"[{n}/{len(jobs)}] task {tid} trial {trial}: reward {log.reward:g} "
                     f"({log.termination_reason}, {log.steps} steps, {log.seconds:.0f}s)"
                     + (f", {len(res.attempts)} retries" if res.attempts else ""))
            else:
                echo(f"[{n}/{len(jobs)}] task {tid} trial {trial}: FAILED after {MAX_ATTEMPTS} attempts: "
                     f"{res.attempts[-1]}")

    if run_cfg.concurrency == 1:
        for tid, trial in jobs:
            job(tid, trial)
    else:
        with ThreadPoolExecutor(max_workers=run_cfg.concurrency) as pool:
            for f in [pool.submit(job, tid, trial) for tid, trial in jobs]:
                f.result()

    # --- score ---
    by_task: dict[str, list[EpisodeResult]] = {tid: [] for tid in task_ids}
    for r in results:
        by_task[r.task_id].append(r)
    per_task = {}
    for tid in task_ids:
        done = sorted((r for r in by_task[tid] if r.ok), key=lambda r: r.trial)
        rewards = [r.log.reward for r in done]
        per_task[tid] = {
            "split": "test" if tid in test_ids else "train",
            "successes": sum(1 for x in rewards if x == 1.0),
            "trials": len(rewards),
            "rewards": rewards,
            "terminations": [r.log.termination_reason for r in done],
            "infra_failures": sum(1 for r in by_task[tid] if not r.ok),
            "labels": labels[tid]["labels"], "width": labels[tid]["width"], "depth": labels[tid]["depth"],
        }

    scores = {}
    for split in ("train", "test"):
        rows = {t: (p["successes"], p["trials"]) for t, p in per_task.items() if p["split"] == split and p["trials"]}
        if rows:
            scores[split] = metrics.summarize(rows, labels)
    if len(scores) == 2:
        scores["all"] = metrics.summarize(
            {t: (p["successes"], p["trials"]) for t, p in per_task.items() if p["trials"]}, labels)

    status = "aborted" if stop.is_set() else "complete"
    usage: dict = {}
    for r in results:
        if r.ok:
            for role, u in r.log.usage.items():
                agg = usage.setdefault(role, {})
                for k, v in u.items():
                    agg[k] = agg.get(k, 0) + v
    entry = {
        **config,
        "status": status,
        "seconds": round(time.time() - t0, 1),
        "episodes": {"planned": len(jobs), "completed": sum(r.ok for r in results),
                     "infra_failed": sum(not r.ok for r in results),
                     "retries": sum(len(r.attempts) for r in results if r.ok),
                     "not_run": len(jobs) - len(results)},
        "usage": usage,
        "scores": scores,
        "per_task": per_task,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)) if run_dir.is_relative_to(REPO_ROOT) else str(run_dir),
    }
    (run_dir / "summary.md").write_text(summary_markdown(entry))
    (run_dir / "result.json").write_text(json.dumps(entry, indent=2) + "\n")
    if status == "complete":
        append_result(entry, results_path)
    return entry


# --- reporting ---------------------------------------------------------------


def _fmt(s: dict, key: str) -> str:
    if key not in s:
        return "-"
    ci = s.get(f"{key}_ci")
    return f"{s[key]:.3f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def summary_markdown(entry: dict) -> str:
    exp = entry["experiment"]
    agent = (entry["models"].get("agent") or {})
    ep = entry["episodes"]
    out = [
        f"# {entry['run_id']}",
        "",
        f"- experiment: `{exp['name']}` version `{exp['version'][:12]}` (commit `{(exp['git_commit'] or '-')[:10]}`"
        f"{', DIRTY' if exp['dirty'] else ''}); policy in prompt: {exp['uses_policy']}; "
        f"skills: {', '.join(exp['skills']) or 'none'}",
        f"- agent: `{agent.get('model')}` at `{agent.get('api_base')}`, temperature {agent.get('temperature')}, "
        f"reasoning effort {agent.get('reasoning_effort')}",
        f"- split `{entry['split']}`, tasks `{entry['tasks']}`, {entry['trials']} trials, max steps {entry['max_steps']}",
        f"- status: {entry['status']}; episodes {ep['completed']}/{ep['planned']} completed, "
        f"{ep['infra_failed']} failed on infrastructure, {ep['retries']} retries; {entry['seconds']:.0f}s",
        "",
    ]
    ks = [k for k in metrics.KS if k <= entry["trials"]]
    for split, s in entry["scores"].items():
        o = s["overall"]
        out += [f"## {split} ({o['tasks']} tasks)", "",
                "| metric | " + " | ".join(f"k={k}" for k in ks) + " |",
                "|---|" + "---|" * len(ks),
                "| pass^k | " + " | ".join(_fmt(o, f"pass^{k}") for k in ks) + " |",
                "| pass@k | " + " | ".join(_fmt(o, f"pass@{k}") for k in ks) + " |", ""]
        for group in ("by_intent", "by_width", "by_depth"):
            out += [f"**{group.replace('_', ' ')}**", "", "| | tasks | " + " | ".join(f"pass^{k}" for k in ks) + " |",
                    "|---|---|" + "---|" * len(ks)]
            for key, g in s[group].items():
                out.append(f"| {key} | {g['tasks']} | " + " | ".join(_fmt(g, f"pass^{k}") for k in ks) + " |")
            out.append("")
    out += ["## Tasks", "", "| task | split | labels | rewards | endings |", "|---|---|---|---|---|"]
    for tid, p in entry["per_task"].items():
        rewards = " ".join(f"{r:g}" for r in p["rewards"]) + (f" (+{p['infra_failures']} failed)" if p["infra_failures"] else "")
        out.append(f"| {tid} | {p['split']} | {', '.join(p['labels'])} | {rewards} | {', '.join(sorted(set(p['terminations'])))} |")
    return "\n".join(out) + "\n"
