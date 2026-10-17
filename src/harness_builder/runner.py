"""Run one experiment and evaluate it: `uv run sb eval <experiment> [flags]`.

Run settings are flags (see RunSettings). Conversations run one at a time
unless concurrency > 1. Each run is self-contained in the experiment's
folder (git-ignored):

    experiments/<name>/runs/<run_id>/results.json   settings, scores and per-task rewards
    experiments/<name>/runs/<run_id>/summary.md     the same, readable
    experiments/<name>/runs/<run_id>/episodes.jsonl progress: one line per finished episode (for --resume)
    experiments/<name>/runs/<run_id>/conversations/<task>_t<n>.json   full log (train tasks only)
    experiments/<name>/runs/<run_id>/conversations/<task>_t<n>.md     the same, readable
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from harness_builder import data, metrics, transcript
from harness_builder.config import REPO_ROOT, TAU2_COMMIT
from harness_builder.experiment import EXPERIMENTS_DIR, Experiment
from harness_builder.labels import load_labels
from harness_builder.model import ModelSettings, agent_settings, judge_settings, user_settings
from harness_builder.retail.episode import EpisodeLog, EpisodeSpec, run_episode

RUNS_DIR = "runs"
RESULTS_FILE = "results.json"
SPLITS = ("train", "test", "all")
TASK_SETS = ("all", "half:authoring", "half:scoring")
MAX_ATTEMPTS = 3  # per episode; only exceptions (transport, judge parsing) are retried
MAX_CONSECUTIVE_FAILURES = 5  # stop the run: the endpoint is probably down


class RunRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class RunSettings:
    """Everything that varies between runs of one experiment (the `sb eval` flags)."""

    agent: str = "AGENT"  # .env profile: <AGENT>_MODEL, <AGENT>_API_BASE, <AGENT>_API_KEY
    model: str | None = None  # overrides <AGENT>_MODEL
    temperature: float | None = None  # overrides <AGENT>_TEMPERATURE (default 0.0)
    reasoning_effort: str | None = None  # low | medium | high | none; overrides <AGENT>_REASONING_EFFORT
    split: str = "train"  # train | test | all; test and all need final=True
    trials: int = 4
    tasks: str | tuple[str, ...] = "all"  # all | half:authoring | half:scoring | task ids
    seed: int | None = None  # required for half:*
    concurrency: int = 1
    max_steps: int = 100
    final: bool = False
    allow_dirty: bool = False

    def __post_init__(self):
        if self.split not in SPLITS:
            raise RunRefused(f"split must be one of {SPLITS}")
        if self.trials < 1 or self.concurrency < 1 or self.max_steps < 1:
            raise RunRefused("trials, concurrency and max_steps must be >= 1")
        if isinstance(self.tasks, str) and self.tasks not in TASK_SETS:
            raise RunRefused(f"tasks must be one of {TASK_SETS} or a list of task ids")
        if isinstance(self.tasks, str) and self.tasks.startswith("half:") and self.seed is None:
            raise RunRefused("--seed is required for half:authoring / half:scoring")


# --- task selection ----------------------------------------------------------


def halves(seed: int) -> dict[str, list[str]]:
    """Random halves of train for tuning iterations (authoring vs scoring)."""
    train = sorted(data.splits()["train"], key=int)
    random.Random(seed).shuffle(train)
    mid = len(train) // 2
    return {"half:authoring": sorted(train[:mid], key=int), "half:scoring": sorted(train[mid:], key=int)}


def select_tasks(split: str, tasks, seed: int | None, final: bool) -> list[str]:
    splits = data.splits()
    if split in ("test", "all") and not final:
        raise RunRefused(f"split {split!r} includes test tasks; it runs only with --final (CLAUDE.md rule 1)")
    pool = {"train": splits["train"], "test": splits["test"], "all": splits["train"] + splits["test"]}[split]

    if tasks == "all":
        chosen = list(pool)
    elif isinstance(tasks, str):
        if split != "train":
            raise RunRefused("half:authoring / half:scoring select from train; use --split train")
        chosen = halves(seed)[tasks]
    else:
        unknown = [t for t in tasks if t not in pool]
        if unknown:
            raise RunRefused(f"tasks {unknown} are not in split {split!r}")
        chosen = list(tasks)
    if not final and set(splits["test"]).intersection(chosen):
        raise RunRefused("selection includes test tasks; it runs only with --final")
    return sorted(set(chosen), key=int)


# --- reading runs back ---------------------------------------------------------


def load_runs(root: Path = EXPERIMENTS_DIR, experiment: str | None = None, complete_only: bool = True) -> list[dict]:
    """Every run's results.json under experiments/*/runs/, oldest first."""
    pattern = f"{experiment or '*'}/{RUNS_DIR}/*/{RESULTS_FILE}"
    runs = [json.loads(p.read_text()) for p in root.glob(pattern)]
    if complete_only:
        runs = [r for r in runs if r.get("status") == "complete"]
    return sorted(runs, key=lambda r: r["started_at"])


# --- running -----------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", text).strip("-")


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


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


EPISODES_FILE = "episodes.jsonl"  # one line per finished episode (no conversation text)


def find_run(exp: Experiment, run_id: str) -> Path:
    """A run folder of this experiment; `last` = the newest run that did not complete."""
    runs = exp.dir / RUNS_DIR
    if run_id == "last":
        open_runs = [p.parent for p in runs.glob(f"*/{RESULTS_FILE}")
                     if json.loads(p.read_text()).get("status") != "complete"]
        if not open_runs:
            raise RunRefused(f"experiments/{exp.name} has no unfinished run to resume")
        return max(open_runs, key=lambda p: p.name)
    path = runs / run_id
    if not (path / RESULTS_FILE).is_file():
        raise RunRefused(f"no run {run_id!r} in experiments/{exp.name}/runs/")
    return path


def _settings_from_record(record: dict) -> RunSettings:
    st = dict(record["settings"])
    if isinstance(st["tasks"], list):
        st["tasks"] = tuple(st["tasks"])
    return RunSettings(**st)


def _read_episodes(path: Path) -> dict[tuple[str, int], dict]:
    """Finished episodes from a run's progress file; later lines win."""
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                done[(e["task_id"], e["trial"])] = e
    return done


def run_eval(
    exp: Experiment,
    settings: RunSettings = RunSettings(),
    *,
    resume: str | None = None,  # a run id of this experiment, or "last"
    make_spec: Callable[[str, int], EpisodeSpec] | None = None,
    models: dict[str, ModelSettings | None] | None = None,
    runs_dir: Path | None = None,  # default: experiments/<name>/runs/
    echo: Callable[[str], None] = print,
) -> dict:
    """Run every selected task `trials` times, score the run, write it out. Returns its results.

    With `resume`, reopen that run with its own settings, skip the episodes it
    already finished, and run the rest (including ones that failed).
    """
    previous = None
    if resume:
        run_dir = find_run(exp, resume) if runs_dir is None else runs_dir / resume
        previous = json.loads((run_dir / RESULTS_FILE).read_text())
        if previous["status"] == "complete":
            raise RunRefused(f"run {run_dir.name} is already complete")
        if previous["experiment"]["version"] != exp.version:
            raise RunRefused(f"experiments/{exp.name} changed since run {run_dir.name} started "
                             "(prompt or skills differ); start a new run instead")
        settings = _settings_from_record(previous)
    s = settings
    if exp.dirty and not s.allow_dirty:
        raise RunRefused(f"experiments/{exp.name} has uncommitted changes; commit them so the result "
                         "maps to a version (or pass --allow-dirty for a throwaway run)")
    if exp.skills_dir is not None and not exp.skill_names:
        raise RunRefused(f"experiments/{exp.name}/skills has no skills yet (no */SKILL.md)")
    task_ids = previous["task_ids"] if previous else select_tasks(s.split, s.tasks, s.seed, s.final)

    if make_spec is None:
        try:
            agent = agent_settings(s.agent, s.model, s.temperature, s.reasoning_effort)
            user, judge = user_settings(), judge_settings()
        except RuntimeError as e:  # a profile's variables are missing from .env
            raise RunRefused(f"{e} (profile {s.agent!r} needs {s.agent.upper()}_MODEL, "
                             f"{s.agent.upper()}_API_BASE and {s.agent.upper()}_API_KEY)") from e
        models = {"agent": agent, "user": user, "judge": judge}

        def make_spec(task_id: str, trial: int) -> EpisodeSpec:
            return EpisodeSpec(task_id=task_id, experiment=exp, trial=trial, max_steps=s.max_steps,
                               agent=agent, user=user, judge=judge)
    models = models or {}
    public = {role: (m.public() if m else None) for role, m in models.items()}
    if previous and public != previous["models"]:
        raise RunRefused(f"model settings differ from run {run_dir.name} "
                         f"(was {previous['models']}, now {public}); start a new run instead")

    test_ids = set(data.splits()["test"])
    labels = load_labels()
    tasks_by_id = data.tasks_by_id()
    if previous:
        record = previous
        record["status"] = "running"
        record.setdefault("resumed_at", []).append(datetime.now(timezone.utc).isoformat(timespec="seconds"))
        run_id = record["run_id"]
    else:
        model_label = _slug(models["agent"].model.split("/")[-1]) if models.get("agent") else "injected"
        started = datetime.now(timezone.utc)
        run_id = f"{started:%Y%m%d-%H%M%S}_{exp.version[:8]}_{model_label}_{s.split}_n{s.trials}"
        run_dir = (runs_dir or exp.dir / RUNS_DIR) / run_id
        record = {
            "run_id": run_id,
            "status": "running",
            "started_at": started.isoformat(timespec="seconds"),
            "experiment": exp.public(),
            "scaffolding": {"git_sha": _git("rev-parse", "HEAD"),
                            "dirty": bool(_git("status", "--porcelain", "--", "src"))},
            "settings": {**asdict(s), "tasks": s.tasks if isinstance(s.tasks, str) else list(s.tasks)},
            "models": public,  # agent, user simulator, judge: model, endpoint label, temperature, reasoning effort
            "task_ids": task_ids,
            "tau2_data_commit": TAU2_COMMIT,
        }
    conv_dir = run_dir / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = run_dir / EPISODES_FILE
    _write_json(run_dir / RESULTS_FILE, record)

    done = {k: e for k, e in _read_episodes(episodes_path).items() if e["ok"]}
    all_jobs = [(tid, trial) for tid in task_ids for trial in range(s.trials)]
    jobs = [j for j in all_jobs if j not in done]
    echo(f"run {exp.name}/{run_id}: {len(task_ids)} tasks x {s.trials} trials = {len(all_jobs)} conversations"
         + (f", {len(done)} already done, {len(jobs)} to run" if previous else "")
         + f", {s.concurrency} at a time")

    lock = threading.Lock()
    stop = threading.Event()
    state = {"consecutive_failures": 0, "ran": 0}
    t0 = time.time()

    def job(tid: str, trial: int) -> None:
        if stop.is_set():
            return
        res = run_with_retries(make_spec, tid, trial)
        with lock:
            state["ran"] += 1
            state["consecutive_failures"] = 0 if res.ok else state["consecutive_failures"] + 1
            if state["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
                stop.set()
            entry = {"task_id": tid, "trial": trial, "ok": res.ok, "attempt_errors": res.attempts}
            if res.ok:
                log = res.log
                entry.update(reward=log.reward, termination_reason=log.termination_reason, steps=log.steps,
                             seconds=log.seconds, usage=log.usage)
                done[(tid, trial)] = entry
                if tid not in test_ids:  # never write test conversations (CLAUDE.md rule 1)
                    d = log.to_dict()
                    d["attempt_errors"] = res.attempts
                    stem = f"{tid}_t{trial}"
                    _write_json(conv_dir / f"{stem}.json", d)
                    (conv_dir / f"{stem}.md").write_text(transcript.render(d, tasks_by_id[tid]))
            with open(episodes_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            n = len(all_jobs) - len(jobs) + state["ran"]
            if res.ok:
                echo(f"[{n}/{len(all_jobs)}] task {tid} trial {trial}: reward {log.reward:g} "
                     f"({log.termination_reason}, {log.steps} steps, {log.seconds:.0f}s)"
                     + (f", {len(res.attempts)} retries" if res.attempts else ""))
            else:
                echo(f"[{n}/{len(all_jobs)}] task {tid} trial {trial}: FAILED after {MAX_ATTEMPTS} attempts: "
                     f"{res.attempts[-1]}")

    if s.concurrency == 1:
        for tid, trial in jobs:
            job(tid, trial)
    else:
        with ThreadPoolExecutor(max_workers=s.concurrency) as pool:
            for f in [pool.submit(job, tid, trial) for tid, trial in jobs]:
                f.result()

    # --- score (everything finished so far, across resumes) ---
    all_entries = _read_episodes(episodes_path)
    per_task = {}
    for tid in task_ids:
        mine = [all_entries[(tid, t)] for t in range(s.trials) if (tid, t) in all_entries]
        ok = [e for e in mine if e["ok"]]
        rewards = [e["reward"] for e in ok]
        per_task[tid] = {
            "split": "test" if tid in test_ids else "train",
            "successes": sum(1 for x in rewards if x == 1.0),
            "trials": len(rewards),
            "rewards": rewards,
            "terminations": [e["termination_reason"] for e in ok],
            "infra_failures": sum(1 for e in mine if not e["ok"]),
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

    usage: dict = {}
    for e in all_entries.values():
        if e["ok"]:
            for role, u in e["usage"].items():
                agg = usage.setdefault(role, {})
                for k, v in u.items():
                    agg[k] = agg.get(k, 0) + v
    completed = sum(1 for e in all_entries.values() if e["ok"])
    remaining = len(all_jobs) - completed
    record.update({
        "status": "complete" if remaining == 0 else "aborted" if stop.is_set() else "incomplete",
        "seconds": round(record.get("seconds", 0) + time.time() - t0, 1),
        "episodes": {"planned": len(all_jobs), "completed": completed,
                     "infra_failed": sum(1 for e in all_entries.values() if not e["ok"]),
                     "retries": sum(len(e["attempt_errors"]) for e in all_entries.values() if e["ok"]),
                     "not_run": remaining},
        "usage": usage,
        "scores": scores,
        "per_task": per_task,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)) if run_dir.is_relative_to(REPO_ROOT) else str(run_dir),
    })
    _write_json(run_dir / RESULTS_FILE, record)
    (run_dir / "summary.md").write_text(summary_markdown(record))
    return record


# --- reporting ---------------------------------------------------------------


def _fmt(s: dict, key: str) -> str:
    if key not in s:
        return "-"
    ci = s.get(f"{key}_ci")
    return f"{s[key]:.3f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def summary_markdown(r: dict) -> str:
    exp = r["experiment"]
    st = r["settings"]
    agent = r["models"].get("agent") or {}
    ep = r["episodes"]
    out = [
        f"# {exp['name']} / {r['run_id']}",
        "",
        f"- experiment: `{exp['name']}` version `{exp['version'][:12]}` (commit `{(exp['git_commit'] or '-')[:10]}`"
        f"{', DIRTY' if exp['dirty'] else ''}); policy in prompt: {exp['uses_policy']}; "
        f"skills: {', '.join(exp['skills']) or 'none'}",
        f"- agent: `{agent.get('model')}` at `{agent.get('api_base')}` (profile {st['agent']}), "
        f"temperature {agent.get('temperature')}, reasoning effort {agent.get('reasoning_effort')}",
        f"- split `{st['split']}`, tasks `{st['tasks']}`, {st['trials']} trials, max steps {st['max_steps']}, "
        f"concurrency {st['concurrency']}",
        f"- status: {r['status']}; episodes {ep['completed']}/{ep['planned']} completed, "
        f"{ep['infra_failed']} failed on infrastructure, {ep['retries']} retries, {ep['not_run']} not run; "
        f"{r['seconds']:.0f}s" + (f"; resumed {len(r['resumed_at'])}x" if r.get("resumed_at") else ""),
        "",
    ]
    ks = [k for k in metrics.KS if k <= st["trials"]]
    for split, s in r["scores"].items():
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
    for tid, p in r["per_task"].items():
        rewards = " ".join(f"{x:g}" for x in p["rewards"]) + (f" (+{p['infra_failures']} failed)" if p["infra_failures"] else "")
        out.append(f"| {tid} | {p['split']} | {', '.join(p['labels'])} | {rewards} | {', '.join(sorted(set(p['terminations'])))} |")
    return "\n".join(out) + "\n"
