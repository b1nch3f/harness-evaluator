"""Command-line entry point: `uv run sb ...`."""

from __future__ import annotations

import typer

from test_bench.config import load_env

load_env()

app = typer.Typer(no_args_is_help=True, help="Evaluate agent harnesses on tau2-bench retail.")


@app.callback()
def main() -> None:
    """Evaluate agent harnesses on tau2-bench retail."""


@app.command()
def info() -> None:
    """Print installed versions of the pinned dependencies."""
    from importlib.metadata import version

    for pkg in ("tau2", "google-adk", "litellm", "wandb"):
        typer.echo(f"{pkg}=={version(pkg)}")


@app.command()
def labels() -> None:
    """Write data/labels/retail.json and print label counts per split."""
    from test_bench.labels import LABELS_PATH, label_counts, write_labels

    rows = write_labels()
    typer.echo(f"wrote {LABELS_PATH.relative_to(LABELS_PATH.parents[2])} ({len(rows)} tasks)")
    for split in ("train", "test"):
        n = sum(1 for r in rows.values() if r["split"] == split)
        typer.echo(f"{split} ({n} tasks): {label_counts(rows, split)}")


@app.command()
def ping() -> None:
    """Send one chat and one tool-call request to the agent model."""
    from test_bench.ping import main as run_ping

    if not run_ping():
        raise typer.Exit(1)


@app.command()
def harnesses() -> None:
    """List the harnesses under harnesses/ and their versions."""
    from test_bench.harness import list_harnesses, load_harness

    for name in list_harnesses():
        e = load_harness(name)
        skills = "no skills folder" if e.skills_dir is None else f"skills: {', '.join(e.skill_names) or '(empty)'}"
        typer.echo(f"{name:16} version {e.version[:12]}{' DIRTY' if e.dirty else ''}  "
                   f"policy: {'yes' if e.uses_policy else 'no'}  {skills}")


@app.command(name="eval")
def eval_(
    harness: str = typer.Argument(..., help="folder name under harnesses/"),
    agent: str = typer.Option("AGENT", help="agent model profile: reads <AGENT>_MODEL, _API_BASE, _API_KEY from .env"),
    model: str = typer.Option(None, help="override <AGENT>_MODEL"),
    temperature: float = typer.Option(None, help="override <AGENT>_TEMPERATURE (default 0.0)"),
    reasoning_effort: str = typer.Option(None, help="low | medium | high | none; override <AGENT>_REASONING_EFFORT"),
    split: str = typer.Option("train", help="train | test | all (test and all need --final)"),
    trials: int = typer.Option(4, help="conversations per task"),
    tasks: str = typer.Option("all", help="all | half:authoring | half:scoring | comma-separated task ids"),
    seed: int = typer.Option(None, help="seed for half:authoring / half:scoring"),
    concurrency: int = typer.Option(1, help="conversations at a time"),
    max_steps: int = typer.Option(100, help="step limit per conversation"),
    final: bool = typer.Option(False, "--final", help="allow test tasks (only when the owner asks)"),
    allow_dirty: bool = typer.Option(False, "--allow-dirty", help="run with uncommitted harness changes"),
    resume: str = typer.Option(None, help="continue an unfinished run: its run id, or 'last'. "
                                          "Uses that run's own settings; other flags are ignored."),
) -> None:
    """Run a harness and evaluate it. Results go to harnesses/<name>/runs/<run_id>/."""
    from test_bench.config import REPO_ROOT
    from test_bench.harness import load_harness
    from test_bench.runner import RunRefused, RunSettings, run_eval

    task_sel = tasks if tasks == "all" or tasks.startswith("half:") else tuple(t.strip() for t in tasks.split(","))
    try:
        settings = RunSettings(agent=agent, model=model, temperature=temperature, reasoning_effort=reasoning_effort,
                               split=split, trials=trials, tasks=task_sel, seed=seed, concurrency=concurrency,
                               max_steps=max_steps, final=final, allow_dirty=allow_dirty)
        record = run_eval(load_harness(harness), settings, resume=resume, echo=typer.echo)
    except RunRefused as e:
        typer.echo(f"refused: {e}")
        raise typer.Exit(2)
    typer.echo("")
    typer.echo((REPO_ROOT / record["run_dir"] / "summary.md").read_text())
    typer.echo(f"run saved in {record['run_dir']}/ (results.json, summary.md, conversations/)")
    if record["status"] != "complete":
        why = "stopped after repeated failures" if record["status"] == "aborted" else "has unfinished conversations"
        typer.echo(f"run {why} ({record['episodes']['not_run']} left); `sb results` skips it until done.\n"
                   f"continue with: uv run sb eval {harness} --resume {record['run_id']}")
        raise typer.Exit(1)


@app.command()
def results(harness: str = typer.Option(None, help="only this harness")) -> None:
    """Show every completed run (harnesses/*/runs/*/results.json), oldest first."""
    from test_bench.runner import load_runs

    runs = load_runs(harness=harness)
    if not runs:
        typer.echo("no runs yet")
        return
    typer.echo(f"{'run started':20} {'harness':14} {'version':9} {'agent model':22} {'n':>2} "
               f"{'scored on':11} {'pass^1':>7} {'pass^2':>7} {'pass^4':>7}")
    for r in runs:
        agent = (r["models"].get("agent") or {}).get("model", "-").split("/")[-1]
        for split, s in r["scores"].items():
            o = s["overall"]
            scored = f"{split} ({o['tasks']})"
            cells = " ".join(f"{o[k]:7.3f}" if k in o else f"{'-':>7}" for k in ("pass^1", "pass^2", "pass^4"))
            typer.echo(f"{r['started_at'][:19]:20} {r['harness']['name']:14} {r['harness']['version'][:8]:9} "
                       f"{agent[:22]:22} {r['settings']['trials']:>2} {scored:11} {cells}")


@app.command()
def push(
    harness: str = typer.Argument(..., help="harness folder name under harnesses/"),
    run_id: str = typer.Option("latest", "--run-id", help="run folder name; `latest` = newest completed run"),
) -> None:
    """Push a completed run's overall pass@k / pass^k and configuration to W&B (no per-task data)."""
    from test_bench.runner import load_runs
    from test_bench.wandb_push import push as push_run

    runs = load_runs(harness=harness)
    if run_id != "latest":
        runs = [r for r in runs if r["run_id"] == run_id]
    if not runs:
        typer.echo(f"no completed run {run_id!r} for harness {harness!r}")
        raise typer.Exit(1)
    push_run(max(runs, key=lambda r: r["run_id"]), echo=typer.echo)


@app.command()
def episode(
    harness: str = typer.Argument(..., help="folder name under harnesses/"),
    task_id: str = typer.Option(..., help="tau2 retail task id (train only)"),
    agent: str = typer.Option("AGENT", help="agent model profile in .env"),
) -> None:
    """Run one conversation and print its score. The log goes to harnesses/<name>/runs/episodes/."""
    import json

    from test_bench import data, transcript
    from test_bench.config import REPO_ROOT
    from test_bench.harness import load_harness
    from test_bench.retail.episode import default_spec, run_episode

    if task_id in data.splits()["test"]:
        typer.echo("refusing: that is a test-split task (CLAUDE.md rule 1)")
        raise typer.Exit(2)
    exp = load_harness(harness)
    log = run_episode(default_spec(task_id, exp, agent_profile=agent)).to_dict()
    out = exp.dir / "runs" / "episodes" / f"task{task_id}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(log, indent=2, default=str))
    out.with_suffix(".md").write_text(transcript.render(log, data.tasks_by_id()[task_id]))
    typer.echo(f"task {task_id} [{harness}] reward={log['reward']} termination={log['termination_reason']} "
               f"steps={log['steps']} tool_calls={len(log['tool_calls'])} tool_errors={log['tool_errors']} "
               f"skill_calls={len(log['skill_tool_calls'])} {log['seconds']}s")
    typer.echo(f"transcript: {out.with_suffix('.md').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    app()
