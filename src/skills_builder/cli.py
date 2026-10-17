"""Command-line entry point: `uv run sb ...`."""

from __future__ import annotations

import typer

from skills_builder.config import load_env

load_env()

app = typer.Typer(no_args_is_help=True, help="Skills experiment on tau2-bench retail.")


@app.callback()
def main() -> None:
    """Skills experiment on tau2-bench retail."""


@app.command()
def info() -> None:
    """Print installed versions of the pinned dependencies."""
    from importlib.metadata import version

    for pkg in ("tau2", "google-adk", "litellm", "wandb"):
        typer.echo(f"{pkg}=={version(pkg)}")


@app.command()
def labels() -> None:
    """Write data/labels/retail.json and print label counts per split."""
    from skills_builder.labels import LABELS_PATH, label_counts, write_labels

    rows = write_labels()
    typer.echo(f"wrote {LABELS_PATH.relative_to(LABELS_PATH.parents[2])} ({len(rows)} tasks)")
    for split in ("train", "test"):
        n = sum(1 for r in rows.values() if r["split"] == split)
        typer.echo(f"{split} ({n} tasks): {label_counts(rows, split)}")


@app.command()
def ping() -> None:
    """Send one chat and one tool-call request to the agent model."""
    from skills_builder.ping import main as run_ping

    if not run_ping():
        raise typer.Exit(1)


@app.command()
def experiments() -> None:
    """List the experiments under experiments/ and their versions."""
    from skills_builder.experiment import list_experiments, load_experiment

    for name in list_experiments():
        e = load_experiment(name)
        skills = "no skills folder" if e.skills_dir is None else f"skills: {', '.join(e.skill_names) or '(empty)'}"
        typer.echo(f"{name:16} version {e.version[:12]}{' DIRTY' if e.dirty else ''}  "
                   f"policy: {'yes' if e.uses_policy else 'no'}  {skills}")


@app.command(name="eval")
def eval_(
    experiment: str = typer.Argument(..., help="folder name under experiments/"),
    final: bool = typer.Option(False, "--final", help="allow test tasks (only when the owner asks)"),
    agent: str = typer.Option(None, help="override config.yaml agent.profile (a .env prefix)"),
    trials: int = typer.Option(None, help="override config.yaml run.trials"),
    split: str = typer.Option(None, help="override config.yaml run.split: train | test | all"),
    tasks: str = typer.Option(None, help="override run.tasks: all | half:authoring | half:scoring | 0,5,12"),
    allow_dirty: bool = typer.Option(False, "--allow-dirty", help="run with uncommitted experiment changes"),
) -> None:
    """Run an experiment from experiments/<name>/config.yaml and evaluate it."""
    from skills_builder.config import REPO_ROOT
    from skills_builder.experiment import load_experiment
    from skills_builder.runner import EvalSettings, RunRefused, run_eval

    task_sel = None
    if tasks:
        task_sel = tasks if tasks == "all" or tasks.startswith("half:") else [t.strip() for t in tasks.split(",")]
    try:
        entry = run_eval(load_experiment(experiment),
                         EvalSettings(final=final, allow_dirty=allow_dirty, agent_profile=agent,
                                      trials=trials, tasks=task_sel, split=split),
                         echo=typer.echo)
    except RunRefused as e:
        typer.echo(f"refused: {e}")
        raise typer.Exit(2)
    typer.echo("")
    typer.echo((REPO_ROOT / entry["run_dir"] / "summary.md").read_text())
    if entry["status"] != "complete":
        typer.echo("run aborted (too many consecutive failures); not added to results.json")
        raise typer.Exit(1)
    typer.echo(f"added to results.json; conversations in {entry['run_dir']}/conversations/")


@app.command()
def results(experiment: str = typer.Option(None, help="only this experiment")) -> None:
    """Show every run in results.json, oldest first."""
    from skills_builder.runner import load_results

    runs = [r for r in load_results() if not experiment or r["experiment"]["name"] == experiment]
    if not runs:
        typer.echo("no runs yet")
        return
    typer.echo(f"{'run started':20} {'experiment':14} {'version':9} {'agent model':22} {'split':5} {'n':>2} "
               f"{'scored on':9} {'pass^1':>7} {'pass^2':>7} {'pass^4':>7}")
    for r in runs:
        agent = (r["models"].get("agent") or {}).get("model", "-").split("/")[-1]
        for split, s in r["scores"].items():
            o = s["overall"]
            scored = f"{split} ({o['tasks']})"
            cells = " ".join(f"{o[k]:7.3f}" if k in o else f"{'-':>7}" for k in ("pass^1", "pass^2", "pass^4"))
            typer.echo(f"{r['started_at'][:19]:20} {r['experiment']['name']:14} {r['experiment']['version'][:8]:9} "
                       f"{agent[:22]:22} {r['split']:5} {r['trials']:>2} {scored:9} {cells}")


@app.command()
def episode(
    experiment: str = typer.Argument(..., help="folder name under experiments/"),
    task_id: str = typer.Option(..., help="tau2 retail task id (train only)"),
) -> None:
    """Run one conversation and print its score. The log goes to runs/ (git-ignored)."""
    import json

    from skills_builder import data, transcript
    from skills_builder.config import REPO_ROOT
    from skills_builder.experiment import load_experiment
    from skills_builder.retail.episode import default_spec, run_episode

    if task_id in data.splits()["test"]:
        typer.echo("refusing: that is a test-split task (CLAUDE.md rule 1)")
        raise typer.Exit(2)
    exp = load_experiment(experiment)
    log = run_episode(default_spec(task_id, exp)).to_dict()
    out = REPO_ROOT / "runs" / "episodes" / f"{experiment}_task{task_id}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(log, indent=2, default=str))
    out.with_suffix(".md").write_text(transcript.render(log, data.tasks_by_id()[task_id]))
    typer.echo(f"task {task_id} [{experiment}] reward={log['reward']} termination={log['termination_reason']} "
               f"steps={log['steps']} tool_calls={len(log['tool_calls'])} tool_errors={log['tool_errors']} "
               f"skill_calls={len(log['skill_tool_calls'])} {log['seconds']}s")
    typer.echo(f"transcript: {out.with_suffix('.md').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    app()
