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
def episode(
    task_id: str = typer.Option(..., help="tau2 retail task id"),
    condition: str = typer.Option("A_baseline", help="A0_bare | A_baseline | B_skills | C_skills_only"),
    skills_dir: str = typer.Option(None, help="skills folder (B_skills, C_skills_only)"),
) -> None:
    """Run one conversation and print its score. The full log goes to runs/ (git-ignored)."""
    import json
    from pathlib import Path

    from skills_builder import data
    from skills_builder.config import REPO_ROOT
    from skills_builder.retail.episode import default_spec, run_episode

    if task_id in data.splits()["test"]:
        typer.echo("refusing: that is a test-split task (CLAUDE.md rule 1)")
        raise typer.Exit(2)
    log = run_episode(default_spec(task_id, condition, Path(skills_dir) if skills_dir else None))
    out = REPO_ROOT / "runs" / f"episode_{condition}_task{task_id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(log.to_dict(), indent=2, default=str))
    typer.echo(f"task {task_id} [{condition}] reward={log.reward} termination={log.termination_reason} "
               f"steps={log.steps} agent_turns={log.agent_turns} tool_calls={len(log.tool_calls)} "
               f"tool_errors={log.tool_errors} skill_calls={len(log.skill_tool_calls)} {log.seconds}s")
    typer.echo(f"score: {json.dumps({k: log.score[k] for k in ('db_match', 'nl_reward', 'note')})}")
    typer.echo(f"usage: {json.dumps(log.usage)}")
    typer.echo(f"log: {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    app()
