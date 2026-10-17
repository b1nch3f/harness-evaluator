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


if __name__ == "__main__":
    app()
