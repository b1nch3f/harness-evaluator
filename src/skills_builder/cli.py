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


if __name__ == "__main__":
    app()
