"""Command-line entry point: `uv run nl2sql <command>`."""

import typer

from nl2sql import __version__

app = typer.Typer(help="NL2SQL lab: text-to-SQL pipeline, fine-tuning and evaluation.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Keep Typer in multi-command mode while there is only one command."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
