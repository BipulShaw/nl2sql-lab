"""Command-line entry point: `uv run nl2sql <command>`."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from nl2sql import __version__
from nl2sql.execute.sqlite_exec import MAX_ROWS  # stdlib-only module, cheap to import

app = typer.Typer(help="NL2SQL lab: text-to-SQL pipeline, fine-tuning and evaluation.", no_args_is_help=True)
data_app = typer.Typer(help="Datasets fetched by scripts/download_data.sh.", no_args_is_help=True)
results_app = typer.Typer(help="Results tables built from run manifests.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(results_app, name="results")
console = Console()


@app.callback()
def main() -> None:
    """Keep Typer in multi-command mode."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@data_app.command("verify")
def data_verify(workers: int = 8, timeout_s: float = 30.0, max_rows: int = MAX_ROWS) -> None:
    """Count examples per split, check each database file exists, and execute every gold query once."""
    from nl2sql.data.verify import verify

    table = Table(
        "split",
        "n",
        "expected",
        "databases",
        "missing dbs",
        "gold errors",
        "of which timeouts",
        "over row cap",
        "max gold rows",
    )
    for name, row in verify(workers, timeout_s, max_rows).items():
        count = str(row["n"]) if row["n"] == row["expected"] else f"[red]{row['n']}[/red]"
        table.add_row(
            name,
            count,
            str(row["expected"]),
            str(row["databases"]),
            ", ".join(row["missing_databases"]) or "0",
            str(row["gold_errors"]),
            str(row["gold_timeouts"]),
            str(row["gold_over_row_cap"]),
            str(row["max_gold_rows"]),
        )
    console.print(table)
    console.print("Gold queries that cannot be scored: data/gold_failures.json")


@app.command("eval")
def eval_command(
    dataset: Annotated[str, typer.Option(help="spider, bird or bird-mini")],
    split: Annotated[str, typer.Option()] = "dev",
    config: Annotated[Path | None, typer.Option(help="configs/<name>.yaml (not needed with --gold)")] = None,
    limit: Annotated[int | None, typer.Option(help="stratified sample of this many examples")] = None,
    seed: Annotated[int, typer.Option(help="sampling seed")] = 42,
    gold: Annotated[
        bool, typer.Option("--gold", help="score the gold SQL itself: harness sanity check")
    ] = False,
    workers: Annotated[int, typer.Option(help="processes executing and scoring SQL")] = 8,
    cache: Annotated[bool, typer.Option(help="reuse cached generations")] = True,
) -> None:
    """Generate, execute and score one dataset split; writes results/runs/<run_id>/."""
    from nl2sql.config import load_config
    from nl2sql.eval.harness import run_eval

    if config is None and not gold:
        raise typer.BadParameter("--config is required unless --gold is set")
    manifest = run_eval(
        dataset,
        split,
        load_config(config) if config else None,
        limit=limit,
        seed=seed,
        gold=gold,
        workers=workers,
        use_cache=cache,
    )
    metrics, counts = manifest["metrics"], manifest["counts"]
    console.print(f"[bold]{manifest['run_id']}[/bold]")
    console.print(
        f"EX {metrics['ex']:.4f}  (valid gold {metrics['ex_valid_gold']:.4f}, "
        f"any column order {metrics['ex_any_column_order']:.4f})  n={manifest['n']}"
    )
    console.print({k: v for k, v in counts.items() if v})


@results_app.command("table")
def results_table() -> None:
    """Regenerate results/RESULTS.md from all run manifests."""
    from nl2sql.eval.results import write_results

    console.print(f"wrote {write_results()}")
