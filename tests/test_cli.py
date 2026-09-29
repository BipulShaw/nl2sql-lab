import json
from pathlib import Path

from typer.testing import CliRunner

from nl2sql import __version__
from nl2sql.cli import app


def test_version_command_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_results_compare_prints_gap_and_p_value(tmp_path: Path) -> None:
    for name, correct in (("a", [True, True, True]), ("b", [True, False, False])):
        (tmp_path / name).mkdir()
        lines = [json.dumps({"id": str(i), "correct": c}) for i, c in enumerate(correct)]
        (tmp_path / name / "predictions.jsonl").write_text("\n".join(lines), encoding="utf-8")

    result = CliRunner().invoke(app, ["results", "compare", str(tmp_path / "a"), str(tmp_path / "b")])

    assert result.exit_code == 0
    assert "+66.7 points" in result.stdout and "p = 0.500" in result.stdout
