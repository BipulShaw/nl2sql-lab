from typer.testing import CliRunner

from nl2sql import __version__
from nl2sql.cli import app


def test_version_command_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
