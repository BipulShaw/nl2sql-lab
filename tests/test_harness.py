import subprocess
from pathlib import Path

import pytest

from nl2sql.eval.harness import git_state


def git(repo: Path, *args: str) -> None:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "init.defaultBranch=main"]
    subprocess.run(["git", *identity, *args], cwd=repo, check=True, capture_output=True)


def test_git_state_is_dirty_for_code_changes_but_not_run_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    git(tmp_path, "add", "code.py")
    git(tmp_path, "commit", "-q", "-m", "init")
    monkeypatch.chdir(tmp_path)

    clean = git_state()
    run_dir = tmp_path / "results" / "runs" / "earlier_run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}\n")
    after_a_run = git_state()
    (tmp_path / "code.py").write_text("x = 2\n")
    edited = git_state()

    assert clean["dirty"] is False and len(clean["tree"]) == 40
    assert after_a_run["dirty"] is False
    assert edited["dirty"] is True and edited["tree"] == clean["tree"]
