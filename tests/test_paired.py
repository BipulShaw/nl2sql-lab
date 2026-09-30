import json
from pathlib import Path

import pytest

from nl2sql.eval.paired import compare_runs, mcnemar_exact_p


def test_mcnemar_small_cases_by_hand() -> None:
    # 5 disagreements all one way: 2 * (1/2)^5
    assert mcnemar_exact_p(5, 0) == pytest.approx(0.0625)
    assert mcnemar_exact_p(0, 6) == pytest.approx(0.03125)


def test_mcnemar_is_symmetric_and_capped_at_one() -> None:
    assert mcnemar_exact_p(64, 55) == mcnemar_exact_p(55, 64)
    assert mcnemar_exact_p(10, 10) == 1.0
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_large_counts_do_not_overflow() -> None:
    assert 0.0 <= mcnemar_exact_p(1500, 1400) <= 1.0


def write_run(path: Path, correct: dict[str, bool] | list[tuple[str, bool]]) -> Path:
    path.mkdir()
    items = correct.items() if isinstance(correct, dict) else correct
    lines = [json.dumps({"id": i, "correct": c}) for i, c in items]
    (path / "predictions.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_compare_runs_counts_each_cell(tmp_path: Path) -> None:
    a = write_run(tmp_path / "a", {"1": True, "2": True, "3": False, "4": False})
    b = write_run(tmp_path / "b", {"1": True, "2": False, "3": True, "4": False})

    result = compare_runs(a, b)

    assert (result["both"], result["only_a"], result["only_b"], result["neither"]) == (1, 1, 1, 1)
    assert result["n"] == 4 and result["p_value"] == 1.0


def test_a_repeated_id_counts_twice(tmp_path: Path) -> None:
    a = write_run(tmp_path / "a", [("137", True), ("137", True), ("200", False)])
    b = write_run(tmp_path / "b", [("137", False), ("137", False), ("200", False)])

    result = compare_runs(a, b)

    assert result["n"] == 3 and result["only_a"] == 2


def test_compare_runs_needs_the_same_examples(tmp_path: Path) -> None:
    a = write_run(tmp_path / "a", {"1": True, "2": True})
    b = write_run(tmp_path / "b", {"1": True, "3": True})

    with pytest.raises(ValueError, match="different examples"):
        compare_runs(a, b)
