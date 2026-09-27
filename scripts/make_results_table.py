"""Regenerate results/RESULTS.md from results/runs/*/manifest.json (same as `nl2sql results table`).

Usage: uv run python scripts/make_results_table.py
"""

from nl2sql.eval.results import write_results

if __name__ == "__main__":
    print(f"wrote {write_results()}")
