"""pytest entrypoint for the golden set. Runs only when `-m eval` is passed.

Usage:
  pytest -m eval -s backend/tests/test_eval_golden.py
"""
import asyncio
import pytest

from evals.run_eval import run_eval, _save


@pytest.mark.eval
def test_golden_set_runs_and_summary_meets_thresholds():
    payload = asyncio.run(run_eval("pytest_marker_run"))
    _save(payload, "pytest_marker_run")
    s = payload["summary"]
    assert s["faithfulness_mean"] >= 0.70, f"faithfulness too low: {s}"
    assert s["answered_rate"] >= 0.80, f"answered_rate too low: {s}"
