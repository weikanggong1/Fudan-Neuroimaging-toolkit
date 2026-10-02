"""Ensure the real-pipeline gate preserves science while separating wall time."""
import copy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def tool():
    path = Path(__file__).resolve().parents[2] / "tools/validate_fnirt_pipeline_lossless.py"
    specification = importlib.util.spec_from_file_location("fnirt_pipeline_validation_tool", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_solver_gate_ignores_only_execution_and_nested_elapsed_seconds(tool):
    reference = {"execution": "optimized", "device": "cuda:0", "levels": [
        {"cost": 0.125, "converged": False, "iterations": 12,
         "topology_projection": {"calls": [
             {"elapsed_seconds": 3.1, "accepted": True, "range": [0.2, 4.5]}
         ]}}
    ]}
    candidate = copy.deepcopy(reference)
    candidate["execution"] = "oracle"
    candidate["levels"][0]["topology_projection"]["calls"][0]["elapsed_seconds"] = 1.2
    unchanged = copy.deepcopy(candidate)
    assert tool.solver_trace(reference) == tool.solver_trace(candidate)
    assert candidate == unchanged
    for field, value in (("cost", 0.126), ("converged", True), ("iterations", 11)):
        changed = copy.deepcopy(candidate)
        changed["levels"][0][field] = value
        assert tool.solver_trace(reference) != tool.solver_trace(changed)
    candidate["levels"][0]["topology_projection"]["calls"][0]["accepted"] = False
    assert tool.solver_trace(reference) != tool.solver_trace(candidate)


def test_missing_solver_trace_cannot_pass(tool):
    with pytest.raises(ValueError, match="missing its solver trace"):
        tool.solver_trace({"elapsed_seconds": 1.0})


def test_bids_text_transform_is_science_and_missing_or_changed_fails(tool, tmp_path):
    import json
    import numpy as np

    first, second = tmp_path / "baseline", tmp_path / "candidate"
    for root in (first, second):
        root.mkdir()
        (root / "fnirt_qc.json").write_text(json.dumps({"levels": [{"cost": 0.1}]}))
    name = "sub-example_from-boldref_to-T1w_mode-image_xfm.txt"
    matrix = np.eye(4)
    for root in (first, second):
        np.savetxt(root / name, matrix)
    assert tool.compare_trees(first, second)["all_equal"]
    matrix[0, 3] = 0.001
    np.savetxt(second / name, matrix)
    assert not tool.compare_trees(first, second)["all_equal"]
    (second / name).unlink()
    assert not tool.compare_trees(first, second)["all_equal"]
