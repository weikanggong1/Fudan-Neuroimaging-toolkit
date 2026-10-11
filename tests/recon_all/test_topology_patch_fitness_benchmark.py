"""评分输出严格门与输入矩阵字节诊断分列，不接受非零误差。"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def benchmark(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "tools/benchmark_recon_topology_patch_fitness.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("prepared_fitness_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gates(benchmark, matrix, *, output_same=True, repeat_same=True):
    return benchmark.regression_gates(comparisons={"fitness":dict(byte_equal=output_same),
        "prior_joint_pdf":dict(byte_equal=True),"ras_to_voxel":matrix},
        repeat={"fitness":dict(byte_equal=repeat_same)},reference_repeat_exact=True,
        initial_feedback_unchanged=True,source_context_equal=True)


def test_signed_zero_preserves_failed_input_byte_diagnostic_and_exact_outputs(benchmark):
    matrix = benchmark.difference(np.array([0.],np.float32),np.array([-0.],np.float32))
    assert matrix["byte_equal"] is False
    assert matrix["different_zero_sign_elements"] == 1 and matrix["differing_elements"] == 0
    result = gates(benchmark,matrix)
    assert result["scoring_execution_gate_passed"] and result["strict_stage_output_bytes"]
    assert not result["strict_preparation_input_bytes"]
    assert not result["overall_internal_byte_reproduction"]
    assert result["preparation_input_values_equal"]


@pytest.mark.parametrize("new", [np.float32(1e-20), np.float32(np.inf), np.float32(np.nan)])
def test_any_nonzero_or_nonfinite_matrix_difference_fails(benchmark,new):
    matrix = benchmark.difference(np.array([new],np.float32),np.array([0.],np.float32))
    assert not gates(benchmark,matrix)["scoring_execution_gate_passed"]


@pytest.mark.parametrize("output_same,repeat_same", [(False,True),(True,False)])
def test_output_or_repeat_byte_difference_always_fails(benchmark,output_same,repeat_same):
    matrix = benchmark.difference(np.eye(4,dtype=np.float32),np.eye(4,dtype=np.float32))
    assert not gates(benchmark,matrix,output_same=output_same,repeat_same=repeat_same)["scoring_execution_gate_passed"]


def test_shape_or_dtype_mismatch_fails(benchmark):
    for candidate,reference in [(np.ones(3,np.float32),np.ones(2,np.float32)),
            (np.ones(2,np.float32),np.ones(2,np.float64))]:
        assert not gates(benchmark,benchmark.difference(candidate,reference))["scoring_execution_gate_passed"]
