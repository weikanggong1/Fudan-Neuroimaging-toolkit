"""Public operator fixture from actual newMSM SDK; not a clinical benchmark."""

import json
from pathlib import Path

import numpy as np
import pytest

from fnit.msm.msmsulc import _adaptive_resample, _vertex_area


@pytest.fixture(scope="module")
def copy_area_fixture():
    pytest.importorskip("fnit.msm._fastpd_native")
    path = Path(__file__).parent / "data" / "msm_multivariate_copy_area.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["official_source_commit"] == "260718953547743c028a45f8c885d163441df87a"
    return {key: np.asarray(value) for key, value in record.items()
            if isinstance(value, list)}


def test_mesh_copy_area_matches_actual_sdk_resampling(copy_area_fixture):
    """Copying the two deformed meshes reconstructs their triangle areas."""
    data = copy_area_fixture
    vertices, faces = data["source_vertices"], data["source_faces"]
    target, target_faces = data["target_vertices"], data["target_faces"]
    old_area = _vertex_area(vertices, faces)
    new_area = _vertex_area(target, target_faces)
    np.testing.assert_allclose(old_area, data["source_copy_vertex_area"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(new_area, data["target_copy_vertex_area"], rtol=0, atol=1e-12)

    actual = _adaptive_resample(
        vertices, faces, data["source_values"], target, target_faces,
        device="cpu", source_precision=True, old_area=old_area, new_area=new_area,
    )
    # The expected three columns were produced by the installed, source-gated
    # official metric_resample(tmp_copy, cp_copy, 1), with nonconstant weights.
    np.testing.assert_allclose(actual, data["expected_values"], rtol=0, atol=2e-14)


def test_fixed_planar_area_changes_actual_sdk_weight_result(copy_area_fixture):
    """The replaced area contract has a detectable error on this public mesh."""
    data = copy_area_fixture
    previous = _adaptive_resample(
        data["source_vertices"], data["source_faces"], data["source_values"],
        data["target_vertices"], data["target_faces"], device="cpu",
        source_precision=True, old_area=data["source_fixed_planar_vertex_area"],
        new_area=data["target_fixed_planar_vertex_area"],
    )
    assert np.max(np.abs(previous - data["expected_values"])) > 2e-4
