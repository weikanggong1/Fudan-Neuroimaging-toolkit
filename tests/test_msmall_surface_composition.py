"""Keep native and fsLR feature-coordinate systems distinct at projection."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri.assets_setup import MESH
from fnit.fmri.surface_pipeline import _refine_msmall
from fnit.msm import MSMAllConfig, MSMAllInputs
from fnit.msm.msmsulc import _ico


def save_mesh(path, level):
    points, faces = _ico(level)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(points.astype(np.float32), intent=1008),
        nib.gifti.GiftiDataArray(faces.astype(np.int32), intent=1009),
    ]), path)
    return path


def setup_case(tmp_path, native_features):
    assets = tmp_path / "assets"
    geometry = []
    native_spheres = []
    inputs = {}
    for hemisphere in "LR":
        native = save_mesh(tmp_path / f"{hemisphere}.native.surf.gii", 2)
        atlas = save_mesh(assets / MESH / f"{hemisphere}.sphere.32k_fs_LR.surf.gii", 1)
        native_spheres.append(tmp_path / f"{hemisphere}.MSMSulc.surf.gii")
        geometry.append(SimpleNamespace(midthickness=native))
        inputs[hemisphere] = MSMAllInputs(native if native_features else atlas,
                                          tmp_path / "individual.func.gii", atlas,
                                          tmp_path / "reference.func.gii")
    return inputs, tuple(native_spheres), tuple(geometry), assets


def mock_solver(monkeypatch, observed):
    import fnit.msm
    def solve(inputs, output, **options):
        observed["inputs"] = inputs
        return {hemisphere: output / f"{hemisphere}.sphere.MSMAll.native.surf.gii"
                for hemisphere in "LR"}
    monkeypatch.setattr(fnit.msm, "run_msmall", solve)
    monkeypatch.setattr("fnit.fmri.surface_pipeline.shutil.which", lambda command: "/bin/wb_command")


def test_fsLR_warp_is_composed_on_native_msmsulc_before_projection(tmp_path, monkeypatch):
    inputs, native_spheres, geometry, assets = setup_case(tmp_path, False)
    observed = {}; commands = []
    mock_solver(monkeypatch, observed)
    monkeypatch.setattr("fnit.fmri.surface_pipeline.subprocess.run",
                        lambda command, **options: commands.append(command))
    result, topology = _refine_msmall(inputs, native_spheres, geometry, assets,
                                      tmp_path / "result", MSMAllConfig(), "cpu", "optimized", "wb_command")
    assert topology == {"L": "fsLR32k", "R": "fsLR32k"}
    for hemisphere, initial_native, command, output in zip("LR", native_spheres, commands, result):
        assert command[1:4] == ["-surface-sphere-project-unproject", str(initial_native),
                                str(inputs[hemisphere].source_sphere)]
        assert Path(command[4]) != output  # Preserve the solver's 32k sphere.
        assert command[5] == str(output)
        assert observed["inputs"][hemisphere].initial_sphere is None  # Identity in MSMSulc 32k space.


def test_native_features_use_initial_native_registration_without_atlas_composition(tmp_path, monkeypatch):
    inputs, native_spheres, geometry, assets = setup_case(tmp_path, True)
    observed = {}; mock_solver(monkeypatch, observed)
    def forbidden(*args, **kwargs):
        raise AssertionError("native topology must not be composed as a 32k warp")
    monkeypatch.setattr("fnit.fmri.surface_pipeline.subprocess.run", forbidden)
    _, topology = _refine_msmall(inputs, native_spheres, geometry, assets, tmp_path / "result",
                                MSMAllConfig(), "cpu", "optimized", "wb_command")
    assert topology == {"L": "native", "R": "native"}
    assert observed["inputs"]["L"].initial_sphere == native_spheres[0]


def test_unknown_feature_topology_fails_before_registration(tmp_path, monkeypatch):
    inputs, native_spheres, geometry, assets = setup_case(tmp_path, False)
    inputs["L"] = replace(inputs["L"], source_sphere=save_mesh(tmp_path / "unknown.surf.gii", 3))
    import fnit.msm
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported source coordinate system must be rejected first")
    monkeypatch.setattr(fnit.msm, "run_msmall", forbidden)
    with pytest.raises(ValueError, match="matching native topology or the canonical"):
        _refine_msmall(inputs, native_spheres, geometry, assets, tmp_path / "result",
                      MSMAllConfig(), "cpu", "optimized", "wb_command")
