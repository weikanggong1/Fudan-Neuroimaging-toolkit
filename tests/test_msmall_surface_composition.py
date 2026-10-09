"""Keep native and fsLR feature-coordinate systems distinct at projection."""

from dataclasses import replace
from pathlib import Path
import json
import shutil
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri.assets_setup import MESH
from fnit.fmri.surface_pipeline import (
    _orientation_chain, _refine_msmall, _saved_native_sphere_qc,
)
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
        # This is the actual registration sphere consumed by MSMAll.  Keep it
        # separate from midthickness so the topology oracle is exercised.
        registration = save_mesh(tmp_path / f"{hemisphere}.MSMSulc.surf.gii", 2)
        native_spheres.append(registration)
        geometry.append(SimpleNamespace(midthickness=native))
        inputs[hemisphere] = MSMAllInputs(native if native_features else atlas,
                                          tmp_path / "individual.func.gii", atlas,
                                          tmp_path / "reference.func.gii")
    return inputs, tuple(native_spheres), tuple(geometry), assets


def mock_solver(monkeypatch, observed):
    import fnit.msm
    def solve(inputs, output, **options):
        observed["inputs"] = inputs
        output.mkdir(parents=True, exist_ok=True)
        paths = {hemisphere: output / f"{hemisphere}.sphere.MSMAll.native.surf.gii"
                 for hemisphere in "LR"}
        for hemisphere, path in paths.items():
            shutil.copyfile(inputs[hemisphere].source_sphere, path)
        return paths
    monkeypatch.setattr(fnit.msm, "run_msmall", solve)
    monkeypatch.setattr("fnit.fmri.surface_pipeline.shutil.which", lambda command: "/bin/wb_command")


def test_fsLR_warp_is_composed_on_native_msmsulc_before_projection(tmp_path, monkeypatch):
    inputs, native_spheres, geometry, assets = setup_case(tmp_path, False)
    observed = {}; commands = []
    mock_solver(monkeypatch, observed)
    def compose(command, **options):
        commands.append(command)
        shutil.copyfile(command[2], command[5])
    monkeypatch.setattr("fnit.fmri.surface_pipeline.subprocess.run",
                        compose)
    result, topology = _refine_msmall(inputs, native_spheres, geometry, assets,
                                      tmp_path / "result", MSMAllConfig(), "cpu", "optimized", "wb_command", parallel=False)
    assert topology == {"L": "fsLR32k", "R": "fsLR32k"}
    for hemisphere, initial_native, command, output in zip("LR", native_spheres, commands, result):
        assert command[1:4] == ["-surface-sphere-project-unproject", str(initial_native),
                                str(inputs[hemisphere].source_sphere)]
        assert Path(command[4]) != output  # Preserve the solver's 32k sphere.
        assert command[5] == str(output)
        assert observed["inputs"][hemisphere].initial_sphere is None  # Identity in MSMSulc 32k space.
    report = json.loads((tmp_path / "result/msmall/native_composition_report.json").read_text())
    assert report["orientation_qc"] == "pass"
    assert report["L"]["vertex_count"] == len(nib.load(native_spheres[0]).darrays[0].data)
    assert report["L"]["baseline"] == "input MSMSulc native sphere"


def test_native_features_use_initial_native_registration_without_atlas_composition(tmp_path, monkeypatch):
    inputs, native_spheres, geometry, assets = setup_case(tmp_path, True)
    observed = {}; mock_solver(monkeypatch, observed)
    def forbidden(*args, **kwargs):
        raise AssertionError("native topology must not be composed as a 32k warp")
    monkeypatch.setattr("fnit.fmri.surface_pipeline.subprocess.run", forbidden)
    _, topology = _refine_msmall(inputs, native_spheres, geometry, assets, tmp_path / "result",
                                MSMAllConfig(), "cpu", "optimized", "wb_command", parallel=False)
    assert topology == {"L": "native", "R": "native"}
    assert observed["inputs"]["L"].initial_sphere == native_spheres[0]


def test_native_feature_topology_uses_registration_sphere_not_midthickness_order(tmp_path, monkeypatch):
    inputs, native_spheres, geometry, assets = setup_case(tmp_path, True)
    # Workbench exports can preserve the native vertex set while serializing
    # midthickness triangles in another order.  MSMAll features still follow
    # the registration sphere's order and must not be rejected for that.
    reordered = []
    for item in geometry:
        faces = np.asarray(nib.load(item.midthickness).darrays[1].data).copy()[:, ::-1]
        path = tmp_path / f"{len(reordered)}.midthickness.reordered.surf.gii"
        points = np.asarray(nib.load(item.midthickness).darrays[0].data)
        nib.save(nib.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(points, intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE"),
        ]), path)
        reordered.append(SimpleNamespace(midthickness=path))
    observed = {}; mock_solver(monkeypatch, observed)
    monkeypatch.setattr("fnit.fmri.surface_pipeline.subprocess.run",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("native MSMAll must not call Workbench composition")))
    _, topology = _refine_msmall(inputs, native_spheres, tuple(reordered), assets,
                                tmp_path / "result", MSMAllConfig(), "cpu", "optimized",
                                "wb_command", parallel=False)
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
                      MSMAllConfig(), "cpu", "optimized", "wb_command", parallel=False)


def test_final_composed_native_fold_is_reported_without_changing_coordinates(tmp_path, monkeypatch):
    inputs, native_spheres, geometry, assets = setup_case(tmp_path, False)
    mock_solver(monkeypatch, {})
    changed = {}
    def compose(command, **options):
        image = nib.load(command[2])
        points = np.asarray(image.darrays[0].data).copy()
        if Path(command[5]).name.startswith("L."):
            points[[0, 1]] = points[[1, 0]]
        image.darrays[0].data = points
        nib.save(image, command[5])
        changed[command[5]] = points
    monkeypatch.setattr("fnit.fmri.surface_pipeline.subprocess.run", compose)
    references = {hemi: save_mesh(tmp_path / f"{hemi}.baseline.surf.gii", 2) for hemi in "LR"}
    result, _ = _refine_msmall(
        inputs, native_spheres, geometry, assets, tmp_path / "result", MSMAllConfig(),
        "cpu", "optimized", "wb_command", parallel=False, native_qc_references=references,
    )
    report = json.loads((tmp_path / "result/msmall/native_composition_report.json").read_text())
    assert report["L"]["folded_output_faces"] > 0
    assert report["L"]["absolute_folded_output_faces"] > 0
    assert report["L"]["baseline"] == "undeformed native sphere"
    assert report["orientation_qc"] == "warning"
    assert report["R"]["orientation_qc"] == "pass"
    for path in result:
        np.testing.assert_array_equal(nib.load(path).darrays[0].data, changed[str(path)])


def test_absolute_check_detects_a_fold_already_present_in_relative_baseline(tmp_path):
    path = save_mesh(tmp_path / "folded.surf.gii", 2)
    image = nib.load(path)
    image.darrays[0].data[[0, 1]] = image.darrays[0].data[[1, 0]]
    nib.save(image, path)
    qc = _saved_native_sphere_qc(path, path, baseline="already folded input sphere")
    assert qc["folded_output_faces"] == 0
    assert qc["absolute_folded_output_faces"] == qc["absolute_folded_input_faces"] > 0
    assert qc["orientation_qc"] == "warning"


def test_orientation_chain_preserves_initial_warning_after_clean_msmall_output():
    initial = {"L": {"folded_output_faces": 2}, "R": {"folded_output_faces": 0}}
    clean = {hemi: {"folded_output_faces": 0} for hemi in "LR"}
    qc = _orientation_chain(initial, clean, clean)
    assert qc["initial_msmsulc"]["status"] == "warning"
    assert qc["solver_msmall"]["status"] == qc["final_native"]["status"] == "pass"
    assert qc["all_stages"] == "warning"


def test_orientation_chain_does_not_promote_missing_initial_qc_to_pass():
    clean = {hemi: {"folded_output_faces": 0} for hemi in "LR"}
    qc = _orientation_chain(None, None, clean)
    assert qc["initial_msmsulc"]["status"] == "not_assessed"
    assert qc["solver_msmall"]["status"] == "not_applicable"
    assert qc["all_stages"] == "not_assessed"


def test_saved_native_qc_rejects_changed_face_order(tmp_path):
    reference = save_mesh(tmp_path / "reference.surf.gii", 2)
    output = save_mesh(tmp_path / "output.surf.gii", 2)
    image = nib.load(output)
    image.darrays[1].data = image.darrays[1].data[::-1].copy()
    nib.save(image, output)
    with pytest.raises(ValueError, match="reference vertex order and topology"):
        _saved_native_sphere_qc(output, reference, baseline="reference sphere")


def _local_fold(path):
    """Cross one adjacent edge to exercise unfolding with a small local fold."""
    image = nib.load(path)
    points = np.asarray(image.darrays[0].data).copy()
    first, second, third = image.darrays[1].data[0]
    moved = 0.55 * points[second] + 0.55 * points[third] - 0.1 * points[first]
    points[first] = moved * (100 / np.linalg.norm(moved))
    image.darrays[0].data = points
    nib.save(image, path)


@pytest.mark.parametrize("native_features", [False, True])
def test_final_native_repair_rechecks_saved_float32_and_preserves_solver(
    tmp_path, monkeypatch, native_features,
):
    inputs, native_spheres, geometry, assets = setup_case(tmp_path, native_features)
    mock_solver(monkeypatch, {})
    if native_features:
        import fnit.msm
        solve = fnit.msm.run_msmall
        def folded_solver(*args, **kwargs):
            paths = solve(*args, **kwargs)
            _local_fold(paths["L"])
            return paths
        monkeypatch.setattr(fnit.msm, "run_msmall", folded_solver)
    else:
        def compose(command, **options):
            shutil.copyfile(command[2], command[5])
            if Path(command[5]).name.startswith("L."):
                _local_fold(command[5])
        monkeypatch.setattr("fnit.fmri.surface_pipeline.subprocess.run", compose)
    references = dict(zip("LR", native_spheres))
    result, topology = _refine_msmall(
        inputs, native_spheres, geometry, assets, tmp_path / "result", MSMAllConfig(),
        "cpu", "optimized", "wb_command", parallel=False,
        native_qc_references=references, qc_policy="repair",
    )
    report = json.loads((tmp_path / "result/msmall/native_composition_report.json").read_text())
    assert report["qc_policy"] == "repair"
    assert report["orientation_qc"] == "pass"
    left = report["L"]
    assert left["native_output_qc_before_repair"]["absolute_folded_output_faces"] == 1
    assert left["absolute_folded_output_faces"] == left["folded_output_faces"] == 0
    assert left["fold_repair"]["success"] is True
    assert left["fold_repair"]["moved_vertices"] > 0
    assert left["fold_repair"]["unfold_updates"] == 0
    assert left["fold_repair"]["guard_updates"] > 0
    assert left["fold_repair"]["attempts"][0]["stage"] == "direct_guard"
    assert left["coordinates"] == "saved GIFTI coordinates"
    assert report["R"]["fold_repair"]["applied"] is False
    assert topology == dict.fromkeys("LR", "native" if native_features else "fsLR32k")
    assert nib.load(result[0]).darrays[0].data.dtype == np.float32
    source = tmp_path / "result/msmall/L.sphere.MSMAll.native.surf.gii"
    if native_features:
        assert result[0] != source
        assert _saved_native_sphere_qc(source, native_spheres[0], baseline="reference")[
            "absolute_folded_output_faces"
        ] == 1
    else:
        np.testing.assert_array_equal(nib.load(source).darrays[0].data,
                                      nib.load(inputs["L"].source_sphere).darrays[0].data)


def test_final_native_error_refuses_folded_composition_without_changing_it(tmp_path, monkeypatch):
    inputs, native_spheres, geometry, assets = setup_case(tmp_path, False)
    mock_solver(monkeypatch, {})
    def compose(command, **options):
        shutil.copyfile(command[2], command[5])
        if Path(command[5]).name.startswith("L."):
            _local_fold(command[5])
    monkeypatch.setattr("fnit.fmri.surface_pipeline.subprocess.run", compose)
    with pytest.raises(RuntimeError, match="BOLD projection refused"):
        _refine_msmall(
            inputs, native_spheres, geometry, assets, tmp_path / "result", MSMAllConfig(),
            "cpu", "optimized", "wb_command", parallel=False, qc_policy="error",
        )
    report = json.loads((tmp_path / "result/msmall/native_composition_report.json").read_text())
    assert report["orientation_qc"] == "warning"
    assert report["L"]["fold_repair"]["applied"] is False
    assert report["L"]["absolute_folded_output_faces"] == 1


def test_failed_native_repair_is_rejected_after_saved_gifti_check(tmp_path, monkeypatch):
    inputs, native_spheres, geometry, assets = setup_case(tmp_path, False)
    mock_solver(monkeypatch, {})
    def compose(command, **options):
        shutil.copyfile(command[2], command[5])
        if Path(command[5]).name.startswith("L."):
            _local_fold(command[5])
    monkeypatch.setattr("fnit.fmri.surface_pipeline.subprocess.run", compose)
    def ineffective_repair(points, faces, reference):
        import torch
        assert points.dtype == torch.float64 and points.device.type == "cpu"
        # A claimed helper success cannot substitute for the saved GIFTI QC.
        return points.clone(), {"applied": True, "success": True, "unfold_updates": 1}
    monkeypatch.setattr("fnit.msm._native_repair.repair_native_sphere", ineffective_repair)
    with pytest.raises(RuntimeError, match="fold repair did not pass QC"):
        _refine_msmall(
            inputs, native_spheres, geometry, assets, tmp_path / "result", MSMAllConfig(),
            "cpu", "optimized", "wb_command", parallel=False, qc_policy="repair",
        )
    report = json.loads((tmp_path / "result/msmall/native_composition_report.json").read_text())
    assert report["L"]["fold_repair"]["success"] is False
    assert report["L"]["fold_repair"]["applied"] is True
    assert report["L"]["absolute_folded_output_faces"] == 1


def test_unknown_final_native_policy_fails_before_solver(tmp_path, monkeypatch):
    inputs, native_spheres, geometry, assets = setup_case(tmp_path, False)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid policy must fail before registration")
    monkeypatch.setattr("fnit.msm.run_msmall", forbidden)
    with pytest.raises(ValueError, match="qc_policy must be"):
        _refine_msmall(inputs, native_spheres, geometry, assets, tmp_path / "result",
                      MSMAllConfig(), "cpu", "optimized", "wb_command", qc_policy="invalid")


def test_double_precision_repair_cannot_pass_with_a_fold_in_saved_float32(tmp_path, monkeypatch):
    inputs, native_spheres, geometry, assets = setup_case(tmp_path, False)
    mock_solver(monkeypatch, {})
    def compose(command, **options):
        shutil.copyfile(command[2], command[5])
        if Path(command[5]).name.startswith("L."):
            _local_fold(command[5])
    monkeypatch.setattr("fnit.fmri.surface_pipeline.subprocess.run", compose)
    def near_edge_repair(points, faces, reference):
        import torch
        from fnit.msm.msmsulc import _native_output_qc
        original = np.asarray(nib.load(native_spheres[0]).darrays[0].data, dtype=np.float64)
        candidate = original.copy()
        first, second, third = 43, 14, 44
        candidate[first] = (original[second] + original[third]) / 2 + 1e-10 * original[first]
        candidate[first] *= 100 / np.linalg.norm(candidate[first])
        precision = _native_output_qc(candidate, faces, original)
        assert precision["folded_solver_faces"] == 0
        assert precision["folded_output_faces"] == 1
        return torch.as_tensor(candidate, dtype=points.dtype, device=points.device), {
            "applied": True, "success": True, "unfold_updates": 1,
        }
    monkeypatch.setattr("fnit.msm._native_repair.repair_native_sphere", near_edge_repair)
    with pytest.raises(RuntimeError, match="fold repair did not pass QC"):
        _refine_msmall(inputs, native_spheres, geometry, assets, tmp_path / "result",
                      MSMAllConfig(), "cpu", "optimized", "wb_command", parallel=False,
                      qc_policy="repair")
    report = json.loads((tmp_path / "result/msmall/native_composition_report.json").read_text())
    assert report["L"]["fold_repair"]["success"] is False
    assert report["L"]["folded_output_faces"] == 1
