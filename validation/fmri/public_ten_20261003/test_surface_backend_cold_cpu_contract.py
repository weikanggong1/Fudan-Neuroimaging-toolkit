"""Cold CPU helper 的输出保护合同；这些小测试不作为 MRI benchmark。"""

import importlib.util
import inspect
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).with_name("run_surface_backend_cold_cpu_diagnostic.py")
    spec = importlib.util.spec_from_file_location("cold_cpu_contract_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_indexed_run_child_is_allowed_without_writing(runner, tmp_path):
    root = tmp_path / "FNIT"
    target = root / "runs/backend/new-attempt"
    runner.protect_output(root, target, [root / "workspaces/source", root / "runs/formal"])
    assert not root.exists()


@pytest.mark.parametrize("location", ["FNIT", "FNIT/runs", "outside/run"])
def test_non_run_or_parent_output_is_rejected_before_writing(runner, tmp_path, location):
    root = tmp_path / "FNIT"
    with pytest.raises(ValueError):
        runner.protect_output(root, tmp_path / location, [])
    assert not root.exists()


@pytest.mark.parametrize("suffix", ["formal", "formal/child", "parent"])
def test_original_run_and_its_parents_or_children_are_protected(runner, tmp_path, suffix):
    root = tmp_path / "FNIT"
    protected = root / "runs/parent/formal"
    target = root / "runs/parent" / suffix if suffix != "parent" else root / "runs/parent"
    with pytest.raises(ValueError):
        runner.protect_output(root, target, [protected])
    assert not root.exists()


def test_symlink_route_to_original_is_rejected(runner, tmp_path):
    root = tmp_path / "FNIT"
    (root / "runs").mkdir(parents=True)
    source = root / "workspaces/frozen"
    source.mkdir(parents=True)
    (root / "runs/alias").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError):
        runner.protect_output(root, root / "runs/alias/new", [source])
    assert not (source / "new").exists()


def test_existing_attempt_is_never_overwritten(runner, tmp_path):
    root = tmp_path / "FNIT"
    target = root / "runs/backend/attempt"
    target.mkdir(parents=True)
    marker = target / "report.json"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        runner.protect_output(root, target, [])
    assert marker.read_text() == "keep"


def test_cold_route_uses_the_public_complete_api_contract():
    from fnit.fmri import fMRISurface_pipeline

    signature = inspect.signature(fMRISurface_pipeline)
    assert {"device", "cpu_threads", "auto_volume", "recon_all_backend", "recon_all_output_dir",
            "recon_all_options", "volume_options"}.issubset(signature.parameters)
    assert signature.parameters["recon_all"].default is None


def test_real_surface_result_paths_are_serialized_without_losing_fields(runner, tmp_path):
    import dataclasses
    import json
    from fnit.fmri.surface_pipeline import FMRISurfaceResult

    result = FMRISurfaceResult(
        left=tmp_path / "L.func.gii", right=tmp_path / "R.func.gii",
        dtseries=tmp_path / "bold.dtseries.nii", metadata=tmp_path / "bold.json",
        timing_seconds={"total": 12.5}, qc_report=tmp_path / "report.json",
        registered_spheres=(tmp_path / "L.surf.gii", tmp_path / "R.surf.gii"),
        recon_all=tmp_path / "reconstruction/subject", volume_executed=False)
    destination = tmp_path / "files.private.json"
    runner.save(destination, {"result": dataclasses.asdict(result),
                              "configuration": {"owned_output": tmp_path / "derivatives"}})
    actual = json.loads(destination.read_text())
    assert actual["result"] == {
        "left": str(result.left), "right": str(result.right), "dtseries": str(result.dtseries),
        "metadata": str(result.metadata), "timing_seconds": {"total": 12.5},
        "qc_report": str(result.qc_report), "registered_spheres": [str(path) for path in result.registered_spheres],
        "recon_all": str(result.recon_all), "volume_executed": False}
    assert actual["configuration"] == {"owned_output": str(tmp_path / "derivatives")}
    assert not destination.with_suffix(".json.tmp").exists()


def test_serializer_rejects_unknown_objects_instead_of_inventing_values(runner, tmp_path):
    with pytest.raises(TypeError, match="not JSON serializable"):
        runner.save(tmp_path / "invalid.json", {"value": object()})
    assert not (tmp_path / "invalid.json").exists()
