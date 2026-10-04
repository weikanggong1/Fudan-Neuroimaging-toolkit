"""Benchmark-harness contracts only; no external program or MRI benchmark."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest


def load_driver():
    path = Path(__file__).with_name("run_reference_benchmark.py")
    spec = importlib.util.spec_from_file_location("reference_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bound(module, path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": module.sha256(path)}


def test_driver_import_does_not_import_torch_and_official_child_compiles():
    module = load_driver()
    tree = ast.parse(module.OFFICIAL_SCRIPT)
    # The generated child must append a real newline, not literal backslash-n.
    # Literal backslash-n makes the saved execution JSON fail json.loads.
    suffix = tree.body[-1].value.args[0].right
    assert ast.literal_eval(suffix) == "\n"
    script = Path(module.__file__).resolve()
    subprocess.run([sys.executable, "-c", "import runpy,sys;runpy.run_path(sys.argv[1]);assert 'torch' not in sys.modules", str(script)], check=True)


def test_metrics_full_grid_identity_and_undefined_domains(tmp_path):
    module = load_driver()
    path = tmp_path / "zero.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4)), path)
    report = module.compare_reference(path, path)
    assert report["full_FOV"]["RMSE"] == 0
    assert report["full_FOV"]["spatial_pearson_defined"] is False
    assert report["full_FOV"]["NRMSE_defined"] is False
    assert report["reference_nonzero"]["status"] == "empty_domain"
    changed = tmp_path / "shifted.nii.gz"
    affine = np.eye(4)
    affine[0, 3] = 1
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), affine), changed)
    with pytest.raises(ValueError, match="physical grids"):
        module.compare_reference(changed, path)


def test_output_cannot_overlap_source_or_raw_and_attempt_is_not_overwritten(tmp_path):
    module = load_driver()
    run_root = tmp_path / "runs"
    run_root.mkdir()
    protected = run_root / "source"
    protected.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        module.protect_output(protected / "attempt", run_root, [protected])
    with pytest.raises(ValueError, match="descendant"):
        module.protect_output(tmp_path / "outside", run_root, [protected])
    existing = run_root / "attempt"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="retain"):
        module.protect_output(existing, run_root, [protected])


def make_manifest(tmp_path, module):
    source = tmp_path / "source"
    source_file = source / "src/fnit/fmri/reference.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# bound source fixture\n")
    data = np.arange(120, dtype=np.float32).reshape(4, 5, 6) + 1
    bold = tmp_path / "bold.nii.gz"
    original = tmp_path / "original.nii.gz"
    nib.save(nib.Nifti1Image(np.repeat(data[..., None], 180, axis=3), np.eye(4)), bold)
    nib.save(nib.Nifti1Image(data, np.eye(4)), original)
    rst = tmp_path / "node.rst"
    rst.write_text("original node")
    container = tmp_path / "container.sif"
    container.write_text("container fixture; not executable")
    manifest = {"cases": {"CON01": {"frames": 180, "tr_seconds": 2.1,
                "raw": {"bold": bound(module, bold)}, "official": {"bold_reference": {
                    "input": bound(module, bold), "output": bound(module, original),
                    "report": bound(module, rst), "selected_frames_zero_based": list(range(20, 40)),
                    "runtime": {"duration": 13., "hostname": "CPU-reference"}}}}},
                "tools": {"official_container": bound(module, container)}}
    manifest_path = tmp_path / "manifest.private.json"
    manifest_path.write_text(json.dumps(manifest))
    runs = tmp_path / "runs"
    runs.mkdir()
    args = ["--manifest", str(manifest_path), "--case", "CON01", "--phase", "candidate",
            "--output", str(runs / "attempt"), "--allowed-run-root", str(runs),
            "--source-root", str(source), "--device", "cpu",
            "--expected-reference-sha256", module.sha256(source_file)]
    return args, original, rst, runs


def test_real_schema_mapping_and_all_readonly_guards(tmp_path, monkeypatch):
    module = load_driver()
    args, original, rst, runs = make_manifest(tmp_path, module)
    monkeypatch.setattr(module, "candidate", lambda *a: {"reference": str(original),
                                                        "selected_indices": list(range(20, 40)),
                                                        "candidate_reference_API_wall_seconds": 1.})
    module.main(args)
    report = json.loads((runs / "attempt/report.public.json").read_text())
    assert report["status"] == "complete"
    assert report["original_reference_runtime"]["duration"] == 13.
    assert report["input_guards_equal"] is True
    assert report["source_guards_equal"] is True
    assert report["comparison_to_original_reference"]["full_FOV"]["NRMSE"] == 0
    assert str(tmp_path) not in json.dumps(report)


def test_changed_bound_original_node_is_retained_failed_attempt(tmp_path, monkeypatch):
    module = load_driver()
    args, original, rst, runs = make_manifest(tmp_path, module)

    def changed(*a):
        rst.write_text("mutated original")
        return {"reference": str(original), "selected_indices": list(range(20, 40))}

    monkeypatch.setattr(module, "candidate", changed)
    with pytest.raises(RuntimeError, match="changed"):
        module.main(args)
    report = json.loads((runs / "attempt/report.public.json").read_text())
    assert report["status"] == "failed" and report["error_type"] == "RuntimeError"
    assert (runs / "attempt/failure.private.txt").exists()
    assert str(tmp_path) not in json.dumps(report)
