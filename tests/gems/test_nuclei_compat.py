"""旧 nuclei 名称和参数只路由到统一 PyTorch 亚区入口。"""

import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

import fnit
from fnit import gems
from fnit.gems import nuclei, nuclei_cli, pipeline


def test_compatibility_exports_keep_old_signature():
    assert gems.segment_subregions is fnit.segment_subregions is pipeline.segment_subregions
    assert gems.segment_nuclei is fnit.segment_nuclei is nuclei.segment_nuclei
    assert gems.prepare_nuclei_atlas is nuclei.prepare_nuclei_atlas
    assert "segment_nuclei" not in gems.__all__ and "prepare_nuclei_atlas" not in gems.__all__
    parameters = inspect.signature(nuclei.segment_nuclei).parameters
    assert list(parameters)[:5] == ["norm", "aseg", "wmparc", "atlas_root", "output_dir"]
    assert parameters["structures"].default == ("thalamus", "hippo-left", "hippo-right")
    assert parameters["threads"].default == 4


def test_prepare_legacy_atlas_keyword_routes_once(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(nuclei, "prepare_subregion_atlases", lambda **kwargs: calls.append(kwargs) or tmp_path)
    assert nuclei.prepare_nuclei_atlas(atlas_root=tmp_path, asset_dir="verified_cache") == tmp_path
    assert calls == [{"output_root": tmp_path, "asset_dir": "verified_cache"}]


@pytest.fixture
def legacy_result(monkeypatch, tmp_path):
    rows = ((8101, "Left-AV", "thalamus", "left", "thalamus", 1.25),
            (8201, "Right-AV", "thalamus", "right", "thalamus", 2.5),
            (236, "Left-CA1-body", "hippocampus", "left", "hippo-amygdala-left", 3.75),
            (237, "Left-CA1-head", "hippocampus", "left", "hippo-amygdala-left", 4.5),
            (215, "Left-hippocampal-fissure", "hippocampus", "left", "hippo-amygdala-left", .7),
            (7001, "Left-Lateral-nucleus", "amygdala", "left", "hippo-amygdala-left", 6.),
            (10236, "Right-CA1-body", "hippocampus", "right", "hippo-amygdala-right", 7.5),
            (17001, "Right-Lateral-nucleus", "amygdala", "right", "hippo-amygdala-right", 8.5))
    metadata = {label: pipeline.SubregionLabel(label, name, parent, source, side, source)
                for label, name, parent, side, source, _ in rows}
    data = np.zeros((3, 3, 3), np.int32)
    data.ravel()[:len(rows)] = [row[0] for row in rows]
    fits = {}
    atlases = tmp_path / "atlases"
    for source in nuclei._STRUCTURES.values():
        fine = np.asarray([row[0] for row in rows if row[4] == source] + [0], np.int32).reshape(-1, 1, 1)
        fits[source] = SimpleNamespace(highres_labels=nib.Nifti1Image(fine, np.diag([.5, .5, .5, 1.])))
        (atlases / source).mkdir(parents=True)
        for filename in nuclei.ATLAS_FILES:
            (atlases / source / filename).touch()
    result = pipeline.SubregionResult(
        labels=nib.Nifti1Image(data, np.eye(4)), label_table={row[0]: row[1] for row in rows},
        structure_results=fits, confidence=torch.zeros(data.shape), initialization={},
        label_metadata=metadata,
        volumes={row[0]: {"hard_volume_mm3": 1., "soft_volume_mm3": row[5]} for row in rows})
    calls, threads = [], []

    def segment(**kwargs):
        calls.append(kwargs)
        result.save(kwargs["output_dir"], save_highres=kwargs["save_highres"])
        return result

    monkeypatch.setattr(nuclei, "segment_subregions", segment)
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    return result, atlases, calls, threads


def test_legacy_five_positional_arguments_write_nested_labels_and_soft_tables(legacy_result, tmp_path):
    result, atlases, calls, threads = legacy_result
    output = tmp_path / "output"
    old_data = np.asanyarray(result.labels.dataobj).copy()
    files = nuclei.segment_nuclei("norm.mgz", "aseg.mgz", "wmparc.mgz", atlases, output,
                                  device="cpu", optimization="balanced")
    assert len(calls) == 1 and threads == [4]
    assert calls[0] == {"t1": "norm.mgz", "atlas_root": atlases, "coarse_segmentation": "aseg.mgz",
                        "wmparc": "wmparc.mgz", "structures": tuple(nuclei._STRUCTURES.values()),
                        "device": "cpu", "optimization": "balanced", "output_dir": output, "save_highres": True}
    assert set(files) == {"thalamus", "hippo-left", "hippo-right"}
    assert set(files["thalamus"]) == {"labels", "high_resolution_labels", "volumes"}
    assert set(files["hippo-right"]) == {"labels", "high_resolution_labels", "hippocampal_volumes", "amygdala_volumes"}
    for paths in files.values():
        assert all(path.is_file() for path in paths.values())
        native = nib.load(paths["labels"])
        assert native.shape == result.labels.shape and np.array_equal(native.affine, result.labels.affine)
        assert native.get_data_dtype() == np.dtype("int32")
    for key in ("labels", "high_resolution_labels"):
        right = np.asanyarray(nib.load(files["hippo-right"][key]).dataobj)
        assert set(np.unique(right)) == {0, 236, 7001}
    np.testing.assert_array_equal(result.labels.dataobj, old_data)
    modern = np.asanyarray(nib.load(output / "subregions_native.nii.gz").dataobj)
    np.testing.assert_array_equal(modern, old_data)
    assert 10236 in np.unique(nib.load(output / "highres/hippo-amygdala-right.nii.gz").dataobj)
    volumes = lambda path: {name: float(volume) for name, volume in (line.split() for line in path.read_text().splitlines())}
    hippo = volumes(files["hippo-left"]["hippocampal_volumes"])
    assert hippo == {"CA1-body": 3.75, "CA1-head": 4.5, "hippocampal-fissure": .7,
                     "Whole_hippocampus": 8.25, "Whole_hippocampal_body": 3.75, "Whole_hippocampal_head": 4.5}
    assert volumes(files["thalamus"]["volumes"])["Right-Whole_thalamus"] == 2.5
    assert volumes(files["hippo-right"]["amygdala_volumes"]) == {"Lateral-nucleus": 8.5, "Whole_amygdala": 8.5}


def test_old_atlas_layout_prepares_canonical_packs_before_single_fit(legacy_result, monkeypatch, tmp_path):
    _, _, calls, _ = legacy_result
    old_root = tmp_path / "old_atlas"
    (old_root / "average/HippoSF/atlas").mkdir(parents=True)
    prepared = []
    monkeypatch.setattr(nuclei, "prepare_nuclei_atlas", lambda root, **kwargs: prepared.append((root, kwargs)) or root)
    nuclei.segment_nuclei(norm="norm.mgz", aseg="aseg.mgz", wmparc="wmparc.mgz", atlas_root=old_root,
                         output_dir=tmp_path / "output", structures=("hippo-right",), device="cpu")
    assert prepared == [(old_root, {"asset_dir": old_root})] and len(calls) == 1
    assert calls[0]["structures"] == ("hippo-amygdala-right",)


@pytest.mark.parametrize("available,device", [(False, "cpu"), (True, "cuda:0")])
def test_legacy_default_device_selects_available_backend(legacy_result, monkeypatch, tmp_path, available, device):
    _, atlases, calls, _ = legacy_result
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    nuclei.segment_nuclei("norm", "aseg", "wmparc", atlases, tmp_path / "output")
    assert len(calls) == 1 and calls[0]["device"] == device


@pytest.mark.parametrize("options", [{"threads": 0}, {"structures": ()}, {"structures": ("unknown",)}])
def test_invalid_legacy_parameters_do_not_prepare_or_fit(legacy_result, options, tmp_path):
    _, atlases, calls, threads = legacy_result
    with pytest.raises(ValueError):
        nuclei.segment_nuclei("norm", "aseg", "wmparc", atlases, tmp_path / "output", **options)
    assert not calls and not threads


@pytest.fixture
def routed_call(monkeypatch):
    calls = []
    threads = []
    result = pipeline.SubregionResult(
        labels=nib.Nifti1Image(np.zeros((2, 2, 2), np.int32), np.eye(4)),
        label_table={}, structure_results={}, confidence=torch.zeros((2, 2, 2)), initialization={})

    def segment(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(nuclei_cli, "segment_subregions", segment)
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    return calls, threads


def test_raw_t1_cli_routes_all_four_structures_and_save_options(routed_call, capsys):
    calls, threads = routed_call
    assert nuclei_cli.main(["run", "--t1", "raw_t1.nii.gz", "--atlas-root", "atlases",
                           "--output-dir", "result"]) is None
    assert calls == [(("raw_t1.nii.gz", "atlases"), {
        "coarse_segmentation": None, "wmparc": None, "structures": "all",
        "device": "cuda:0", "optimization": "fast", "output_dir": "result", "save_highres": True})]
    assert threads == [4]
    assert capsys.readouterr().out.strip() == "result"


def test_legacy_norm_cli_maps_side_names_and_optional_prepared_inputs(routed_call):
    calls, threads = routed_call
    nuclei_cli.main(["run", "--norm", "norm.mgz", "--aseg", "aseg.mgz", "--wmparc", "wmparc.mgz",
                    "--atlas-root", "atlases", "--output-dir", "result", "--structure", "thalamus",
                    "--structure", "hippo-left", "--structure", "hippo-right", "--device", "cpu",
                    "--optimization", "balanced", "--threads", "2"])
    assert calls == [(("norm.mgz", "atlases"), {
        "coarse_segmentation": "aseg.mgz", "wmparc": "wmparc.mgz",
        "structures": ("thalamus", "hippo-amygdala-left", "hippo-amygdala-right"),
        "device": "cpu", "optimization": "balanced", "output_dir": "result", "save_highres": True})]
    assert threads == [2]


@pytest.mark.parametrize("available,device", [(False, "cpu"), (True, "cuda:0")])
def test_legacy_norm_cli_default_device_is_available_backend(routed_call, monkeypatch, available, device):
    calls, _ = routed_call
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    nuclei_cli.main(["run", "--norm", "norm.mgz", "--atlas-root", "atlases", "--output-dir", "result"])
    assert len(calls) == 1 and calls[0][1]["device"] == device


@pytest.mark.parametrize("extra", [[], ["--t1", "t1.nii.gz", "--norm", "norm.mgz"],
                                   ["--t1", "t1.nii.gz", "--threads", "0"]])
def test_invalid_input_or_threads_do_not_run_model(routed_call, extra):
    calls, threads = routed_call
    with pytest.raises(SystemExit) as error:
        nuclei_cli.main(["run", "--atlas-root", "atlases", "--output-dir", "result", *extra])
    assert error.value.code == 2
    assert not calls and not threads


def test_legacy_setup_routes_to_unified_atlases_and_saves_configuration(monkeypatch, tmp_path):
    calls = []
    configured = []

    def prepare(output_root, **kwargs):
        calls.append((output_root, kwargs))
        return Path(output_root)

    monkeypatch.setattr(nuclei_cli, "prepare_subregion_atlases", prepare)
    monkeypatch.setattr(nuclei_cli, "save_subregion_root", configured.append)
    nuclei_cli.main(["setup", "--atlas-root", str(tmp_path), "--asset-dir", "verified_cache"])
    assert calls == [(str(tmp_path), {"asset_dir": "verified_cache", "device": "cpu"})]
    assert configured == [tmp_path]


def test_alias_imports_and_cli_help_avoid_native_and_external_backends():
    source = Path(__file__).resolve().parents[2] / "src"
    code = """
import importlib.abc
import sys

class RejectLegacyBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("fnit.gems.native_samseg") or fullname.split(".")[0] in {
            "surfa", "fsl", "freesurfer", "nipype", "dipy", "spm", "mrtrix3", "afni"}:
            raise AssertionError("unexpected backend import: " + fullname)

sys.meta_path.insert(0, RejectLegacyBackends())
from fnit import segment_nuclei, segment_subregions
from fnit.gems import nuclei, nuclei_cli, prepare_nuclei_atlas, prepare_subregion_atlases
assert segment_nuclei is nuclei.segment_nuclei
assert prepare_nuclei_atlas is nuclei.prepare_nuclei_atlas
for argv in (["--help"], ["run", "--help"], ["setup", "--help"]):
    try:
        nuclei_cli.main(argv)
    except SystemExit as error:
        assert error.code == 0
    else:
        raise AssertionError("help did not exit")
assert not any(name.startswith("fnit.gems.native_samseg") for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(source)},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
