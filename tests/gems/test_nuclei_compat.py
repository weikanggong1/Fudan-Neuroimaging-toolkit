"""旧 nuclei 名称和参数只路由到统一 PyTorch 亚区入口。"""

import inspect
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

import fnit
from fnit import gems
from fnit.gems import nuclei, nuclei_cli, pipeline, setup


def test_aliases_are_canonical_function_objects():
    assert nuclei.segment_nuclei is pipeline.segment_subregions
    assert gems.segment_nuclei is gems.segment_subregions is fnit.segment_subregions
    assert fnit.segment_nuclei is pipeline.segment_subregions
    assert nuclei.prepare_nuclei_atlas is setup.prepare_subregion_atlases
    assert gems.prepare_nuclei_atlas is gems.prepare_subregion_atlases
    assert "segment_nuclei" not in gems.__all__ and "prepare_nuclei_atlas" not in gems.__all__
    assert inspect.signature(nuclei.segment_nuclei) == inspect.signature(pipeline.segment_subregions)


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
assert segment_nuclei is segment_subregions is nuclei.segment_nuclei
assert prepare_nuclei_atlas is prepare_subregion_atlases
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
