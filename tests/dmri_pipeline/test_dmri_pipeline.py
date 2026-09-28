import argparse
import hashlib
from types import SimpleNamespace
from importlib.resources import files
import nibabel as nib
import numpy as np
import pytest
import torch

import fnit.dmri_pipeline.pipeline as pipeline_module
import fnit.dmri_pipeline.tbss as tbss_module

from fnit.dmri_pipeline import DMRIPipeline, STANDARD_MAP_NAMES
from fnit.dmri_pipeline.cli import _arguments
from fnit.dmri_pipeline.tbss import TBSSConfig, preprocess_fa
from fnit.eddy.core import _load_topup_field


def test_common_nine_map_contract():
    assert STANDARD_MAP_NAMES == (
        "FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF"
    )


def test_bvec_source_cli_and_invalid_value():
    parser = argparse.ArgumentParser()
    _arguments(parser)
    options = parser.parse_args(
        ["--raw-dir", "raw", "-o", "out", "--fa-template", "fa.nii.gz",
         "--bvec-source", "raw"]
    )
    assert options.bvec_source == "raw"
    with pytest.raises(ValueError, match="bvec_source"):
        DMRIPipeline(device="cpu", bvec_source="invalid")


def test_tbss_config_combines_selected_official_schedule_values():
    config = TBSSConfig().fnirt
    assert config.subsampling == (8, 4, 2, 2, 1, 1)
    assert config.maximum_iterations == (5, 5, 5, 5, 50, 25)
    assert config.warp_resolution_schedule_mm[-2:] == (
        (2.0, 2.0, 2.0),
        (2.0, 2.0, 2.0),
    )


def test_tbss_preprocessing_caps_erodes_and_builds_boundary_weight():
    data = np.zeros((9, 9, 9), dtype=np.float32)
    data[2:7, 2:7, 2:7] = 2
    image = nib.Nifti1Image(data, np.eye(4))
    fa, weight = preprocess_fa(image)
    values = np.asarray(fa.dataobj)
    weights = np.asarray(weight.dataobj)
    assert values.max() == 1
    assert values[2, 4, 4] == 0
    assert values[3, 4, 4] == 1
    assert set(np.unique(weights)).issubset({0, 1})
    assert weights[4, 4, 4] == 1
    assert weights[1, 4, 4] == 0
    assert weights[1, 1, 1] == 0
    assert weights[0, 0, 0] == 1


def test_eddy_without_topup_uses_zero_susceptibility():
    field, derivative, voxel_sizes = _load_topup_field(None, (4, 5, 6), "cpu", 1)
    assert torch.count_nonzero(field) == 0
    assert torch.count_nonzero(derivative) == 0
    assert voxel_sizes is None


def test_packaged_ukb_fnirt_configs_have_official_hashes():
    expected = {
        "oxford_s1.cnf": "3df6320e97300f9cef8a77b6c7c6306b8256351d5a7022d9966c076f24747227",
        "oxford_s2.cnf": "b2fe418169f8ce1dde5e277a09e8edcdb6a0dc3f3065972ae69404020ef639ec",
        "oxford_s3.cnf": "9d304dfd224b8e48f20364dcce77c7618722cf3a9bcd40d764a6d25ee8020eaf",
    }
    root = files("fnit.dmri_pipeline").joinpath("assets")
    assert {
        name: hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest()
        for name in expected
    } == expected


def test_partial_pa_acquisition_is_rejected(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    image = nib.Nifti1Image(
        np.ones((4, 4, 4, 2), dtype=np.float32), np.eye(4)
    )
    nib.save(image, raw / "AP.nii.gz")
    nib.save(image.slicer[..., :1], raw / "PA.nii.gz")
    (raw / "AP.bval").write_text("0 1000")
    (raw / "AP.bvec").write_text(
        "0 1" + chr(10) + "0 0" + chr(10) + "0 0"
    )
    (raw / "AP.json").write_text(
        '{"PhaseEncodingDirection":"j","TotalReadoutTime":0.05}'
    )
    template = nib.Nifti1Image(
        np.ones((4, 4, 4), dtype=np.float32), np.eye(4)
    )
    fa = tmp_path / "fa.nii.gz"
    skeleton = tmp_path / "skeleton.nii.gz"
    nib.save(template, fa)
    nib.save(template, skeleton)

    with pytest.raises(FileNotFoundError, match="incomplete PA"):
        DMRIPipeline(device="cpu").run(
            raw,
            tmp_path / "out",
            fa_template=fa,
            fa_skeleton=skeleton,
        )


@pytest.mark.parametrize("bvec_source", ("rotated", "raw"))
def test_mmorf_branch_calls_public_mmorf_function(monkeypatch, tmp_path, bvec_source):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("AP.nii.gz", "AP.bval", "AP.bvec", "AP.json"):
        (raw / name).touch()

    shape = (5, 6, 7)
    scalar = nib.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4))
    tensor_data = np.zeros((*shape, 6), dtype=np.float32)
    tensor_data[..., (0, 3, 5)] = (1.4e-3, 0.5e-3, 0.4e-3)
    tensor = nib.Nifti1Image(tensor_data, np.eye(4))
    paths = {}
    for name, image in {
        "t1": scalar,
        "fa_template": scalar,
        "t1_template": scalar,
        "tensor_template": tensor,
        "mask": scalar,
    }.items():
        paths[name] = tmp_path / f"{name}.nii.gz"
        nib.save(image, paths[name])

    monkeypatch.setattr(
        pipeline_module,
        "_prepare_ap_only",
        lambda *args, **kwargs: {"mask": paths["mask"]},
    )

    class FakeEDDY:
        def __init__(self, device=None):
            pass

        def run(self, **kwargs):
            return SimpleNamespace(qc={})

    class FakeDTIFIT:
        def __init__(self, device=None):
            pass

        def run(self, *args, **kwargs):
            maps = {
                name: scalar
                for name in ("FA", "MD", "L1", "L2", "L3", "MO")
            }
            maps["tensor"] = tensor
            return SimpleNamespace(maps=maps, qc={})

    class FakeNODDI:
        def __init__(self, device=None):
            pass

        def run(self, *args, **kwargs):
            captured["noddi_bvecs"] = args[2]
            return SimpleNamespace(ndi=scalar, odi=scalar, fwf=scalar, qc={})

    class Saveable:
        def __init__(self, image):
            self.image = image

        def save(self, path):
            nib.save(self.image, path)

    class FakeSynthStrip:
        def __init__(self, weights=None, device=None):
            pass

        def __call__(self, image):
            return SimpleNamespace(image=Saveable(scalar), mask=Saveable(scalar))

    class FakeFLIRT:
        def __init__(self, device=None):
            pass

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(matrix=np.eye(4), qc={})

    monkeypatch.setattr(pipeline_module, "TorchEDDY", FakeEDDY)
    monkeypatch.setattr(pipeline_module, "TorchDTIFIT", FakeDTIFIT)
    monkeypatch.setattr(pipeline_module, "TorchAMICONODDI", FakeNODDI)
    monkeypatch.setattr(pipeline_module, "SynthStrip", FakeSynthStrip)
    monkeypatch.setattr(pipeline_module, "TorchFLIRT", FakeFLIRT)
    def fake_select_shell(*args, **kwargs):
        captured["shell_bvecs"] = args[2]
        return "shell.nii.gz", "shell.bval", "shell.bvec"

    monkeypatch.setattr(pipeline_module, "select_shell", fake_select_shell)

    captured = {}

    def fake_run_mmorf(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        warp = nib.Nifti1Image(
            np.zeros((*shape, 3), dtype=np.float32), np.eye(4)
        )
        return SimpleNamespace(warp=warp, qc={})

    monkeypatch.setattr(pipeline_module, "run_mmorf", fake_run_mmorf)
    monkeypatch.setattr(
        pipeline_module,
        "apply_mmorf_warp",
        lambda image, *args, **kwargs: image,
    )

    output_dir = tmp_path / "out"
    result = DMRIPipeline(
        device="cpu",
        registration_backend="mmorf",
        synthstrip_weights="synthstrip.pt",
        bvec_source=bvec_source,
    ).run(
        raw,
        output_dir,
        fa_template=paths["fa_template"],
        t1=paths["t1"],
        t1_template=paths["t1_template"],
        tensor_template=paths["tensor_template"],
    )

    assert captured["args"][2] is tensor
    assert captured["kwargs"]["device"].type == "cpu"
    assert captured["kwargs"]["output_dir"] == output_dir / "registration"
    np.testing.assert_array_equal(
        captured["kwargs"]["moving_tensor_affine"], np.eye(4)
    )
    assert set(result.standard_maps) == set(STANDARD_MAP_NAMES)
    expected_bvecs = (
        raw / "AP.bvec" if bvec_source == "raw"
        else output_dir / "eddy" / "data.eddy_rotated_bvecs"
    )
    assert captured["shell_bvecs"] == expected_bvecs
    assert captured["noddi_bvecs"] == expected_bvecs
    assert result.qc["bvec_source"] == bvec_source


def test_tbss_passes_volumes_to_fnirt(monkeypatch, tmp_path):
    shape = (9, 9, 9)
    data = np.zeros(shape, dtype=np.float32)
    data[2:7, 2:7, 2:7] = 0.5
    image = nib.Nifti1Image(data, np.eye(4))
    maps = {name: image for name in STANDARD_MAP_NAMES}
    maps["ICVF"] = nib.Nifti1Image(data[..., None], np.eye(4))
    skeleton = nib.Nifti1Image(np.full(shape, 10000, dtype=np.float32), np.eye(4))
    reference_path = tmp_path / "reference.nii.gz"
    skeleton_path = tmp_path / "skeleton.nii.gz"
    nib.save(image, reference_path)
    nib.save(skeleton, skeleton_path)
    captured = {}

    class FakeFLIRT:
        def __init__(self, device=None):
            pass

        def __call__(self, moving, fixed, **kwargs):
            return SimpleNamespace(
                moving_to_fixed_world=np.eye(4),
                matrix=np.eye(4),
                qc={},
            )

    class FakeFNIRT:
        def __init__(self, device=None, config=None):
            captured["fnirt_config"] = config

        def __call__(self, moving, fixed, initial):
            captured["moving"] = moving
            captured["fixed"] = fixed
            return SimpleNamespace(
                coefficient_image=nib.Nifti1Image(
                    np.zeros((*shape, 3), dtype=np.float64), np.eye(4)
                ),
                nonlinear_jacobian=nib.Nifti1Image(
                    np.ones(shape, dtype=np.float32), np.eye(4)
                ),
                qc={},
            )

    class FakeApplyWarp:
        def __init__(self, device=None):
            pass

        def __call__(self, source, reference, **kwargs):
            loaded = nib.load(str(source)) if isinstance(source, (str, bytes)) else source
            return SimpleNamespace(image=loaded)

    monkeypatch.setattr(tbss_module, "TorchFLIRT", FakeFLIRT)
    monkeypatch.setattr(tbss_module, "TorchFNIRT", FakeFNIRT)
    monkeypatch.setattr(tbss_module, "TorchApplyWarp", FakeApplyWarp)
    result = tbss_module.TorchTBSS(device="cpu")(maps, reference_path, skeleton_path)

    assert isinstance(captured["moving"], nib.Nifti1Image)
    assert isinstance(captured["fixed"], nib.Nifti1Image)
    np.testing.assert_array_equal(captured["moving"].affine, np.eye(4))
    assert set(result.standard_maps) == set(STANDARD_MAP_NAMES)
    assert result.standard_maps["ICVF"].shape == shape
    assert result.skeleton_maps["ICVF"].shape == shape
    assert result.qc["oxford_subsampling_fwhm_lambda_iteration_values_combined"]
    assert result.qc["official_oxford_three_process_execution"] is False
    assert result.qc["official_oxford_three_process_handoff"] is True
    assert result.qc["official_implicit_zero_masks_and_masked_smoothing"] is True
    assert result.qc["official_stage_2_3_scg"] is True
    assert captured["fnirt_config"].minimization_methods[-2:] == ("scg", "scg")
    assert captured["fnirt_config"].process_stages == (1, 1, 1, 1, 2, 3)
    assert result.qc["topology_projection_matches_fsl"] is False
    assert "official_oxford_three_stage_config" not in result.qc
    assert len(result.qc["known_differences"]) == 3
