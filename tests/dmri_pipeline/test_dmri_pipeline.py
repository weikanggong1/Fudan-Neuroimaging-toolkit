import hashlib
from types import SimpleNamespace
from importlib.resources import files
import nibabel as nib
import numpy as np
import pytest
import torch
import surfa as sf

import fnit.dmri_pipeline.tbss as tbss_module

from fnit.dmri_pipeline import DMRIPipeline, STANDARD_MAP_NAMES
from fnit.dmri_pipeline.tbss import TBSSConfig, preprocess_fa
from fnit.eddy.core import _load_topup_field


def test_common_nine_map_contract():
    assert STANDARD_MAP_NAMES == (
        "FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF"
    )


def test_tbss_config_is_the_three_official_ukb_schedules():
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
            pass

        def __call__(self, moving, fixed, initial):
            captured["moving"] = moving
            captured["fixed"] = fixed
            geometry = sf.ImageGeometry(shape, vox2world=np.eye(4))
            return SimpleNamespace(
                coefficient_image=nib.Nifti1Image(
                    np.zeros((*shape, 3), dtype=np.float64), np.eye(4)
                ),
                nonlinear_jacobian=sf.Volume(
                    np.ones(shape, dtype=np.float32), geometry=geometry
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

    assert isinstance(captured["moving"], sf.Volume)
    assert isinstance(captured["fixed"], sf.Volume)
    assert set(result.standard_maps) == set(STANDARD_MAP_NAMES)
    assert result.standard_maps["ICVF"].shape == shape
    assert result.skeleton_maps["ICVF"].shape == shape
