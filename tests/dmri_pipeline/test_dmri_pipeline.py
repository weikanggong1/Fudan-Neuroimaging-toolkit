import argparse
import hashlib
import json
from types import SimpleNamespace
from importlib.resources import files
import nibabel as nib
import numpy as np
import pytest
import torch

import fnit.dmri_pipeline.pipeline as pipeline_module
import fnit.dmri_pipeline.tbss as tbss_module

from fnit.dmri_pipeline import DMRIPipeline, STANDARD_MAP_NAMES
from fnit.dmri_pipeline.bids import locate_bids_dwi, stage_bids_dwi
from fnit.dmri_pipeline.cli import _arguments
from fnit.dmri_pipeline.tbss import TBSSConfig, preprocess_fa
from fnit.fnirt import FNIRTConfig, TBSSFNIRTConfig
from fnit.eddy.topup_field import _load_topup_field


def test_common_nine_map_contract():
    assert STANDARD_MAP_NAMES == (
        "FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF"
    )


def test_bvec_source_cli_and_invalid_value():
    parser = argparse.ArgumentParser()
    _arguments(parser)
    options = parser.parse_args(
        ["--raw-dir", "raw", "-o", "out", "--fa-template", "fa.nii.gz",
         "--bvec-source", "raw", "--noddi-fit-method", "classic"]
    )
    assert options.bvec_source == "raw"
    assert options.noddi_fit_method == "classic"
    with pytest.raises(ValueError, match="bvec_source"):
        DMRIPipeline(device="cpu", bvec_source="invalid")
    with pytest.raises(ValueError, match="noddi_fit_method"):
        DMRIPipeline(device="cpu", noddi_fit_method="invalid")


@pytest.mark.parametrize("seed", [0, -1, 2**32, True, 1.5, "12345"])
def test_eddy_seed_rejects_automatic_zero_and_invalid_values(seed):
    with pytest.raises(ValueError, match="eddy_gp_seed"):
        DMRIPipeline(device="cpu", eddy_gp_seed=seed)


def test_eddy_seed_cli_and_default():
    parser = argparse.ArgumentParser()
    _arguments(parser)
    args = parser.parse_args([
        "--raw-dir", "raw", "-o", "out", "--fa-template", "fa.nii.gz",
        "--eddy-gp-seed", "12345",
    ])
    assert args.eddy_gp_seed == 12345
    assert DMRIPipeline(device="cpu").eddy_gp_seed is None
    assert DMRIPipeline(device="cpu", eddy_gp_seed=np.int64(12345)).eddy_gp_seed == 12345


def _bids_case(root, *, with_t1):
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Real-format fixture", "BIDSVersion": "1.11.0"})
    )
    base = root / "sub-01" / "ses-1"
    dwi_dir, fmap_dir = base / "dwi", base / "fmap"
    dwi_dir.mkdir(parents=True)
    fmap_dir.mkdir()
    image = dwi_dir / "sub-01_ses-1_dir-AP_run-01_dwi.nii.gz"
    reverse = fmap_dir / "sub-01_ses-1_dir-PA_epi.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((6, 6, 6, 3), np.float32), np.eye(4)), image)
    nib.save(nib.Nifti1Image(np.ones((6, 6, 6), np.float32), np.eye(4)), reverse)
    (base / "sub-01_ses-1_dwi.bval").write_text("0 1000 2000\n")
    (base / "sub-01_ses-1_dwi.bvec").write_text("0 1 0\n0 0 1\n0 0 0\n")
    (root / "dwi.json").write_text(json.dumps({
        "PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05,
        "B0FieldSource": "pepolar1",
    }))
    (fmap_dir / "sub-01_ses-1_dir-PA_epi.json").write_text(json.dumps({
        "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05,
        "B0FieldIdentifier": "pepolar1",
    }))
    if with_t1:
        anat = base / "anat"
        anat.mkdir()
        nib.save(nib.Nifti1Image(np.ones((6, 6, 6), np.float32), np.eye(4)),
                 anat / "sub-01_ses-1_T1w.nii.gz")
    return image, reverse


@pytest.mark.parametrize("with_t1,backend", [(False, "tbss"), (True, "mmorf")])
def test_bids_entry_stages_realistic_inputs_and_optional_t1(monkeypatch, tmp_path,
                                                               with_t1, backend):
    root = tmp_path / "bids"
    root.mkdir()
    image, reverse = _bids_case(root, with_t1=with_t1)
    captured = {}

    def fake_run(self, raw_dir, output_dir, **kwargs):
        captured.update(raw_dir=raw_dir, output_dir=output_dir, **kwargs)
        return SimpleNamespace(output_dir=output_dir)

    monkeypatch.setattr(DMRIPipeline, "run", fake_run)
    pipeline = DMRIPipeline(device="cpu", registration_backend=backend,
                            synthstrip_weights="weights.pt" if with_t1 else None)
    output = tmp_path / "output"
    pipeline.run_bids(root, output, subject="01", session="1", run="01",
                      direction="AP", fa_template="fa.nii.gz")
    staged = output / "bids_input"
    assert captured["raw_dir"] == staged
    assert (staged / "AP.nii.gz").resolve() == image
    assert nib.load(str(staged / "PA.nii.gz")).shape == (6, 6, 6, 1)
    np.testing.assert_array_equal(np.atleast_1d(np.loadtxt(staged / "PA.bval")), [0])
    assert json.loads((staged / "PA.json").read_text())["PhaseEncodingDirection"] == "j-"
    assert json.loads((staged / "bids_selection.json").read_text())["reverse_pe"] == reverse.relative_to(root).as_posix()
    assert (captured["t1"] is not None) == with_t1


def test_bids_mmorf_requires_t1_and_ambiguous_dwi_is_rejected(tmp_path):
    root = tmp_path / "bids"
    root.mkdir()
    image, _ = _bids_case(root, with_t1=False)
    second = image.with_name(image.name.replace("run-01", "run-02"))
    second.symlink_to(image)
    with pytest.raises(ValueError, match="expected one BIDS DWI run"):
        locate_bids_dwi(root, subject="01", session="1")
    with pytest.raises(ValueError, match="requires a T1w"):
        DMRIPipeline(device="cpu", registration_backend="mmorf").run_bids(
            root, tmp_path / "out", subject="01", run="01", direction="AP",
            fa_template="fa.nii.gz",
        )
    assert image.is_file()


def test_bids_reverse_dwi_preserves_its_bvalues(tmp_path):
    root = tmp_path / "bids"
    root.mkdir()
    _bids_case(root, with_t1=False)
    fmap = root / "sub-01" / "ses-1" / "fmap"
    for path in fmap.iterdir():
        path.unlink()
    reverse = root / "sub-01" / "ses-1" / "dwi" / "sub-01_ses-1_dir-PA_dwi.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((6, 6, 6, 2), np.float32), np.eye(4)), reverse)
    reverse.with_suffix("").with_suffix(".bval").write_text("0 1000\n")
    reverse.with_suffix("").with_suffix(".json").write_text(json.dumps({
        "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05,
        "B0FieldIdentifier": "pepolar1",
    }))
    inputs = locate_bids_dwi(root, subject="01", direction="AP")
    staged = stage_bids_dwi(inputs, tmp_path / "staged")
    np.testing.assert_array_equal(np.loadtxt(staged / "PA.bval"), [0, 1000])


def test_bids_single_sessionless_t1_is_not_counted_twice(tmp_path):
    root = tmp_path / "bids"
    dwi = root / "sub-01" / "dwi"
    anat = root / "sub-01" / "anat"
    dwi.mkdir(parents=True)
    anat.mkdir()
    (root / "dataset_description.json").write_text(
        '{"Name":"test","BIDSVersion":"1.11.0"}'
    )
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4, 2), np.float32), np.eye(4)),
             dwi / "sub-01_dir-AP_dwi.nii.gz")
    (dwi / "sub-01_dir-AP_dwi.bval").write_text("0 1000\n")
    (dwi / "sub-01_dir-AP_dwi.bvec").write_text("0 1\n0 0\n0 0\n")
    (dwi / "sub-01_dir-AP_dwi.json").write_text(
        '{"PhaseEncodingDirection":"j-","TotalReadoutTime":0.05}'
    )
    t1 = anat / "sub-01_T1w.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), t1)
    assert locate_bids_dwi(root, subject="01").t1w == t1


def test_bids_without_reverse_phase_encoding_stages_ap_only(tmp_path):
    root = tmp_path / "bids"
    root.mkdir()
    _bids_case(root, with_t1=False)
    staged = tmp_path / "staged"
    stage_bids_dwi(locate_bids_dwi(root, subject="01", direction="AP"), staged)
    assert (staged / "PA.nii.gz").is_file()
    for path in (root / "sub-01" / "ses-1" / "fmap").iterdir():
        path.unlink()
    inputs = locate_bids_dwi(root, subject="01", direction="AP")
    assert inputs.reverse is None
    staged = stage_bids_dwi(inputs, staged, overwrite=True)
    assert (staged / "AP.nii.gz").is_file()
    assert not (staged / "PA.nii.gz").exists()


def test_bids_subject_level_fieldmap_can_target_session_dwi(tmp_path):
    root = tmp_path / "bids"
    root.mkdir()
    _bids_case(root, with_t1=False)
    participant = root / "sub-01"
    source = participant / "ses-1" / "fmap"
    target = participant / "fmap"
    target.mkdir()
    for path in source.iterdir():
        path.rename(target / path.name.replace("_ses-1", ""))
    inputs = locate_bids_dwi(root, subject="01", session="1", direction="AP")
    assert inputs.reverse.parent == target


def test_bids_intended_for_selects_one_of_two_reverse_fieldmaps(tmp_path):
    root = tmp_path / "bids"
    root.mkdir()
    image, reverse = _bids_case(root, with_t1=False)
    fmap = reverse.parent
    other = fmap / "sub-01_ses-1_dir-PA_run-02_epi.nii.gz"
    other.symlink_to(reverse)
    other.with_suffix("").with_suffix(".json").write_text(json.dumps({
        "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05,
        "IntendedFor": "ses-1/dwi/unrelated_dwi.nii.gz",
    }))
    reverse.with_suffix("").with_suffix(".json").write_text(json.dumps({
        "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05,
        "IntendedFor": image.relative_to(root / "sub-01").as_posix(),
    }))
    assert locate_bids_dwi(root, subject="01", direction="AP").reverse == reverse


def test_bids_uncompressed_dwi_is_staged_as_nii_gz(tmp_path):
    root = tmp_path / "bids"
    root.mkdir()
    image, _ = _bids_case(root, with_t1=False)
    uncompressed = image.with_suffix("")
    nib.save(nib.load(str(image)), uncompressed)
    image.unlink()
    inputs = locate_bids_dwi(root, subject="01", direction="AP")
    staged = stage_bids_dwi(inputs, tmp_path / "staged")
    assert nib.load(str(staged / "AP.nii.gz")).shape == (6, 6, 6, 3)


def test_bids_cli_selects_single_subject_run():
    parser = argparse.ArgumentParser()
    _arguments(parser)
    args = parser.parse_args([
        "--bids-root", "bids", "--subject", "sub-01", "--session", "ses-1",
        "--direction", "AP", "--run", "01", "-o", "out", "--fa-template", "fa.nii.gz",
    ])
    assert (args.bids_root, args.subject, args.direction) == ("bids", "sub-01", "AP")


def test_bids_tbss_ignores_multiple_t1w(monkeypatch, tmp_path):
    root = tmp_path / "bids"
    root.mkdir()
    _bids_case(root, with_t1=True)
    anat = root / "sub-01" / "ses-1" / "anat"
    source = anat / "sub-01_ses-1_T1w.nii.gz"
    (anat / "sub-01_ses-1_run-02_T1w.nii.gz").symlink_to(source)
    captured = {}

    def fake_run(self, raw_dir, output_dir, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_dir=output_dir)

    monkeypatch.setattr(DMRIPipeline, "run", fake_run)
    DMRIPipeline(device="cpu").run_bids(
        root, tmp_path / "output", subject="01", direction="AP",
        fa_template="fa.nii.gz",
    )
    assert captured["t1"] is None


def test_tbss_config_combines_selected_official_schedule_values():
    config = TBSSConfig().fnirt
    assert config == TBSSFNIRTConfig()
    assert DMRIPipeline(device="cpu").fnirt_config == config
    assert DMRIPipeline(device="cpu", fnirt_config="default").fnirt_config == FNIRTConfig()
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
            captured["gp_seed"] = kwargs["gp_seed"]
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
        def __init__(self, device=None, fit_method="amico"):
            captured["noddi_fit_method"] = fit_method

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
        "prepare_mmorf_warp",
        lambda image, *args, **kwargs: SimpleNamespace(
            apply=lambda source, **options: source
        ),
    )

    output_dir = tmp_path / "out"
    result = DMRIPipeline(
        device="cpu",
        registration_backend="mmorf",
        synthstrip_weights="synthstrip.pt",
        bvec_source=bvec_source,
        noddi_fit_method="classic" if bvec_source == "raw" else "amico",
        eddy_gp_seed=12345,
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
    assert captured["gp_seed"] == result.qc["eddy_gp_seed"] == 12345
    assert captured["noddi_fit_method"] == (
        "classic" if bvec_source == "raw" else "amico"
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

        def prepare(self, source, reference, **kwargs):
            captured["sampling_preparations"] = captured.get("sampling_preparations", 0) + 1
            return self

        def apply(self, source, reference=None):
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
    assert captured["sampling_preparations"] == 1
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
