from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fnit import cli as root_cli
from fnit._nib import new_image
from fnit.flirt.coordinates import flirt_to_world_affine
from fnit.fnirt import FNIRTConfig, GMFNIRTConfig, T1FNIRTConfig, TBSSFNIRTConfig, cli, standalone
from fnit.fnirt.io import make_fsl_coefficient_image
from fnit.fnirt.spline import fsl_control_shape


def _save(path, data, affine):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine), path)


def _fixture(tmp_path):
    moving_path = tmp_path / "moving.nii.gz"
    fixed_path = tmp_path / "fixed.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    matrix_path = tmp_path / "moving_to_fixed.mat"
    moving_affine = np.array(
        [[1.2, 0, 0, -5], [0, 1.3, 0, 4], [0, 0, 1.4, 2], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    fixed_affine = np.array(
        [[2, 0, 0, -8], [0, 2, 0, -10], [0, 0, 2, -6], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    moving_data = np.indices((5, 6, 7)).sum(0).astype(np.float32) + 1
    fixed_data = np.indices((4, 5, 6)).sum(0).astype(np.float32) + 2
    _save(moving_path, moving_data, moving_affine)
    _save(fixed_path, fixed_data, fixed_affine)
    _save(mask_path, np.ones(fixed_data.shape), fixed_affine)
    fsl_matrix = np.array(
        [[1.01, 0.01, 0, 2], [0, 0.99, -0.01, -1], [0, 0, 1.02, 3], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    np.savetxt(matrix_path, fsl_matrix)
    return moving_path, fixed_path, mask_path, matrix_path, fsl_matrix


def _fake_result(fixed, affine_forward=None):
    if affine_forward is None:
        affine_forward = np.eye(4)
    knot_spacing = (2, 2, 2)
    coefficient_shape = fsl_control_shape(fixed.shape[:3], knot_spacing)
    coefficient_image = make_fsl_coefficient_image(
        np.zeros((*coefficient_shape, 3), dtype=np.float32),
        fixed.shape[:3],
        nib.affines.voxel_sizes(fixed.affine),
        knot_spacing,
        affine_forward,
    )
    moved = new_image(np.full(fixed.shape[:3], 0.25, dtype=np.float32), fixed)
    jacobian = new_image(np.full(fixed.shape[:3], 1.1, dtype=np.float32), fixed)
    return SimpleNamespace(
        coefficient_image=coefficient_image,
        moved=moved,
        nonlinear_jacobian=jacobian,
        full_pull_jacobian=new_image(
            np.full(fixed.shape[:3], 9.9, dtype=np.float32), fixed
        ),
    )


def test_run_fnirt_converts_affine_and_writes_atomic_outputs(tmp_path, monkeypatch):
    moving_path, fixed_path, mask_path, matrix_path, fsl_matrix = _fixture(tmp_path)
    captured = {"calls": 0}

    class FakeFNIRT:
        def __init__(self, *, device, config, execution="optimized"):
            captured["device"] = device
            captured["config"] = config

        def __call__(self, moving, fixed, moving_to_fixed, *, reference_mask):
            captured["calls"] += 1
            captured["moving"] = moving
            captured["fixed"] = fixed
            captured["world_affine"] = np.asarray(moving_to_fixed.matrix)
            captured["mask"] = reference_mask
            return _fake_result(fixed, fsl_matrix)

    monkeypatch.setattr(standalone, "TorchFNIRT", FakeFNIRT)
    cout = tmp_path / "nested" / "warp.nii.gz"
    iout = tmp_path / "nested" / "moved.nii.gz"
    jout = tmp_path / "nested" / "jacobian.nii.gz"
    result = standalone.run_fnirt(
        moving_path,
        fixed_path,
        matrix_path,
        cout=cout,
        iout=iout,
        jout=jout,
        refmask=mask_path,
        device="cuda:1",
    )

    moving = captured["moving"]
    fixed = captured["fixed"]
    expected = flirt_to_world_affine(
        fsl_matrix,
        moving.affine,
        fixed.affine,
        moving.shape[:3],
        fixed.shape[:3],
        nib.affines.voxel_sizes(moving.affine),
        nib.affines.voxel_sizes(fixed.affine),
    )
    assert np.allclose(captured["world_affine"], expected)
    assert captured["world_affine"].shape == (4, 4)
    assert captured["device"] == "cuda:1"
    assert captured["config"] == FNIRTConfig()
    assert captured["mask"].shape[:3] == fixed.shape[:3]
    assert captured["calls"] == 1
    coefficient = nib.load(cout)
    fixed_image = nib.load(fixed_path)
    moved_image = nib.load(iout)
    jacobian_image = nib.load(jout)
    assert int(coefficient.header["intent_code"]) == 2007
    np.testing.assert_allclose(coefficient.get_sform(), fsl_matrix)
    np.testing.assert_allclose(coefficient.header["pixdim"][1:4], (2, 2, 2))
    np.testing.assert_allclose(
        [
            coefficient.header["qoffset_x"],
            coefficient.header["qoffset_y"],
            coefficient.header["qoffset_z"],
        ],
        fixed.shape[:3],
    )
    np.testing.assert_allclose(
        [
            coefficient.header["intent_p1"],
            coefficient.header["intent_p2"],
            coefficient.header["intent_p3"],
        ],
        nib.affines.voxel_sizes(fixed.affine),
    )
    assert moved_image.shape == fixed_image.shape
    assert jacobian_image.shape == fixed_image.shape
    np.testing.assert_allclose(moved_image.affine, fixed_image.affine)
    np.testing.assert_allclose(jacobian_image.affine, fixed_image.affine)
    assert np.allclose(np.asarray(moved_image.dataobj), 0.25)
    assert np.allclose(np.asarray(jacobian_image.dataobj), 1.1)
    assert not list((tmp_path / "nested").glob(".*.tmp-*"))
    assert result.moved.shape[:3] == fixed.shape[:3]

    with pytest.raises(FileExistsError, match="output exists"):
        standalone.run_fnirt(
            moving_path,
            fixed_path,
            matrix_path,
            cout=cout,
            refmask=mask_path,
        )
    assert captured["calls"] == 1

    standalone.run_fnirt(
        moving_path,
        fixed_path,
        matrix_path,
        cout=cout,
        refmask=mask_path,
        overwrite=True,
    )
    assert captured["calls"] == 2


def test_run_fnirt_rejects_unsupported_contracts(tmp_path):
    moving_path, fixed_path, mask_path, matrix_path, _ = _fixture(tmp_path)
    with pytest.raises(NotImplementedError, match="only GM_2_MNI152GM_2mm"):
        standalone.run_fnirt(
            moving_path,
            fixed_path,
            matrix_path,
            cout=tmp_path / "warp.nii.gz",
            refmask=mask_path,
            config="unsupported.cnf",
        )
    modified = tmp_path / standalone.SUPPORTED_CONFIG
    modified.write_text("--miter=1,1,1,1\n")
    with pytest.raises(NotImplementedError, match="not the unmodified FSL"):
        standalone.run_fnirt(
            moving_path,
            fixed_path,
            matrix_path,
            cout=tmp_path / "warp.nii.gz",
            refmask=mask_path,
            config=modified,
        )
    with pytest.raises(ValueError, match="must not replace"):
        standalone.run_fnirt(
            moving_path,
            fixed_path,
            matrix_path,
            iout=moving_path,
            refmask=mask_path,
            overwrite=True,
        )


def test_run_fnirt_selects_t1_config(tmp_path, monkeypatch):
    moving, fixed, mask, affine, matrix = _fixture(tmp_path)
    captured = {}

    class FakeFNIRT:
        def __init__(self, *, device, config, execution="optimized"):
            captured["config"] = config

        def __call__(self, source, target, initial, *, reference_mask):
            assert reference_mask.shape == target.shape
            return _fake_result(target, matrix)

    monkeypatch.setattr(standalone, "TorchFNIRT", FakeFNIRT)
    standalone.run_fnirt(
        input=moving,
        reference=fixed,
        affine=affine,
        cout=tmp_path / "t1_coeff.nii.gz",
        refmask=mask,
        config=standalone.T1_CONFIG,
        device="cpu",
    )
    assert isinstance(captured["config"], T1FNIRTConfig)
    assert (tmp_path / "t1_coeff.nii.gz").is_file()


def test_cli_forwards_t1_config(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "run_fnirt", lambda *args, **kwargs: captured.update(kwargs))
    assert cli.main([
        "--in", "T1.nii.gz", "--ref", "MNI.nii.gz",
        "--refmask", "MNI_mask.nii.gz", "--config", standalone.T1_CONFIG,
    ]) == 0
    assert captured["config"] == T1FNIRTConfig()


def test_run_fnirt_uses_fsl_identity_default_cout_and_auto_device(
    tmp_path, monkeypatch
):
    moving_path, fixed_path, mask_path, _, _ = _fixture(tmp_path)
    captured = {}

    class FakeFNIRT:
        def __init__(self, *, device, config, execution="optimized"):
            captured["device"] = device

        def __call__(self, moving, fixed, moving_to_fixed, *, reference_mask):
            captured["world_affine"] = np.asarray(moving_to_fixed.matrix)
            return _fake_result(fixed, np.eye(4))

    monkeypatch.setattr(standalone, "TorchFNIRT", FakeFNIRT)
    monkeypatch.setattr(standalone.torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("FSLOUTPUTTYPE", "NIFTI_GZ")
    standalone.run_fnirt(moving_path, fixed_path, refmask=mask_path)

    moving = nib.load(moving_path)
    fixed = nib.load(fixed_path)
    expected = flirt_to_world_affine(
        np.eye(4),
        moving.affine,
        fixed.affine,
        moving.shape[:3],
        fixed.shape[:3],
        nib.affines.voxel_sizes(moving.affine),
        nib.affines.voxel_sizes(fixed.affine),
    )
    np.testing.assert_allclose(captured["world_affine"], expected)
    assert captured["device"] == "cuda"
    default_cout = tmp_path / "moving_warpcoef.nii.gz"
    assert default_cout.is_file()
    np.testing.assert_allclose(nib.load(default_cout).get_sform(), np.eye(4))


@pytest.mark.parametrize(
    ("output_type", "extension"),
    (("NIFTI", ".nii"), ("NIFTI_GZ", ".nii.gz")),
)
def test_run_fnirt_resolves_extensionless_outputs(
    tmp_path, monkeypatch, output_type, extension
):
    moving_path, fixed_path, mask_path, matrix_path, fsl_matrix = _fixture(tmp_path)

    class FakeFNIRT:
        def __init__(self, *, device, config, execution="optimized"):
            pass

        def __call__(self, moving, fixed, moving_to_fixed, *, reference_mask):
            return _fake_result(fixed, fsl_matrix)

    monkeypatch.setattr(standalone, "TorchFNIRT", FakeFNIRT)
    monkeypatch.setenv("FSLOUTPUTTYPE", output_type)
    standalone.run_fnirt(
        moving_path,
        fixed_path,
        matrix_path,
        cout=tmp_path / "warp",
        refmask=mask_path,
        device="cpu",
    )
    assert (tmp_path / f"warp{extension}").is_file()


def test_run_fnirt_rejects_unsupported_fsloutputtype(tmp_path, monkeypatch):
    moving_path, fixed_path, mask_path, matrix_path, _ = _fixture(tmp_path)
    monkeypatch.setenv("FSLOUTPUTTYPE", "ANALYZE_GZ")
    with pytest.raises(NotImplementedError, match="FSLOUTPUTTYPE=NIFTI"):
        standalone.run_fnirt(
            moving_path,
            fixed_path,
            matrix_path,
            cout=tmp_path / "warp",
            refmask=mask_path,
        )


def test_run_fnirt_does_not_resolve_reference_mask_from_fsldir(tmp_path, monkeypatch):
    moving_path, fixed_path, _, matrix_path, _ = _fixture(tmp_path)
    fixed_image = nib.load(fixed_path)
    standard_mask = (
        tmp_path / "fsl" / "data" / "standard" / standalone.DEFAULT_REFERENCE_MASK
    )
    _save(standard_mask, np.ones(fixed_image.shape), fixed_image.affine)
    monkeypatch.setenv("FSLDIR", str(tmp_path / "fsl"))

    class MustNotRun:
        def __init__(self, **kwargs):
            raise AssertionError("model ran without an explicit reference mask")

    monkeypatch.setattr(standalone, "TorchFNIRT", MustNotRun)
    with pytest.raises(ValueError, match="refmask is required"):
        standalone.run_fnirt(
            moving_path,
            fixed_path,
            matrix_path,
            cout=tmp_path / "warp.nii.gz",
            config="gm",
        )


def test_run_fnirt_requires_a_binary_reference_mask(tmp_path, monkeypatch):
    moving_path, fixed_path, _, matrix_path, _ = _fixture(tmp_path)
    fixed_image = nib.load(fixed_path)
    mask_path = tmp_path / "nonbinary_mask.nii.gz"
    values = np.ones(fixed_image.shape, dtype=np.float32)
    values[0, 0, 0] = 0.5
    _save(mask_path, values, fixed_image.affine)
    called = {"value": False}

    class MustNotRun:
        def __init__(self, **kwargs):
            called["value"] = True

    monkeypatch.setattr(standalone, "TorchFNIRT", MustNotRun)
    with pytest.raises(ValueError, match="binary"):
        standalone.run_fnirt(
            moving_path,
            fixed_path,
            matrix_path,
            cout=tmp_path / "warp.nii.gz",
            refmask=mask_path,
        )
    assert not called["value"]


def test_no_clobber_conflict_rolls_back_outputs_created_by_this_run(
    tmp_path, monkeypatch
):
    moving_path, fixed_path, mask_path, matrix_path, fsl_matrix = _fixture(tmp_path)
    cout = tmp_path / "warp.nii.gz"
    iout = tmp_path / "moved.nii.gz"
    competitor = b"created by another process"

    class FakeFNIRT:
        def __init__(self, *, device, config, execution="optimized"):
            pass

        def __call__(self, moving, fixed, moving_to_fixed, *, reference_mask):
            iout.write_bytes(competitor)
            return _fake_result(fixed, fsl_matrix)

    monkeypatch.setattr(standalone, "TorchFNIRT", FakeFNIRT)
    with pytest.raises(FileExistsError, match="output exists"):
        standalone.run_fnirt(
            moving_path,
            fixed_path,
            matrix_path,
            cout=cout,
            iout=iout,
            refmask=mask_path,
            device="cpu",
        )
    assert not cout.exists()
    assert iout.read_bytes() == competitor
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_in_memory_input_requires_explicit_cout(tmp_path):
    moving_path, fixed_path, mask_path, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="cout is required"):
        standalone.run_fnirt(
            nib.load(moving_path),
            nib.load(fixed_path),
            refmask=mask_path,
        )


def test_cli_forwards_fsl_roles_and_rejects_unknown_options(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_fnirt", fake_run)
    arguments = [
        "--in", "moving.nii.gz",
        "--ref", "fixed.nii.gz",
        "--aff", "moving_to_fixed.mat",
        "--cout", "warp.nii.gz",
        "--iout", "moved.nii.gz",
        "--jout", "jacobian.nii.gz",
        "--refmask", "mask.nii.gz",
        "--config", standalone.SUPPORTED_CONFIG,
        "--device", "cuda:0",
        "--execution", "reference",
        "--overwrite",
    ]
    assert cli.main(arguments) == 0
    assert captured["args"] == (
        "moving.nii.gz",
        "fixed.nii.gz",
        "moving_to_fixed.mat",
    )
    assert captured["kwargs"] == {
        "cout": "warp.nii.gz",
        "iout": "moved.nii.gz",
        "jout": "jacobian.nii.gz",
        "refmask": "mask.nii.gz",
        "config": GMFNIRTConfig(),
        "device": "cuda:0",
        "execution": "reference",
        "overwrite": True,
    }
    with pytest.raises(SystemExit) as error:
        cli.main(arguments + ["--fout", "dense.nii.gz"])
    assert error.value.code == 2


def test_cli_allows_fsl_default_affine_cout_and_device(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_fnirt", fake_run)
    assert cli.main(
        [
            "--in", "moving.nii.gz",
            "--ref", "fixed.nii.gz",
            "--refmask", "mask.nii.gz",
        ]
    ) == 0
    assert captured["args"] == ("moving.nii.gz", "fixed.nii.gz", None)
    assert captured["kwargs"] == {
        "cout": None,
        "iout": None,
        "jout": None,
        "refmask": "mask.nii.gz",
        "config": FNIRTConfig(),
        "device": None,
        "execution": "optimized",
        "overwrite": False,
    }


def test_root_cli_dispatches_to_the_same_fnirt_wrapper(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(standalone, "run_fnirt", fake_run)
    root_cli.main(
        [
            "fnirt",
            "--in", "moving.nii.gz",
            "--ref", "fixed.nii.gz",
            "--aff", "moving_to_fixed.mat",
            "--cout", "warp.nii.gz",
            "--iout", "moved.nii.gz",
            "--jout", "jacobian.nii.gz",
            "--refmask", "mask.nii.gz",
            "--device", "cuda:1",
        ]
    )
    assert captured["args"] == (
        "moving.nii.gz", "fixed.nii.gz", "moving_to_fixed.mat"
    )
    assert captured["kwargs"] == {
        "cout": "warp.nii.gz",
        "iout": "moved.nii.gz",
        "jout": "jacobian.nii.gz",
        "refmask": "mask.nii.gz",
        "config": FNIRTConfig(),
        "device": "cuda:1",
        "overwrite": False,
    }
