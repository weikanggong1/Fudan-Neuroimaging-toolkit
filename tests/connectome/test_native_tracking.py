"""Native invocation, exact image geometry and unknown-seed checkpoint controls.

Executable behavior is stubbed here. Real MRI end-to-end validation is recorded
separately; these fixtures test the production adapter and serialization rules.
"""

from pathlib import Path
import os
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.connectome import tracking
from fnit.connectome import native_runtime
from fnit.connectome import pipeline
from fnit.connectome.checkpoints import StreamlinePoints
from fnit.connectome.pipeline import _pack_core, _restore_core


@pytest.fixture
def native_stub(tmp_path, monkeypatch):
    binary = tmp_path / "native-cache/bin/tckgen"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"verified native runtime test control")
    calls = []
    manifest = dict(source_commit="fixed-reference", binary_sha256="fixed-binary",
                    source_archive_sha256="fixed-source", runtime_files={"bin/tckgen": "fixed"})
    monkeypatch.setattr(native_runtime, "ensure_tckgen", lambda: binary)
    monkeypatch.setattr(native_runtime, "native_runtime_manifest", lambda path: dict(manifest))
    paths = [np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]], np.float32),
             np.array([[0., 1., 1.], [2., 1., 1.]], np.float32)]

    def run(command, **options):
        calls.append((command, options))
        tck = nib.streamlines.TckFile(
            nib.streamlines.Tractogram(paths, affine_to_rasmm=np.eye(4)),
            header={"step_size": "0.5", "total_count": "11", "min_dist": "2", "max_angle": "45"})
        nib.streamlines.save(tck, command[2])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tracking.subprocess, "run", run)
    return binary, calls, paths


def inputs():
    return (torch.ones((4, 3, 2, 45)), torch.eye(4, dtype=torch.float64),
            torch.ones((6, 5, 4, 5)), torch.diag(torch.tensor([.75, .75, .75, 1.], dtype=torch.float64)),
            torch.ones((6, 5, 4)))


def test_nifti2_serialization_preserves_values_and_double_sform(tmp_path):
    data = torch.linspace(-1, 1, 90, dtype=torch.float32).reshape(2, 3, 3, 5)
    affine = np.eye(4)
    affine[:3, :3] = [[1.2345678912345, .003456789, 0], [0, .9876543212345, 0], [0, 0, 2.1234567890123]]
    affine[:3, 3] = [100.1234567890123, -17.1234567890123, 8.9876543219876]
    first = tracking.write_tracking_image(data, affine, tmp_path / "one.nii", voxel_spacing_mm=[1.2, 1., 2.])
    second = tracking.write_tracking_image(data, affine, tmp_path / "two.nii", voxel_spacing_mm=[1.2, 1., 2.])
    image = nib.load(first)
    assert isinstance(image, nib.Nifti2Image)
    np.testing.assert_array_equal(image.affine[:3].view(np.uint64), affine[:3].view(np.uint64))
    np.testing.assert_array_equal(image.get_fdata(dtype=np.float32).view(np.uint32), data.numpy().view(np.uint32))
    assert image.header.get_zooms()[:3] == (1.2, 1., 2.)
    assert image.header.get_xyzt_units()[0] == "mm"
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("bad", [np.zeros((4, 4)), np.full((4, 4), np.nan), np.eye(3),
                                  np.diag([1., 1., 1., 2.])])
def test_image_invalid_affines_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="affine"):
        tracking.write_tracking_image(torch.ones((2, 2, 2)), bad, tmp_path / "bad.nii")


@pytest.mark.parametrize("spacing", [[0., 1., 1.], [1., 1.], [1., 1., float("nan")]])
def test_image_invalid_spacing_rejected(tmp_path, spacing):
    with pytest.raises(ValueError, match="spacing"):
        tracking.write_tracking_image(torch.ones((2, 2, 2)), np.eye(4), tmp_path / "bad.nii",
                                      voxel_spacing_mm=spacing)


def test_native_command_seed_order_unknown_coordinates_and_packed_paths(native_stub, monkeypatch):
    binary, calls, expected = native_stub
    monkeypatch.setenv("MRTRIX_CONFIGFILE", "unrelated-user-config")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/test/conda/lib")
    result = tracking.probabilistic_tractography(*inputs(), n_seeds=17, seed=42, tracking_threads=3)
    command, options = calls[0]
    assert command[0] == str(binary.resolve())
    for name, value in (("-algorithm", "iFOD2"), ("-seeds", "17"), ("-select", "0"),
                        ("-maxlength", "250.0"), ("-samples", "3"), ("-power", "0.5"),
                        ("-cutoff", "0.1"), ("-nthreads", "3"), ("-angle", "45.0")):
        assert command[command.index(name) + 1] == value
    assert "-force" not in command and "-step" not in command and "-minlength" not in command
    environment = options["env"]
    assert environment["MRTRIX_RNG_SEED"] == "42"
    assert environment["MRTRIX_NTHREADS"] == "3"
    assert environment["MRTRIX_CONFIGFILE"] == "/dev/null"
    assert environment["FNIT_MRTRIX_ISOLATED_CONFIG"] == "1"
    assert environment["LD_LIBRARY_PATH"] == str(binary.parent.parent / "lib") + ":" + str(Path(sys.prefix) / "lib") + ":/test/conda/lib"
    assert result.accepted_seeds is None and result.seeds_attempted == 17
    assert result.mean_fa is None and result.lengths_mm.tolist() == [2., 2.]
    for actual, wanted in zip(result.paths, expected, strict=True):
        np.testing.assert_array_equal(actual.numpy().view(np.uint32), wanted.view(np.uint32))
    assert result.paths[0].untyped_storage().data_ptr() == result.paths[1].untyped_storage().data_ptr()
    assert result.native_provenance["tck_header"]["total_count"] == "11"
    assert result.native_provenance["runtime"]["binary_sha256"] == "fixed-binary"
    assert result.native_provenance["input_images"]["wm_fod"]["shape"] == [4, 3, 2, 45]
    assert result.native_provenance["input_images"]["five_tissue"]["shape"] == [6, 5, 4, 5]
    assert not Path(command[1]).exists(), "temporary serialized inputs should be released"


def test_config_and_loader_isolation_preserves_parent_environment(native_stub, monkeypatch):
    monkeypatch.setenv("MRTRIX_CONFIGFILE", "/unrelated/user/config")
    monkeypatch.setenv("LD_PRELOAD", "/unrelated/library.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(Path(sys.prefix) / "lib") + ":/extra/lib")
    before = dict(os.environ)
    tracking.probabilistic_tractography(*inputs(), n_seeds=17)
    child = native_stub[1][0][1]["env"]
    assert child["FNIT_MRTRIX_ISOLATED_CONFIG"] == "1"
    assert child.get("HOME") == before.get("HOME")
    assert child["MRTRIX_CONFIGFILE"] == "/dev/null"
    assert "LD_PRELOAD" not in child
    assert child["LD_LIBRARY_PATH"].split(":") == [str(native_stub[0].parent.parent / "lib"),
                                               str(Path(sys.prefix) / "lib"), "/extra/lib"]
    assert dict(os.environ) == before


def test_checkpoint_fingerprint_binds_deployment_patch_and_builder(native_stub, monkeypatch):
    manifest = dict(source_commit="fixed-reference", binary_sha256="fixed-binary",
                    runtime_files={"bin/tckgen": {"sha256": "fixed-binary"}},
                    builder_sha256="builder-v1", source_file_sha256={"core/file/config.cpp": "original"},
                    configuration_isolation={"patched_sha256": "patch-v1"})
    monkeypatch.setattr(native_runtime, "native_runtime_manifest", lambda path: dict(manifest))
    first = pipeline._native_tracking_fingerprint()
    manifest["configuration_isolation"] = {"patched_sha256": "patch-v2"}
    changed_patch = pipeline._native_tracking_fingerprint()
    assert changed_patch != first
    assert changed_patch["binary_sha256"] == first["binary_sha256"]
    manifest["builder_sha256"] = "builder-v2"
    changed_builder = pipeline._native_tracking_fingerprint()
    assert changed_builder != changed_patch
    assert changed_builder["source_file_sha256"] == {"core/file/config.cpp": "original"}


def test_direct_files_are_unchanged_and_not_reserialized(native_stub, tmp_path):
    paths = []
    for name, data, affine in (("fod", inputs()[0], inputs()[1]),
                               ("five", inputs()[2], inputs()[3]),
                               ("gmwmi", inputs()[4], inputs()[3])):
        path = tracking.write_tracking_image(data, affine, tmp_path / (name + ".nii"))
        paths.append(path)
    original = [path.read_bytes() for path in paths]
    result = tracking.probabilistic_tractography(paths[0], five_tissue=paths[1], gmwmi=paths[2], n_seeds=17)
    assert [path.read_bytes() for path in paths] == original
    assert all(not value["serialized_nifti2"] for value in result.native_provenance["input_images"].values())
    assert native_stub[1][0][0][1] == str(paths[0].resolve())


def test_provided_file_geometry_mismatch_rejected(native_stub, tmp_path):
    values = inputs()
    path = tracking.write_tracking_image(values[0], values[1], tmp_path / "fod.nii")
    shifted = values[1].clone(); shifted[0, 3] = .01
    with pytest.raises(ValueError, match="file input affine"):
        tracking.probabilistic_tractography(path, shifted, *values[2:], n_seeds=17)
    assert not native_stub[1]


def test_custom_step_minimum_and_native_tck_retention(native_stub, tmp_path):
    output = tmp_path / "actual.tck"
    result = tracking.probabilistic_tractography(*inputs(), n_seeds=17, step_mm=.25,
                                               min_length_mm=3., output_tck=output)
    command = native_stub[1][0][0]
    assert command[command.index("-step") + 1] == ".25" or command[command.index("-step") + 1] == "0.25"
    assert command[command.index("-minlength") + 1] == "3.0"
    assert output.is_file() and tracking._sha256(output) == result.native_provenance["tck_sha256"]
    previous = output.read_bytes()
    with pytest.raises(ValueError, match="new .tck"):
        tracking.probabilistic_tractography(*inputs(), n_seeds=17, output_tck=output)
    assert output.read_bytes() == previous


def test_optional_fa_uses_precise_sampler(native_stub, monkeypatch):
    import fnit.connectome.tcksample_precise as precise
    calls = []

    def sampler(paths, fa, affine):
        calls.append((paths, fa, affine))
        return torch.tensor([.35, .65])

    monkeypatch.setattr(precise, "sample_streamline_mean_precise", sampler)
    fa = torch.full((4, 3, 2), .5)
    result = tracking.probabilistic_tractography(*inputs(), n_seeds=17, fa=fa)
    assert result.mean_fa.tolist() == pytest.approx([.35, .65])
    assert calls[0][0] is result.paths and calls[0][1] is fa


@pytest.mark.parametrize("options", [dict(compile_arc=True), dict(batch_size=8192), dict(arc_proposals=16),
                                      dict(n_seeds=True), dict(seed=-1), dict(tracking_threads=0),
                                      dict(lmax=3), dict(step_mm=0.), dict(cutoff=float("nan")),
                                      dict(min_length_mm=300.)])
def test_retired_or_invalid_parameters_fail_before_native_execution(native_stub, options):
    arguments = dict(n_seeds=17, **options) if "n_seeds" not in options else options
    with pytest.raises(ValueError):
        tracking.probabilistic_tractography(*inputs(), **arguments)
    assert not native_stub[1]


def test_native_failure_is_not_silently_replaced(native_stub, monkeypatch):
    monkeypatch.setattr(tracking.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=7, stderr="real native failure", stdout=""))
    with pytest.raises(RuntimeError, match="exit code 7: real native failure"):
        tracking.probabilistic_tractography(*inputs(), n_seeds=17)


def test_zero_accepted_tracks_retains_requested_budget_without_fabrication(native_stub, monkeypatch):
    def empty(command, **kwargs):
        nib.streamlines.save(nib.streamlines.Tractogram([], affine_to_rasmm=np.eye(4)), command[2])
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(tracking.subprocess, "run", empty)
    result = tracking.probabilistic_tractography(*inputs(), n_seeds=17)
    assert result.paths == () and result.endpoints.shape == (0, 2, 3)
    assert result.lengths_mm.shape == (0,) and result.seeds_attempted == 17
    assert result.accepted_seeds is None


def test_nibabel_images_preserve_anatomy_header_spacing(native_stub):
    values = inputs()
    five = nib.Nifti2Image(values[2].numpy(), values[3].numpy())
    five.header.set_zooms((.8, .8, .8, 1.))
    result = tracking.probabilistic_tractography(values[0], values[1], five, gmwmi=values[4], n_seeds=17)
    images = result.native_provenance["input_images"]
    assert images["five_tissue"]["voxel_spacing_mm"] == [.8, .8, .8]
    assert images["gmwmi"]["voxel_spacing_mm"] == [.8, .8, .8]


def checkpoint_core():
    shape = (4, 3, 2)
    eye = torch.eye(4, dtype=torch.float64)
    paths = (torch.tensor([[0., 0., 0.], [1., 0., 0.]]),)
    track = tracking.Tractogram(paths, paths[0][None], torch.tensor([1.]), torch.tensor([.5]),
                                17, None, {"runtime": {"binary_sha256": "test-control"}})
    return dict(seg=torch.ones(shape), seg_affine=eye, five=torch.ones((*shape, 5)),
                gmwmi=torch.ones(shape), transform=eye, five_affine=eye,
                wm_sh=torch.ones((*shape, 45)), fa=torch.ones(shape),
                mask=torch.ones(shape, dtype=torch.bool), tracks=track,
                weights=torch.ones(1, dtype=torch.float64), dwi_affine=eye, dwi_shape=shape)


def restored_arrays(arrays):
    return {name: (torch.cat(value.paths) if isinstance(value, StreamlinePoints) else
                   value if isinstance(value, torch.Tensor) else torch.as_tensor(value))
            for name, value in arrays.items()}


def test_checkpoint_preserves_unknown_native_seed_and_provenance():
    core = checkpoint_core()
    arrays, metadata = _pack_core(core)
    assert "track_seeds" not in arrays and metadata["accepted_seed_coordinates"] == "unknown"
    result = _restore_core(restored_arrays(arrays), metadata, torch.device("cpu"))
    assert result["tracks"].accepted_seeds is None
    assert result["tracks"].native_provenance == core["tracks"].native_provenance
    assert torch.equal(result["tracks"].paths[0], core["tracks"].paths[0])


def test_checkpoint_can_preserve_declared_seed_coordinates_from_public_objects():
    core = checkpoint_core()
    core["tracks"].accepted_seeds = core["tracks"].paths[0][:1]
    arrays, metadata = _pack_core(core)
    assert metadata["accepted_seed_coordinates"] == "stored"
    result = _restore_core(restored_arrays(arrays), metadata, torch.device("cpu"))
    assert torch.equal(result["tracks"].accepted_seeds, core["tracks"].accepted_seeds)


def test_old_checkpoint_schema_is_never_reused_as_native():
    arrays, metadata = _pack_core(checkpoint_core())
    metadata["format_revision"] = 1
    with pytest.raises(ValueError, match="incomplete"):
        _restore_core(restored_arrays(arrays), metadata, torch.device("cpu"))
