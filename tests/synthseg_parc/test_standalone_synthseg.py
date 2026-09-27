"""The public 33-class SynthSeg Python and CLI interfaces share one path."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fnit import SynthSeg
from fnit import cli
from fnit import synthseg_parc
from fnit.synthseg_parc import synthseg


def _models(tmp_path):
    labels = np.concatenate((np.arange(33), np.arange(22)))
    names = np.array([f"label-{label}" for label in labels])
    for name, values in (
        ("synthseg_segmentation_labels_2.0.npy", labels),
        ("synthseg_segmentation_names_2.0.npy", names),
        ("synthseg_topological_classes_2.0.npy", np.zeros(55, dtype=int)),
    ):
        np.save(tmp_path / name, values)
    model = tmp_path / "synthseg_2.0.h5"
    model.write_bytes(b"test checkpoint")
    return model


def test_independent_api_uses_one_weight_directory_and_recon_soft_volumes(tmp_path, monkeypatch):
    model = _models(tmp_path)
    posterior = torch.zeros((33, 2, 2, 2), dtype=torch.float32)
    posterior[1] = 1
    posterior[1, 0, 0, 0] = 0.5
    posterior[2, 0, 0, 0] = 0.5000005
    observed = []

    class FakeSegmenter:
        def __init__(self, weights, labels, device):
            observed.append((Path(weights), Path(labels), str(device)))
            self.labels = torch.arange(33)

        def posterior(self, image):
            return posterior

    prepared = SimpleNamespace(
        image=torch.zeros((2, 2, 2)), aligned_affine=np.diag([2., 1., 1., 1.]),
        content_slices=(slice(0, 2),) * 3,
        volume_affine=np.diag([2., 1., 1., 1.]), voxel_volume_mm3=2.0,
    )
    monkeypatch.setattr(synthseg, "SynthSegSegmenter", FakeSegmenter)
    monkeypatch.setattr(synthseg, "preprocess_t1", lambda image, device: prepared)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    result = SynthSeg(weights=model, device="cpu")("t1.nii.gz")
    assert observed == [(model, tmp_path / "synthseg_segmentation_labels_2.0.npy", "cpu")]
    assert int(result.segmentation.data[0, 0, 0]) == 1
    assert result.segmentation.data.dtype == np.float32
    assert result.near_tie_voxels == 1
    assert result.volumes_mm3[1] == 15.0
    assert result.volumes_mm3[2] == 1.0
    assert result.total_intracranial_mm3 == 16.0
    csv_path = tmp_path / "volumes.csv"
    result.write_volumes_csv("case.nii.gz", csv_path)
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("subject,total intracranial,label-1,label-2")
    assert lines[1].startswith("case,16.0,15.0,1.0")
    result.total_intracranial_mm3 = 1280294.25
    result.write_volumes_csv("case.nii.gz", csv_path)
    assert csv_path.read_text().splitlines()[1].startswith("case,1280294.2,")


def test_public_cli_calls_shared_synthseg_api(tmp_path, monkeypatch):
    image = tmp_path / "t1.nii.gz"
    image.write_bytes(b"input")
    saved = []
    calls = []

    class FakeResult:
        near_tie_voxels = 0

        def __init__(self):
            self.segmentation = self

        def save(self, path):
            saved.append(Path(path))

        def write_volumes_csv(self, source, path):
            calls.append(("csv", Path(source), Path(path)))

    class FakeSynthSeg:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def __call__(self, image, **kwargs):
            calls.append(("run", Path(image), kwargs))
            return FakeResult()

    monkeypatch.setattr(synthseg_parc, "SynthSeg", FakeSynthSeg)
    cli.main(["synthseg", "--i", str(image), "--o", str(tmp_path / "public.nii.gz"),
              "--weights", str(tmp_path), "--csv-vols", str(tmp_path / "public.csv")])
    assert calls[:3] == [
        ("init", {"weights": str(tmp_path), "device": "cpu", "threads": 4}),
        ("run", image, {"keep_geometry": False, "color_lut": None}),
        ("csv", image, tmp_path / "public.csv"),
    ]
    assert saved == [tmp_path / "public.nii.gz"]


def test_synthseg_cli_does_not_import_unrelated_pipelines(monkeypatch):
    import builtins

    original_import = builtins.__import__
    observed = []

    def guarded_import(name, *args, **kwargs):
        if name == "topup.cli":
            raise AssertionError("SynthSeg must parse before importing TOPUP")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(cli, "_run_synthseg", lambda args: observed.append((args.i, args.o)))
    cli.main(["synthseg", "--i", "input.nii.gz", "--o", "output.mgz"])
    assert observed == [("input.nii.gz", "output.mgz")]
