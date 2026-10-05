"""SynthSeg+ public output and repeated-call contracts."""

import csv
from argparse import Namespace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.synthseg_parc import SynthSegPlus
from fnit.synthseg_parc.segment import SynthSegParcResult
from fnit.synthseg_parc.synthseg_plus import SynthSegPlusResult


def test_soft_volume_csv_requires_volume_inference(tmp_path):
    image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int32), np.eye(4))
    result = SynthSegPlusResult(image, image, image,
                                {3: "Left-Cerebral-Cortex", 1001: "ctx-lh-bankssts"})
    with pytest.raises(ValueError, match="volumes=True"):
        result.write_volumes_csv("subject_T1w.nii.gz", tmp_path / "vol.csv")
    result.volumes_mm3 = {3: 12.345, 1001: 3.125}
    result.total_intracranial_mm3 = 100.5
    result.write_volumes_csv("subject_T1w.nii.gz", tmp_path / "vol.csv")
    with (tmp_path / "vol.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows == [["subject", "total intracranial", "Left-Cerebral-Cortex",
                     "ctx-lh-bankssts"],
                    ["subject_T1w", "100.5", "12.345", "3.125"]]


def test_plus_reuses_both_weighted_models(tmp_path, monkeypatch):
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)), t1)
    model = object.__new__(SynthSegPlus)
    model.device = "cpu"
    model.segment_weights = model.segmentation_labels = model.parc_weights = t1
    model.topology_classes = t1
    model.label_names = {0: "Background"}
    model._segmenter = model._parcellator = None
    created = []

    def constructor(*args, **kwargs):
        assert kwargs == {"cudnn_tf32": True}
        instance = object()
        created.append(instance)
        return instance

    from fnit.synthseg_parc import synthseg_plus
    monkeypatch.setattr(synthseg_plus, "SynthSegSegmenter", constructor)
    monkeypatch.setattr(synthseg_plus, "SynthSegParc", constructor)

    def run(*args, **kwargs):
        assert kwargs["segmenter"] is created[0]
        assert kwargs["parcellator"] is created[1]
        labels = torch.zeros((2, 2, 2), dtype=torch.int64)
        return SynthSegParcResult(labels, labels, labels, np.eye(4), np.eye(4),
                                  (2, 2, 2))

    monkeypatch.setattr(synthseg_plus, "run_synthseg_parc_t1", run)
    model(t1, keep_geometry=False)
    model(t1, keep_geometry=False)
    assert len(created) == 2


@pytest.mark.parametrize("fast", [False, True])
def test_cli_parc_volume_option_requests_soft_volumes(tmp_path, monkeypatch, fast):
    source = tmp_path / "t1.nii.gz"
    source.touch()
    calls = []

    class DummyImage:
        def save(self, path):
            Path(path).write_text("labels")

    class DummyResult:
        combined = DummyImage()
        cortical_parcellation = DummyImage()

        def write_volumes_csv(self, source, path):
            calls.append((source, path))
            Path(path).write_text("subject,total intracranial\n")

    class DummyModel:
        def __init__(self, **kwargs):
            pass

        def __call__(self, source, *, keep_geometry, volumes, fast):
            assert volumes is True
            assert fast is args.fast
            return DummyResult()

    import fnit.synthseg_parc
    monkeypatch.setattr(fnit.synthseg_parc, "SynthSegPlus", DummyModel)
    from fnit.cli import _run_synthseg
    args = Namespace(i=str(source), o=str(tmp_path / "plus.nii.gz"),
                     parc=True, csv_vols=str(tmp_path / "vol.csv"),
                     color_lut=None, threads=2, weights=None,
                     parc_weights=None, device="cpu", keep_geometry=False,
                     parc_out=None, fast=fast)
    _run_synthseg(args)
    assert calls == [(source, args.csv_vols)]
    assert (tmp_path / "plus.nii.gz").is_file()
    assert (tmp_path / "vol.csv").is_file()
