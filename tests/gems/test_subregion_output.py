"""Output roundtrips preserve native/fine geometry and channel layout."""

import csv
import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

from fnit.gems.pipeline import SubregionLabel, SubregionResult


def test_native_and_fine_outputs_and_explicit_posterior_channels(tmp_path):
    native_affine = np.diag([-1., 1., 1., 1.])
    fine_affine = np.diag([.5, .5, .5, 1.])
    labels = np.full((2, 3, 4), 17001, np.int32)
    posterior = torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4) / 48
    fit = SimpleNamespace(labels=torch.as_tensor(labels), posterior=posterior,
                          highres_labels=nib.Nifti1Image(labels, fine_affine),
                          affine=fine_affine, min_jacobian=float("nan"))
    label = SubregionLabel(17001, "Right-Lateral-nucleus", "amygdala", "amygdala", "right",
                          "hippo-amygdala-right")
    result = SubregionResult(nib.Nifti1Image(labels, native_affine),
                            {0: "Unknown", 17001: label.name}, {label.source: fit},
                            torch.ones(labels.shape), {},
                            {17001: {"hard_volume_mm3": 24., "soft_volume_mm3": 23.5}},
                            {17001: label}, input_source="t1.nii.gz")
    files = result.save(tmp_path / "default")
    assert not any("posterior" in name for name in files)
    np.testing.assert_array_equal(nib.load(files["labels"]).dataobj, labels)
    np.testing.assert_allclose(nib.load(files["labels"]).affine, native_affine)
    fine = nib.load(files["highres/hippo-amygdala-right"])
    np.testing.assert_allclose(fine.affine, fine_affine)
    assert fine.get_data_dtype() == np.dtype(np.int32)
    report = json.loads(files["report"].read_text())
    assert report["fit_min_jacobians"][label.source] is None
    with files["label_table"].open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 1 and rows[0]["hemisphere"] == "right"
    files = result.save(tmp_path / "posterior", save_highres=False, save_posteriors=True)
    assert "highres/hippo-amygdala-right" not in files
    image = nib.load(files["posterior/hippo-amygdala-right"])
    np.testing.assert_allclose(image.affine, fine_affine)
    np.testing.assert_array_equal(image.dataobj, np.moveaxis(posterior.numpy(), 0, -1))
    assert image.shape == (2, 3, 4, 2)
