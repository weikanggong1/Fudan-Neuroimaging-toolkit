"""Post-volume surface entrypoint with a UKB recon-all ZIP."""

from pathlib import Path
from types import SimpleNamespace
import json
import zipfile

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri import surface_pipeline


@pytest.mark.parametrize("scanner_offset", [0.0, 2.0])
def test_surface_from_volume_checks_recon_grid_and_uses_clean_bold(
    tmp_path, monkeypatch, scanner_offset
):
    volume = tmp_path / "volume"
    (volume / "reg").mkdir(parents=True)
    volume_mgh = tmp_path / "volume.mgz"
    nib.save(nib.MGHImage(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), volume_mgh)
    volume_affine = nib.load(str(volume_mgh)).affine
    mgh_path = tmp_path / "001.mgz"
    scanner_affine = np.eye(4)
    scanner_affine[0, 3] = scanner_offset
    nib.save(nib.MGHImage(np.zeros((4, 4, 4), dtype=np.float32), scanner_affine), mgh_path)
    reference = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), volume_affine)
    nib.save(reference, volume / "T1_brain.nii.gz")
    nib.save(reference, volume / "MNI152_2mm_brain.nii.gz")
    clean = volume / "filtered_func_data_clean_MNI152_2mm.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4, 5), dtype=np.float32), volume_affine), clean)
    (volume / "pipeline_report.json").write_text(json.dumps({
        "outputs": {"clean_mni": clean.name},
        "wm_csf_motion_regression": {"wm": True, "csf": True, "motion": True},
    }))
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4, 3), dtype=np.float32), volume_affine),
             volume / "reg/MNI152_2mm_to_T1_pull_ras.nii.gz")
    np.savetxt(volume / "reg/T1_to_MNI152_2mm_affine.mat", np.eye(4))
    archive_path = tmp_path / "structural.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(mgh_path, "FreeSurfer/mri/orig/001.mgz")
        for name in ("orig.mgz", "wmparc.mgz"):
            archive.write(mgh_path, f"FreeSurfer/mri/{name}")
        for hemi in ("lh", "rh"):
            for name in ("white", "pial", "sphere.reg", "thickness"):
                archive.writestr(f"FreeSurfer/surf/{hemi}.{name}", b"surface")

    captured = {}

    def prepare(**kwargs):
        subject = Path(kwargs["subject_dir"])
        assert (subject / "surf/lh.white").is_file()
        captured["affine"] = kwargs["initial_t1_to_mni_world"]
        return SimpleNamespace(left="left", right="right", subject_rois="roi", atlas_rois="atlas")

    def project(**kwargs):
        captured["clean"] = kwargs["clean_mni"]
        return "projected"

    monkeypatch.setattr(surface_pipeline, "prepare_fs_sphere_projection_inputs", prepare)
    monkeypatch.setattr(surface_pipeline, "run_surface_from_mni", project)
    kwargs = dict(
        volume_dir=volume, recon_all=archive_path, hcp_assets_dir=tmp_path,
        output_dir=tmp_path / "surface", registration="fs",
    )
    if scanner_offset:
        with pytest.raises(ValueError, match="different grids"):
            surface_pipeline.run_surface_from_volume(**kwargs)
        assert not captured
        return
    result = surface_pipeline.run_surface_from_volume(**kwargs)
    assert result == "projected"
    assert captured["clean"] == clean
    np.testing.assert_allclose(captured["affine"], np.eye(4), atol=1e-5)
