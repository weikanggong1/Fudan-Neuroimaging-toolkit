"""The generic benchmark delegates geometry and brain axes to the adapter."""
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "benchmark_multimodal_cpu", Path(__file__).resolve().parents[2] / "tools/benchmark_multimodal_cpu.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_surface_points_faces_are_delegated(tmp_path):
    surface = nib.gifti.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(np.eye(3, dtype=np.float32), intent="NIFTI_INTENT_POINTSET"),
        nib.gifti.GiftiDataArray(np.array([[0, 1, 2]], dtype=np.int32), intent="NIFTI_INTENT_TRIANGLE"),
    ])
    path = tmp_path / "sphere.surf.gii"; nib.save(surface, path)
    result = MODULE.compare_outputs({"sphere": path}, {"sphere": path})["sphere"]
    assert result == {"status": "format_needs_adapter_comparison", "same_bytes": True}


def test_cifti_brain_axis_is_delegated(tmp_path):
    series = nib.cifti2.SeriesAxis(0, .735, 4)
    brain = nib.cifti2.BrainModelAxis.from_surface([0, 2], 3, name="CortexLeft")
    image = nib.Cifti2Image(np.zeros((4, 2), dtype=np.float32),
                           nib.cifti2.Cifti2Header.from_axes((series, brain)))
    path = tmp_path / "bold.dtseries.nii"; nib.save(image, path)
    result = MODULE.compare_outputs({"cifti": path}, {"cifti": path})["cifti"]
    assert result["status"] == "format_needs_adapter_comparison"


def test_single_gifti_metric_keeps_numeric_comparison(tmp_path):
    image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(np.arange(3, dtype=np.float32))])
    path = tmp_path / "metric.func.gii"; nib.save(image, path)
    result = MODULE.compare_outputs({"metric": path}, {"metric": path})["metric"]
    assert result["status"] == "compared"
    assert result["max_absolute_error"] == 0
