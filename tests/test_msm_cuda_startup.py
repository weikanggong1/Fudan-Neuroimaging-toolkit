"""A public small-mesh API control, not a registration benchmark."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize("method", ["msmall", "msmsulc"])
def test_registration_initializes_cuda_in_a_new_python_process(tmp_path, method):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    code = r'''
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from fnit.msm import MSMAllConfig, MSMAllInputs, MSMSulcConfig, MSMSulcInputs, run_msmall, run_msmsulc
from fnit.msm.msmsulc import _ico

assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
directory = Path(sys.argv[1])
vertices, faces = _ico(1)
sphere = directory / "public.surf.gii"
metric = directory / "public.func.gii"
affine = directory / "public.mat"
nib.save(nib.GiftiImage(darrays=[
    nib.gifti.GiftiDataArray(vertices.astype(np.float32), intent=1008),
    nib.gifti.GiftiDataArray(faces.astype(np.int32), intent=1009),
]), sphere)
nib.save(nib.GiftiImage(darrays=[
    nib.gifti.GiftiDataArray((vertices[:, i] / 100).astype(np.float32), intent=1002)
    for i in range(3)
]), metric)
np.savetxt(affine, np.eye(4))
if sys.argv[2] == "msmall":
    entry = MSMAllInputs(sphere, metric, sphere, metric)
    config = MSMAllConfig(simval=(2,), iterations=(1,), control_grid=(1,),
                         sampling_grid=(2,), data_grid=(1,), regularization=(1e-5,))
    result = run_msmall({"L": entry, "R": entry}, directory / "result", device="cuda:0", config=config)
else:
    entry = MSMSulcInputs(sphere, sphere, metric, sphere, metric, affine)
    config = MSMSulcConfig(iterations=(1,1,1,1), control_grid=(1,1,1,1),
                          sampling_grid=(2,2,2,2), data_grid=(1,1,1,1))
    result = run_msmsulc({"L": entry, "R": entry}, directory / "result", device="cuda:0", config=config)
assert torch.cuda.is_initialized()
for path in result.values():
    image = nib.load(path)
    assert image.darrays[0].data.shape == vertices.shape
    assert np.isfinite(image.darrays[0].data).all()
assert (directory / "result/registration_report.json").is_file()
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), method], env=environment,
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
