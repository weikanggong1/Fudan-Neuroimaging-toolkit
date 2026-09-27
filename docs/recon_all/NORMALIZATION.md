# Python/PyTorch T1 intensity normalization

This single stage replaces the FreeSurfer 8.2 command `mri_normalize -g 1 -seed 1234 -mprage nu.mgz T1.mgz` for a conformed T1 `nu.mgz` and its `talairach.xfm`. It runs the 1D spline, gentle correction, and two 3D control-point/bias iterations. It does not call a FreeSurfer executable and needs no FreeSurfer license, native bundle, model weights, or extra template. The current `fnit-recon-all` entry calls this Python stage, while its full morphometry outputs remain approximate.

Install the repository package with `pip install -e .`. The stage uses PyTorch, NumPy, SciPy, Numba, and NiBabel; Numba is included in the package dependencies for the exact CPU Gaussian path. `SimpleITK` is used by the separate [N4 SITK stage](N4_SITK_VALIDATION.md), not by this normalizer.

## Command line

```bash
fnit-normalize \
  --input /subjects/sub01/mri/nu.mgz \
  --xfm /subjects/sub01/mri/transforms/talairach.xfm \
  --output /subjects/sub01/mri/T1.mgz \
  --device cuda:0
```

On a host without CUDA, use `--device cpu`; omitting `--device` selects CUDA when available and CPU otherwise. The command prints a JSON record with elapsed seconds for each step. `--three-d-iterations 0` and `1` reproduce the intermediate FreeSurfer `-n 0` and `-n 1` checkpoints; the default is `2`. `--diagnostic-dir /path` writes the float32 input, control mask, and float32 bias of each 3D iteration as MGH files.

## Python API

```python
from fnit.recon_all.normalization import normalize_t1

report = normalize_t1(
    "/subjects/sub01/mri/nu.mgz",
    "/subjects/sub01/mri/transforms/talairach.xfm",
    "/subjects/sub01/mri/T1.mgz",
    device="cuda:0",
)
print(report["total_seconds"], report["steps"])
```

The output is a uint8 `T1.mgz`; internal float32 images are retained between passes. `nu.mgz` must be the conformed FreeSurfer T1 stage input, with a Talairach transform in matching geometry. The source translation follows FreeSurfer commit `d932c45b7941662ea380a05efef580568b98d41a`. In the `fs_sub01` CPU diagnostic pair, every intermediate float32 image, control mask, bias image, and final 256³ uint8 volume matched the native output voxel for voxel; the H100 pair also matched at the final T1 volume. See the [checkpoint and timing report](../../validation/recon_all/python_gpu_port/experimental/NORMALIZE_FIRST_PASS.md). This is a one-subject match, and the production full recon-all entry has not yet been switched to this stage.

## CPU and GPU work

| Step with `--device cuda:0` | Where it runs |
|---|---|
| Read/write MGH/MGZ, Talairach transform parsing, 1D histograms and spline coefficients | CPU, NiBabel/NumPy/SciPy |
| 1D voxel scaling and the bias application after each pass | H100, PyTorch float32 |
| Gentle and both 3D control-point selections, tissue histograms, neighborhood tests | CPU, NumPy/SciPy |
| Voronoi chessboard distance and index sorting | CPU, SciPy/NumPy |
| Voronoi wavefront averages and three sigma-8 Gaussian convolutions | H100, PyTorch float32 |

The CPU implementation uses Numba for the ordered Gaussian accumulation required for exact voxel parity. The production normalizer contains no subprocess call or native FreeSurfer binary lookup. The separate native benchmark invokes FreeSurfer only to establish the reference output and time.

For the same `fs_sub01` T1, same-host no-diagnostic CLI pairs were **76.94 s native vs 60.55 s Python on headcw CPU** and **116.31 s native vs 75.60 s Python on gpucw1 H100 GPU1**. Both Python outputs had 0/16,777,216 voxels different from their native partner. On H100, the two CPU control-point searches took 20.24 and 32.60 s of the 68.87 s resident Python run; these remain the main speed limit. The full staged command, per-step times, hashes, and remaining cross-subject validation gate are in the [validation report](../../validation/recon_all/python_gpu_port/experimental/NORMALIZE_FIRST_PASS.md).

## Second normalization with aseg and brain mask

The separate `mri_normalize -seed 1234 -mprage -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz` translation is available as a Python API and command. It accepts three conformed MGH/MGZ inputs on the same voxel grid and calls no FreeSurfer executable. Its Fast Marching medial WM ridge, outlier removal, initial bias, gentle correction, and two 3D passes run with NumPy/Numba/PyTorch. The exact CPU path was validated; the CUDA path of this second command has not yet been paired with the native result.

```bash
fnit-normalize-aseg \
  --norm /subjects/sub01/mri/norm.mgz \
  --aseg /subjects/sub01/mri/aseg.presurf.mgz \
  --brainmask /subjects/sub01/mri/brainmask.mgz \
  --output /subjects/sub01/mri/brain.mgz \
  --device cpu
```

```python
from fnit.recon_all.normalization import normalize_t1_aseg

report = normalize_t1_aseg(
    "/subjects/sub01/mri/norm.mgz",
    "/subjects/sub01/mri/aseg.presurf.mgz",
    "/subjects/sub01/mri/brainmask.mgz",
    "/subjects/sub01/mri/brain.mgz",
    device="cpu",
)
```

On the frozen `fs_sub01` inputs, the independent Python ridge, filtered control mask, outlier map, and initial float32 bias output matched the official diagnostic volumes exactly. The complete default two-pass output matched the official `brain.mgz` in all 16,777,216 voxels, including the 284-byte MGH header and voxel payload. The fresh native run appended 996 bytes of trailing metadata, and the earlier completed subject output appended 451 bytes; compressed and complete decompressed file hashes therefore differ. This is an isolated single-subject stage validation; full-pipeline numerical equivalence has not been established. See the [second-pass validation report](../../validation/recon_all/python_gpu_port/NORMALIZE_SECOND_PASS.md).
