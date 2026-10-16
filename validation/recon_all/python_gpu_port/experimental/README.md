# `mri_normalize` experiment: accuracy gate failed

This isolated prototype is **not** a recon-all replacement. The reference is
FreeSurfer 8.2.0, build `d932c45`, on the completed `fs_sub01` T1 subject.
The executed commands in `scripts/recon-all.log` were:

```text
mri_normalize -g 1 -seed 1234 -mprage nu.mgz T1.mgz
mri_normalize -seed 1234 -mprage -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz
```

The first native call took 109.49 s and uses Talairach-window histogram peaks,
a 12-point intensity spline, gentle normalization, and two 3D passes. The
second took 145.07 s. It selects medial white-matter control points from
aseg labels 2/41, estimates a bias field, then runs gentle normalization and
two 3D passes. These paths are in the pinned [program source](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_normalize/mri_normalize.cpp)
and `utils/mrinorm.cpp` at the same commit.

[`normalize_gpu.py`](normalize_gpu.py) implements a deliberately limited Torch
baseline. Pass 1 scales the global 90–140 intensity mode to 110. Pass 2 uses
26-neighbor morphological depth maxima of aseg white matter, rejects controls
below the white-matter mode minus 10, then estimates one field with 4× pooled
Gaussian smoothing (sigma 8 voxels) and scales to 110. It preserves input MGH
geometry and voxel type. It omits the native spline, Euclidean distance ridge
and gradient suppression, outlier rules, iterative Voronoi averaging, and
adaptive gentle/3D normalization passes.

| Pass | Prototype CPU wall (s) | Foreground MAE | Foreground exact | Differing voxels |
| --- | ---: | ---: | ---: | ---: |
| `nu → T1` | 0.783 | 3.805 | 21.24% | 1,753,902 |
| `norm → brain` | 7.458 | 3.263 | 18.99% | 1,028,737 |

These results are from control-host CPU against frozen reference outputs, with no
native rerun on that host. The wall times cannot establish speedup, and the
large voxel errors reject both replacements. No H100 timing was run after the
accuracy failure. Detailed counts, hashes, and error quantiles are in
[`normalize_cpu_report.json`](normalize_cpu_report.json). The replay command
is in [`benchmark_normalize.py`](benchmark_normalize.py); the three synthetic
tests passed on control-host. Future work should translate and validate each native
intermediate control map and bias field before attempting a full pass.
