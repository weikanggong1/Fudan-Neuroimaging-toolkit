# `mri_cc -aseg` Python port: paired FreeSurfer 8.2 validation

This isolated stage implements the active `mri_cc` branch at FreeSurfer source
commit `d932c45b7941662ea380a05efef580568b98d41a`. It consumes
`aseg.auto_noCCseg.mgz` and `norm.mgz`, and writes `aseg.auto.mgz` and
`transforms/cc_up.lta`. It calls NumPy, SciPy, and nibabel; no FreeSurfer binary
is used by the Python stage.

The native reference was rerun twice on headcw using the same frozen inputs,
without recon-all. A normal run took 89.67 s; a diagnostic run took 92.93 s and
provided `m.mgz`, `asegx.mgz`, and `s1`–`s4` intermediate images. Both native
`aseg.auto.mgz` outputs matched the archived output by compressed-file SHA-256.

| Check | Frozen `fs_sub01` result |
|---|---:|
| Native versus Python output voxel differences | **0 / 16,777,216** |
| Native versus Python MGH voxel-payload byte differences | **0** |
| Native versus Python MGH header byte differences | **0** |
| Voxels changed from `aseg.auto_noCCseg.mgz` | 2,263 each |
| CC labels 251, 252, 253, 254, 255 | 714, 270, 292, 327, 660 each |
| `cc_up.lta` matrix maximum absolute difference | 7.63 × 10⁻⁶ |
| Native `mri_cc` wall time | 89.67 s |
| Python stage wall time | 17.41 s |
| Ratio | 5.15× |

The Python LTA has the same volume geometry and was accepted by native
`lta_convert --inlta ... --outlta ...`. It differs numerically by small
single-precision matrix rounding. The full `.mgz` compressed-file hashes differ:
the Python writer keeps the input's trailing metadata tags, whereas native
`MRIwrite` rewrites them. The image header and complete voxel payload match.

Intermediate validation showed exact voxel agreement for the bilateral cortex
midline mask, `s1` mutual white-matter seed, `s2` center-slice expansion,
`s3` adjacent-slice expansion, `s4` component selection, and the five-slice
fornix mask. The final five-part label image also matches exactly. The first
resampled whole-aseg diagnostic image differs at one voxel outside the
callosum region because of transform rounding; this does not alter `s1`–`s4`
or the final output.

The implementation lives in `mri_cc_plane_python.py`,
`mri_cc_masks_python.py`, and `mri_cc_python.py`. Its file API is
`run_mri_cc(aseg_file, norm_file, output_file, lta_file)`; its in-memory API
is `segment_callosum(aseg, norm)`. The stage is now wired after `mri_ca_normalize` in the recon-all dispatcher;
it writes `aseg.auto_noCCseg.mgz`, `aseg.auto.mgz`, and `cc_up.lta`, then copies
the edited segmentation to the current `aseg.presurf.mgz` and `aseg.mgz`
approximation. The paired numerical evidence covers one subject and the
fixed 256³ recon-all invocation; broader input validation is still needed.
The stage currently runs on CPU through NumPy and SciPy.

The executable comparison is `validate_mri_cc_python.py`; its completed
machine-readable record is `mri_cc_python_pair_report.json`.

## Current Conda E2E candidate inputs (2026-09-27)

After the corpus-callosum stage was connected to the runner, a new isolated
subject test used the current candidate `synthseg.rca.mgz` and `norm.mgz` as
inputs. Official `mri_cc` and the Python port each changed 2,263 voxels; their
`aseg.auto.mgz` outputs were identical at all 16,777,216 voxels, with matching
MGH header and payload bytes. They also matched the archived official
`aseg.auto.mgz` voxelwise. On headcw, the official command took 99.16 s; the
Python function took 17.34 s (whole validator process 18.96 s). The connected
runner helper reduced the prior candidate `aseg.auto` difference from 2,263 to
0 voxels, while the later official final `aseg` still differs by 104,986 voxels.
The focused integration test passed 1/1 on gpucw1. The full recon-all subject
has not yet been rerun after this connection. See the
[current-input report](native_cpp_conda_20260927/MRI_CC_FEASIBILITY.md) and its
machine-readable paired and integrated-stage results.
