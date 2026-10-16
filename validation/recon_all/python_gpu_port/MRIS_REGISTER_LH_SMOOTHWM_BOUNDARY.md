# LH independent smoothwm curvature, fold cleanup and overlap repair

The frozen `fs_sub01` LH `smoothwm`, original sphere, exact Python sulc
`debug0056` seed and atlas have SHA-256 values in the paired JSON reports.
This is a bounded FreeSurfer 8.2 comparison on stored native checkpoints,
not an end-to-end reconstruction.

The production Python/PyTorch `smoothwm_mean_curvature` computes raw H from
the frozen surface without reading the native raw H. Its
[all-vertex report](mris_register_lh_smoothwm_production_raw_control-host.json)
matches **106,622/106,622 float32 values bitwise** and has the same raw-array
SHA-256 `319c755b667a04331ef0c545714aaf5897c9a04e0c08a882b69b48542f0ab39c`.
The measured 14.05 s is one CPU run including Numba startup, not a paired
native speed comparison.

Using that independent raw H, the [first-update report](mris_register_lh_default_epoch0057_independent_fixed_control-host.json)
selects `dt=2.721714973449707` and matches the native saved LH
`debug0057` surface at **106,622/106,622 ordered vertices**, maximum
coordinate error 0 mm. It starts from the previously verified exact Python
`debug0056` sulc seed. Setup, force, averaging, spring and line search took
22.51, 18.25, 6.91, 0.008 and 0.55 s respectively in that shared-host CPU run;
these exclude file I/O and are not a native speed ratio.

The [continuous probe](probe_mris_register_sno2_epoch.py) retains its own
coordinates between updates and reads native snapshots only for comparison.
The sigma-4, 2, 1 and 0.5 normal smoothwm passes match **all 45 saved
`debug0057`–`debug0101` surfaces**, 106,622/106,622 ordered vertices and all
faces at each checkpoint, maximum coordinate error 0 mm. Five bounded runs
cover the path without gaps:

| Report | Native epochs checked | Exact vertices at every epoch |
| --- | --- | ---: |
| [sigma 4 first segment](mris_register_lh_default_epoch0057_0068_continuous_control-host.json) | 0057–0068 | 106,622 / 106,622 |
| [sigma 4 second segment](mris_register_lh_default_epoch0069_0080_resume_control-host.json) | 0069–0080 | 106,622 / 106,622 |
| [sigma 2](mris_register_lh_default_epoch0081_0087_resume_control-host.json) | 0081–0087 | 106,622 / 106,622 |
| [sigma 1](mris_register_lh_default_epoch0088_0094_resume_control-host.json) | 0088–0094 | 106,622 / 106,622 |
| [sigma 0.5](mris_register_lh_default_epoch0095_0101_resume_control-host.json) | 0095–0101 | 106,622 / 106,622 |

Each resumed run verifies that its starting native checkpoint has the same
coordinate-array SHA-256 as the preceding independently computed Python
state and the same file SHA-256 as the preceding report. Native intermediate
coordinates are not injected during a run. The probe receives the averaging
schedule observed in the native log. The later fold cleanup and negative-face
repair are checked below; independent stopping decisions remain unverified. The sum of measured CPU
update times across these segments is about 763 s, excluding each segment's
repeated setup and file I/O. This is not a matched native or GPU benchmark.

The previous [failed diagnostic](mris_register_lh_default_epoch0057_0058_continuous_control-host.json)
reprojected coordinates inside both distance and area gradient helpers at
the second integration step. It matched 18/106,622 vertices at `debug0058`
with 0.00157928466796875 mm maximum error. The FreeSurfer source calls
`mrisProjectSurface` at the end of each integration iteration and once at
the start of a new averaging stage. Passing `project=False` to both helpers
within an existing stage removes the duplicate projections and restores
exact second-step parity. Starting the 256-average stage with one projection
then matches `debug0059`; the same one-projection rule also matches every later normal pass through `debug0101`.

The first continuation reports evaluated reloaded native vertices as
float64 because that is the `nibabel` reader default. Their
`native_reference_sse` fields therefore do **not** measure the float32
registration objective and should not be compared with the native log.
The [corrected probe](probe_mris_register_sno2_epoch.py) casts reference
vertices to float32. Without rerunning the exact 45-step geometry, the
[hash-linked score audit](mris_register_lh_default_epoch0057_0101_selected_score_audit_control-host.json)
compares each selected float32 trial SSE with the
[native one-decimal status log](mris_register_lh_default_native_run_control-host.log): the largest absolute difference is 0.3441 across 45 updates. For
`debug0058`, the Python selected-trial SSE is 1,418,705.2775 versus the
native printed 1,418,705.2. These comparisons do not establish bitwise SSE
parity. The smoothwm raw-curvature fit currently uses CPU Numba for
source-order matrix operations.

## Fold cleanup after `debug0101`

The [native default run](mris_register_lh_fold_overlap_native_run_accelerator-a-1.log)
starts the additional fold cleanup because the normal path has 193 negative
triangles. Pinned `mrisurf_integrate.cpp` multiplies nonlinear area weight by
100, divides percentage area, distance, correlation and spring weights by
100, and starts a 64-average integration stage. The Python/PyTorch distance
and area helpers now accept these weights while retaining their original
defaults. The continuation restores the sigma-0.5 grids from the exact
`debug0094` seed. Its [101 resume proof](mris_register_lh_resume94_101_proven.json)
and [102 resume proof](mris_register_lh_resume94_102_proven.json) link the
saved input, checkpoint and coordinate SHA-256 values.

The [first fold report](mris_register_lh_fold_cleanup_epoch0102_accelerator-a-1.json)
independently selects `dt=5.583333492279053` and matches native
`debug0102` at all 106,622 ordered vertices. The [continuous report](mris_register_lh_fold_cleanup_epoch0103_0107_accelerator-a-1.json)
then matches all five `debug0103`–`debug0107` surfaces at 106,622/106,622
vertices each, maximum error 0 mm. It also checks all ordered faces. Native
`debug0103` has `dt=0` inside the same 64-average integration call; exactly
one terminal sphere projection reproduces every vertex. Projecting again
at the start of this step caused an earlier 1.91e-6 mm discrepancy.

## Final negative-face smoothing

The native call enters `MRISremoveOverlapWithSmoothing` at 137 negative
triangles after `debug0107`. The new
[`remove_overlap_sphere`](../../../src/fnit/recon_all/mris_register_overlap.py)
uses ordered one-ring neighbors, source-order double accumulation of vertex
displacements, float32 stored updates, radial projection, and the native
marked-neighborhood and stopping rules. The
[replay probe](probe_mris_register_overlap.py) starts from the frozen native
`debug0107` and compares every iteration to the native log. On accelerator-a-1,
**all 102 negative-triangle counts matched**, the final count was zero, and
the final sphere matched the official `lh.sphere.reg` at **106,622/106,622
ordered vertices and all faces**, maximum error 0 mm. Both CPU and H100 CUDA
passed on the same input:

| accelerator-a-1 device | Repair compute, excluding I/O and tensor staging | Exact vertices |
| --- | ---: | ---: |
| [4-thread CPU](mris_register_lh_overlap_cpu_accelerator-a-1.json) | 6.390 s | 106,622 / 106,622 |
| [H100 CUDA](mris_register_lh_overlap_cuda_accelerator-a-1.json) | 2.874 s | 106,622 / 106,622 |

This is one matched-host observation: CUDA was 2.22 times as fast for this
repair computation, including neighbor construction and 102 updates. It is
not a native-vs-Python or full recon-all timing comparison. The RH native
repair has zero negative triangles; the
[PyTorch no-op check](mris_register_rh_overlap_noop_cpu_control-host.json) returns
all 105,541 final vertices exactly in 0.017 s CPU compute.

The fold cleanup and repair have each passed against native checkpoints.
The tests used native snapshots as their starting inputs and do not replace
a connected T1-to-`sphere.reg` run. The original bounded geometry replay
uses the native iteration schedule. A separate source-rule audit now matches
all saved bilateral schedule decisions, and an opt-in LH runtime check
reproduces the first four smoothwm surfaces; see
[the schedule result](MRIS_REGISTER_STATUS.md#source-derived-smoothwm-stopping-and-sigma-schedule).
