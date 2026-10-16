# `mris_inflate -no-save-sulc`: Python stage and fixed-subject parity

The fixed FreeSurfer 8.2.0-1 recon-all calls are:

```text
mris_inflate -no-save-sulc ../surf/lh.smoothwm.nofix ../surf/lh.inflated.nofix
mris_inflate -no-save-sulc ../surf/rh.smoothwm.nofix ../surf/rh.inflated.nofix
```

The source reference is commit `d932c45b7941662ea380a05efef580568b98d41a`, primarily `mris_inflate/mris_inflate.cpp`, `utils/mrisurf_integrate.cpp`, `utils/mrisurf_compute_dxyz.cpp`, `utils/mrisurf_timeStep.cpp`, and `utils/MRIScomputeTriangleProperties_extracted.h`. [`inflate_python.py`](../../../src/fnit/recon_all/inflate_python.py) implements the fixed spring, distance, momentum, stopping, centering, and area-restoration path in NumPy/Numba on CPU. It calls no FreeSurfer program and uses no license at runtime. Its API is `inflate_surface(input_path, output_path)` and its CLI is `python -m fnit.recon_all.inflate_python INPUT OUTPUT`.

## Numerical result on `fs_sub01`

Starting independently from each saved official `smoothwm.nofix`, the current Python CLI output matches the official `inflated.nofix` **at every ordered vertex and face**, with the same FreeSurfer volume geometry metadata:

| Hemisphere | Exact ordered vertices | Exact ordered faces | Maximum vertex distance |
| --- | ---: | ---: | ---: |
| LH | 102,764 / 102,764 | 205,560 / 205,560 | 0 mm |
| RH | 101,454 / 101,454 | 202,936 / 202,936 | 0 mm |

The [file-level report](inflate_exact_fs_sub01_report.json) records source and output SHA-256 values. Whole-file hashes differ because the Python writer supplies different provenance text; geometry and volume metadata are exact. Both hemispheres perform 60 updates for this subject. A fresh native `-W 1` RH diagnostic and the independent Python loop matched **all 60 raw coordinate snapshots** at all 101,454 vertices. The earlier LH `-N 1 -A 16 -W 1` diagnostic also matched each of six averaging-level updates exactly after correction; the final complete LH output matches exactly. These are isolated stage checks, not a complete recon-all run.

The exact result required three source-order arithmetic corrections. FreeSurfer first multiplies the old momentum by float32 `0.9` before adding the double-precision new-gradient term. Its distance vectors, normalization, and neighbor accumulation stay float32. Its per-triangle cross products and lengths also stay float32 before a double-precision total-area sum is rounded to float32. The previous Python version promoted the old momentum, distance accumulation, and face cross products too early. It stayed within 0.001 mm at the inflation output but caused much larger downstream quick-sphere differences; those earlier error and timing reports in [`inflate_stage_report.json`](inflate_stage_report.json) and [`inflate_cpu_report.json`](inflate_cpu_report.json) describe that superseded version.

Connected Python `smoothwm → inflated → qsphere` runs now match the official quick spheres on both hemispheres: 102,764/101,454 ordered vertices, 205,560/202,936 faces, and all volume geometry fields exactly. See the [bilateral chain report](inflate_to_qsphere_chain_exact_report.json) and [`SPHERE_QUICK_STATUS.md`](SPHERE_QUICK_STATUS.md) for the remaining boundaries.

## Timing and scope

A [sequential same-input LH pair](inflate_qsphere_paired_lh_control-host.json) on control-host took **6.986 s native and 41.710 s Python** for full `mris_inflate` CLI execution, including input/output I/O and process startup. Both output geometries and volume metadata were exactly equal. This is one pair, so it estimates about **6.0× slower Python CPU execution**, not a stable distribution. The separate corrected LH/RH Python calls that ran concurrently took 41.34/46.44 s; the RH value is not a paired native comparison. This stage remains CPU-only and has been checked on one T1. Other surface statuses, ripped vertices, degenerate-normal perturbations, optional energy terms, and multi-subject parity remain unverified. It has not yet been connected to a native-free recon-all runner.

## Exact CPU kernel optimization

The current implementation compiles source-order neighbor averaging,
tangent-height RMS, face-corner normal accumulation, and spring neighbor sums
with Numba. The new accumulation and spring kernels matched the previous
float32 arrays bit for bit on the frozen surface and a perturbed copy. The
complete updated CLI then matched the official ordered coordinates, faces,
and volume geometry for both hemispheres. The [structured bilateral
report](inflate_qsphere_numba_optimized_report.json) stores input, output,
and implementation SHA-256 values.

One independent control-host CLI observation took **30.31 s LH** and **24.25 s RH**
for inflation, including interpreter startup and I/O. The earlier clean LH
native/Python pair was 6.986/41.710 s using the previous Python source. These
new calls occurred under different shared-host load, so the change in observed
seconds is not a paired speed estimate. This stage remains CPU-only.
