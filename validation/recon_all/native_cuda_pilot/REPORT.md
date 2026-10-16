# Isolated CUDA pilot: `mris_place_surface` intensity term

Tested on accelerator-a-1 with the FreeSurfer 8.2 `fs_sub01` result. This is an isolated
kernel experiment; the recon-all package and native runtime were not changed.
Source basis: commit `d932c45b7941662ea380a05efef580568b98d41a`,
`utils/mrisurf_compute_dxyz.cpp:1941` (`mrisComputeIntensityTerm`), called from
`utils/mrisurf_mri.cpp:603` (`MRISpositionSurface`).

The fixture contains the real sub01 `mri/mrisps.wpa.mgz` (the native placement
command's saved 256³ uint8 intensity volume) and `surf/lh.orig` (106,622
vertices). We recomputed normals from its faces. Native per-vertex CBV targets
are not saved, so test targets use each vertex's nearest MRI intensity ±3.
The pilot implements the default non-multimodal, `grad_dir=0`,
`fill_interior=0`, `l_intensity=0.2` path, with active vertices. It does not test
the complete native function or official surface output.

| accelerator-a-1 trial | CPU 1 thread | CPU 4 threads | CUDA kernel | CUDA call including alloc/copy | Different float32 gradient values |
|---|---:|---:|---:|---:|---:|
| 1 | 25.34 ms | 9.49 ms | 4.79 ms | 15.58 ms | 0 / 319,866 |
| 2 | 28.37 ms | 9.54 ms | 0.64 ms | 3.39 ms | 0 / 319,866 |
| 3 | 25.34 ms | 9.51 ms | 0.62 ms | 6.42 ms | 0 / 319,866 |

The host-to-device copy includes the 16 MiB MRI plus vertex records. The
device-to-host copy includes all 106,622 gradients. The reported CUDA call
excludes CUDA context initialization; the `cuda_context_ms` JSON field only
timed `cudaFree(nullptr)` after `cudaSetDevice` and cannot be used as a cold
start measurement. Trial 1 includes kernel warm-up. CPU four-thread outputs
were also identical to serial CPU in every trial.

This agreement is not a zero-output artifact: all 106,622 fixture vertices
have a nonzero gradient. Their gradient norms range from 0.00001585 to 1.00000013
(median 1.000000), and 40,003 vertices (37.5%) have an unsaturated intensity
error before the ±5 clipping step. These CPU-only checks are recorded in
`evidence/gradient_magnitude.json` and were computed from the tested fixture.

The completed full sub01 run took 5249.75 s. Its 16 `mris_place_surface` calls
took 1072.53 s, including about 882 s in 24 `MRISpositionSurface` calls. The
optimizer log records 220 iterations, and this intensity term runs once per
iteration. Using the median measured CPU and CUDA call times gives an
**estimated** 4.16 s saved per subject (0.08% of the full run) if all 220
calls have similar cost. The microbenchmark therefore does not justify an
integrated replacement, despite exact gradient agreement in this fixture.

A prior distinct `MRISaverageGradients` CUDA pilot also showed why kernel speed
is insufficient: 1024 iterations on a real mesh took 9.24 ms on CUDA versus
1262.4 ms on CPU with zero float32 differences, yet the rebuilt `mris_sphere`
stage took 356.93 s versus 329.62 s official and its mesh coordinates differed
from official by up to 7.97 mm. It was not suitable for replacement. Evidence:
`work/gpu_native_pilot_results/stage_lh_final_compat_20260924/report.json`.

## Follow-up profiling and decision

An isolated [CPU subfunction profile](CPU_PROFILE.md) subsequently found that
the collision/placement and MHT construction paths dominate this particular
recompiled white-surface command. That rebuilt binary differs from the installed
FreeSurfer binary, so its timings guide experiments but do not certify a replacement.

In that profile, the repulsive term contributed only 0.58% of the positioning
time. The larger measured costs were collision/placement and spatial hash table
construction. Because collision updates share a mutable hash table, the next
trial should record per-vertex move decisions and bucket ordering before a CUDA
port. Any integrated change must pass the full ordered-surface, vertex metric,
segmentation and regional-statistics gates against the installed official binary.

Raw files: `evidence/trial{1,2,3}.json`, `evidence/cpu_control-host.jsonl`,
`evidence/gradient_magnitude.json`.
SHA-256: fixture `e1ca23a1c3ed2da31d2b744af97f65a4b79d48e5a5b356625a681f96bebc6c61`,
tested binary `329accd3e6a00418809b1543a3a7b543111b527473324d386f79626e71b755de`,
source `9471ab5b7c9b9a5f83186da68839d7ac4d64199ad7572096cd19d000b53efa4e`.
