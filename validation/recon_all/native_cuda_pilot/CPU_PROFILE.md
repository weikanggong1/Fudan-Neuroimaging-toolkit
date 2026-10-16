# Isolated FreeSurfer 8.2 `mris_place_surface` CPU profiling

This is a research experiment in `work/native_place_profile_20260924`. It is not part of the published GPU recon-all bundle. The main certification on accelerator-a-1 GPU1 was not touched. The instrumentation and comparison scripts are included beside this report; the original FreeSurfer source, binaries, and MRI inputs remain outside Git.

## Real-input replay and build parity

- Source: FreeSurfer commit `d932c45b7941662ea380a05efef580568b98d41a` (`utils/mrisurf_mri.cpp`, `utils/mrisurf_timeStep.cpp`). The installed reference is `/opt/reference/Freesurfer/8.2.0-1/bin/mris_place_surface` (SHA-256 `67139c3590f961f879a4214cd01f173b42986c7b178d1851dfa3f08e8acf12aa`).
- Input: actual sub-01 `wm.mgz`, `brain.finalsurfs.mgz`, `aseg.presurf.mgz`, `lh.orig`, and `autodet.gw.stats.lh.dat`; same `WhitePreAparc lh` command and 4 threads as the original recon-all log. `evidence/inputs.sha256` records the copies.
- The installed reference binary replay took 98.42 s on control-host. Its `lh.white.preaparc` 106,622 vertices and 213,240 faces, vertex coordinates and ordered faces, and `mrisps.wpa.mgz` voxels exactly matched the original sub-01 outputs (`evidence/original_vs_official.json`).
- The uninstrumented binary rebuilt from the source on control-host took 118.24 s. Its faces and `mrisps.wpa.mgz` voxels matched the official binary, but 98,057/106,622 vertex coordinates differed (maximum 0.6300 mm, P99 0.0070 mm; `evidence/input_vs_clean.json`). Thus this compiler/toolchain build is **not numerically identical to the installed FreeSurfer surface output**. The toolchain uses conda GCC 11.2, C++17, the local ITK 5.3 build, and conda libtiff 6. Timings measured in the instrumented build must be interpreted as *that rebuild's* hotspots; they do not establish exact runtime fractions of the installed binary or qualify a CUDA replacement.

## Instrumentation

`instrument.py` adds per-`MRISpositionSurface` wall timers for spatial hash table construction, target and intensity terms, other gradient terms, repulsive term, step preparation, `mrisAsynchronousTimeStep`, metric properties, RMS, and SSE. `instrument_timestep.py` splits the step into vertex bounds, face and vertex subvolume assignment, merge, and two collision/placement passes. Both patch scripts modify only the copy-on-write source under `/tmp/native_place_profile_20260924/source`; the source and binaries used in the release remain untouched. The built source copies are preserved as `mrisurf_mri_instrumented.cpp` and `mrisurf_timeStep_instrumented.cpp` in this work directory.

The timed build completed the same command in 100.23 s. Its 106,622 vertex coordinates, ordered 213,240 faces, and every `mrisps.wpa.mgz` voxel were **exactly equal** to the uninstrumented rebuild (`evidence/clean_vs_profile.json`). The 100.23 s vs 118.24 s wall-time difference reflects host load and run-to-run variability; it is not a measured acceleration from timing code.

## Measured subfunctions on sub-01

`evidence/profile_summary.json` aggregates the four `MRISpositionSurface` calls (38 optimizer iterations, 42 trials). Their 91.712 s account for most of this 100.23 s command. All percentages below use the 91.712 s scoped positioning total.

| Subfunction/category | Sum | Share |
| --- | ---: | ---: |
| `mrisAsynchronousTimeStep` (collision/placement) | 43.501 s | 47.43% |
| MHT construction/free plus `MRISclearGradient` | 21.046 s | 22.95% |
| `MRIScomputeSSE` | 15.184 s | 16.56% |
| Other gradient terms | 4.426 s | 4.83% |
| `mrisComputeIntensityTerm` | 2.339 s | 2.55% |
| `MRIScomputeMetricProperties` | 1.554 s | 1.69% |
| `mrisComputeRepulsiveTerm` body | 0.532 s | 0.58% |
| RMS computation | 0.283 s | 0.31% |

The 42 timed collision/placement calls total 43.466 s internally: independent-subvolume pass 0 is 28.767 s (66.18%), cross-subvolume pass 1 is 13.603 s (31.30%), and bounds/face/vertex partitioning plus merge are 1.096 s (2.52%). The outer step timer differs by 0.034 s because it also includes setup outside the inner timed function.

## Optimization decision

The greatest *measured* CUDA opportunity is the 28.767 s pass-0 collision/placement body. However, `utils/mrisurf_timeStep.cpp:378-866` updates vertex positions and a shared MHT as it accepts/rejects moves; it explicitly describes collision-order constraints. A GPU parallel port cannot claim parity from a fast kernel alone. The next bounded trial would capture one iteration's input mesh, gradient, MHT occupancy, and per-vertex accepted movement; compare CPU and CUDA placement plus final ordered MHT buckets before replaying the whole command. Pass 1 remains sequential in the current partition scheme and contributes 13.603 s.

The more tractable first optimization is MHT construction: `utils/mrisurf_mri.cpp:590-600` rebuilds one vertex and **two face** tables on each relevant iteration, and `utils/mrisurf_sseTerms.cpp:2944-2950` builds additional vertex and face tables at a different resolution inside SSE. `utils/mrishash.cpp:1032-1120` captures faces/vertices serially. Profile the individual table creations and bucket order next, then test exact table reuse or deterministic parallel construction on the CPU before a CUDA transfer. This has a directly measured 21.046 s outer bound, plus an unmeasured subset of the 15.184 s SSE bound. The CUDA benchmark must include CPU↔GPU transfers and conversion back to the CPU MHT used by collision checks.

The repulsive-gradient body accounts for only 0.58% of positioning time, so accelerating it alone cannot improve the whole stage materially. The separate [intensity-term CUDA pilot](REPORT.md) also has a small end-to-end benefit. A previous `mris_sphere`/`MRISaverageGradients` pilot was slower as a complete stage and changed its surface coordinates.
