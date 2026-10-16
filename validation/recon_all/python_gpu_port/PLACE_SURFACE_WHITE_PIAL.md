# White and pial placement: isolated numerical port status

**Current result (2026-09-26):** On the fixed T1, both LH and RH pial placement independently compute all 41 optimizer decisions from Python SSE/RMS. The installed decision logs enter only as post-decision assertions. LH matches 14 RAM checkpoints; RH matches its available steps 1, 2 and 41 RAM checkpoints. The final optimizer states have 319,866/319,866 and 316,623/316,623 exact float32 coordinates. Medial-wall pinning and intersection repair produce final pial surfaces with 106,622/106,622 LH and 105,541/105,541 RH exact ordered vertices and exact faces. Per-vertex area, thickness, pial curvature, and volume have zero outliers under their stated tolerances. Both sides start from frozen official white surfaces and MRI inputs. Independent white placement, connected orchestration, and end-to-end recon-all remain open.

The earlier source-probe and first-difference sections below record the incremental diagnostic history. The bilateral installed-binary results near the end supersede their provisional acceptance statements.

## Fixed subject and source

- Subject: `fs_sub01` under `/data/workspaces/freesurfer_synth/work/reconall_main_20260924/single_subjects`.
- Installed native program: FreeSurfer 8.2.0-1 build `8.2.0-20260314-d932c45`, `/opt/reference/Freesurfer/8.2.0-1/bin/mris_place_surface`, SHA-256 `67139c3590f961f879a4214cd01f173b42986c7b178d1851dfa3f08e8acf12aa`.
- Source: commit `d932c45b7941662ea380a05efef580568b98d41a`. The source path is `mris_make_surfaces/mris_place_surface.cpp`: input and initial smoothing at lines 407–445, intensity preprocessing at 487–575, border values and the outer loop at 626–855. Supporting functions are `utils/mrisurf_metricProperties.cpp:1100` (`MRISaverageVertexPositions`), `utils/mrisurf_vals.cpp:2029` (`MRISaverageMarkedVals`), `utils/mrisurf_mri.cpp:1692` (`MRIScomputeBorderValues_new`), `utils/mrisurf_mri.cpp:448` (`MRISpositionSurface`), and `utils/mrisurf_timeStep.cpp:852` (`mrisAsynchronousTimeStep`).

The frozen recon-all log contains six full placement commands. All use `--threads 4`, `brain.finalsurfs.mgz`, `wm.mgz`, and `autodet.gw.stats.{hemi}.dat`:

| Stage | Input mesh | Other active inputs/options | Original accelerator-a-1 wall time |
| --- | --- | --- | ---: |
| `lh/rh.white.preaparc` | `{hemi}.orig` | `--white --seg aseg.presurf.mgz --nsmooth 5 --rip-bg-no-annot --rip-bg --rip-bg-lof --restore-255 --outvol mrisps.wpa.mgz` | 162.73 / 156.47 s |
| `lh/rh.white` | `{hemi}.white.preaparc` | `--white --seg aseg.presurf.mgz --nsmooth 0 --rip-label {hemi}.cortex.label --rip-surf {hemi}.white.preaparc --aparc {hemi}.aparc.annot --rip-bg --rip-bg-lof --restore-255 --outvol mrisps.white.mgz` | 143.68 / 153.95 s |
| `lh/rh.pial.T1` | `{hemi}.white` | `--pial --seg aseg.presurf.mgz --nsmooth 0 --rip-label {hemi}.cortex+hipamyg.label --pin-medial-wall {hemi}.cortex.label --aparc {hemi}.aparc.annot --repulse-surf {hemi}.white --white-surf {hemi}.white --restore-255` | 160.38 / 152.11 s |

The six logged calls total 929.32 s. The full recon-all log provides all absolute/relative paths and argument order at lines 2706, 3108, 4867, 5186, 5505, and 5820. No full-stage Python output is claimed.

## Reproducible checkpoints

The installed binary was rerun on control-host with the same fixed input files and stage flags, writing only to `/data/workspaces/freesurfer_synth/work/place_surface_python_20260925`. The diagnostics omit the original `--outvol` write and change only the output path plus the flags below:

1. `--n_averages -1 --subiters 0` skips border targeting and optimization. For `white.preaparc`, `--nsmooth 0` returns `lh.orig` with all coordinates and ordered faces unchanged. With the real `--nsmooth 5`, the native output is the pre-optimization smoothing checkpoint. Pial's real `--nsmooth 0` gives an unchanged `lh/rh.white` mesh at this checkpoint.
2. `--n_averages 1 --subiters 0 --target <path>` for white, or `--n_averages 2 --subiters 0 --target <path>` for pial, runs target search without positioning. Pial's minimum average count of 2 gives one outer pass; white's minimum of 0 gives two. The output mesh stays unchanged; `--target` exposes the per-vertex target coordinates.
3. The isolated [probe builder](build_place_surface_first_iteration_probe.py) recompiles only a copied `mris_place_surface.cpp` against the prebuilt pinned source libraries, setting `niterations=1` and `write_iterations=1`. `DIAG=0x8` writes `lh.probe0000` and `lh.probe0001`. The probe's `PLACE_VALUES_PREFIX` option writes native float32 values plus marked/ripped flags immediately before and after the first `MRISaverageMarkedVals` call. The installed binary cannot emit these optimizer or value snapshots; the probe is a source diagnostic, not the installed reference.

The probe source, object, executable, and all mesh/dump files remain outside Git. Its first snapshot and its no-optimization target match installed-binary checkpoints exactly, but **its first optimizer step is not certified against the installed binary**. Prior profiling showed that a full source-rebuilt call differs from the installed final mesh at 98,057/106,622 vertices (P99 0.0070 mm, maximum 0.6300 mm); compiler/toolchain differences prevent treating a source snapshot as an installed-binary snapshot.

## Exact kernel comparisons

The new module is [place_surface_smoothing.py](../../../src/fnit/recon_all/place_surface_smoothing.py). It constructs the source's ordered 1-ring neighbor list from face order and performs sequential float32 additions. `nibabel.freesurfer.read_geometry` yields float64 coordinates, so the explicit float32 cast is required for exact parity. On the fixed CPU runs:

| Kernel/checkpoint | Native comparison | Result |
| --- | --- | --- |
| Five passes of `MRISaverageVertexPositions`, `lh.orig` | Installed `lh.white.preaparc` pre-iteration mesh | 106,622/106,622 vertices, 213,240/213,240 ordered faces exact; max coordinate error 0 |
| Same, `rh.orig` | Installed right pre-iteration mesh | 105,541/105,541 vertices, 211,078/211,078 faces exact; max error 0 |
| Five passes of `MRISaverageMarkedVals`, left/right white and pial | Source probe before/after arrays; installed versus source no-optimization target also exact for checked left cases | 424,326/424,326 float32 values exact; four maximum errors 0 |

The [smoothing report](place_surface_smoothing_report.json), [marked-value report](place_surface_marked_values_report.json), and [checkpoint displacement report](place_surface_probe_report.json) contain counts and measured CPU times. The [mesh comparator](benchmark_place_surface_smoothing.py), [value comparator](benchmark_place_surface_marked_values.py), and [probe summarizer](summarize_place_surface_probe.py) regenerate them from the frozen files. The PyTorch API accepts a `device` argument, but CUDA numerical parity and runtime have **not** been measured.

## Border target and optimizer boundary

Compared with the pre-optimization mesh, the installed binary's target output differs at 100,960/106,622 vertices for left white (median displacement 0.35 mm, P99 3.15 mm, maximum 9.65 mm) and 100,881/106,622 for left pial (median 2.05 mm, P99 7.85 mm, maximum 9.65 mm). The right target differences are 99,882/105,541 white and 99,797/105,541 pial. The isolated port below computes the first border-search target only; the white call has a later outer pass, and none of the calls yet has a complete Python surface optimizer.

### First border-search pass, fixed preprocessed inputs

The [border-search module](../../../src/fnit/recon_all/place_surface_border.py) translates the pinned `MRIScomputeBorderValues_new` source-order MRI interpolation, derivative samples, sigma escalation, and peak selection into Python/Numba. The [source probe builder](build_place_surface_border_probe.py) captures `involCBV`, the float32 surface-to-voxel matrix, and per-vertex state immediately before/after the first C++ border-search call. The Python calculation uses those captured **inputs**, the frozen `aseg.presurf.mgz`, and the `autodet.gw.stats.{hemi}.dat` thresholds. It never reads the native target values during inference. The probe is diagnostic instrumentation and is not a runtime dependency.

| First-pass checkpoint | Vertices | Value, distance, gradient, mark, sigma | Ordered target xyz |
| --- | ---: | ---: | ---: |
| LH white | 106,622 | All exact | 319,866/319,866 exact |
| RH white | 105,541 | All exact | 316,623/316,623 exact |
| LH pial | 106,622 | All exact | 319,866/319,866 exact |
| RH pial | 105,541 | All exact | 316,623/316,623 exact |

Reports: [LH white](lh_border_full_report.json), [RH white](rh_border_full_report.json), [LH pial](lh_pial_border_full_report.json), [RH pial](rh_pial_border_full_report.json). Reproduce each comparison with [the validator](validate_place_surface_border.py) and `--require-exact`; all four exit successfully. The [installed-binary checkpoint report](border_probe_vs_installed.json) confirms that the source probe's complete no-optimizer `--target` surface equals the installed FreeSurfer binary's `--target` vertex coordinates and ordered faces for all four calls. It also checks the initial vertex positions against the installed pre-iteration mesh. This first-pass parity does not establish final `white`/`pial` mesh parity.

The fixed calls have `CBVfindFirstPeakD1=0`, `CBVfindFirstPeakD2=0`, no alternate border mask, no stop mask, and uint8 input MRI. Other settings and the later white target pass have not been validated. A separate [Python `mris_autodet_gwstats` stage](AUTODET_GWSTATS_STATUS.md) now regenerates the frozen threshold file byte for byte; this particular border validator still reads the frozen official threshold. The independently ported `mris_remesh` also remains native in the current full recon-all entry point. The border-search module currently accepts prepared inputs explicitly and uses Numba on CPU; no CUDA parity has been measured.

### Raw-input preparation checkpoint

The [MRI preparation](../../../src/fnit/recon_all/place_surface_volume.py), [surface-to-voxel geometry](../../../src/fnit/recon_all/place_surface_geometry.py), [initial normals](../../../src/fnit/recon_all/place_surface_normals.py), and [white/pial rip](../../../src/fnit/recon_all/place_surface_rip.py) modules now generate first-pass inputs from the original `brain.finalsurfs.mgz`, `wm.mgz`, `lh/rh.orig` or `lh/rh.white`, surface label, and header metadata. The [raw-input validator](validate_place_surface_inputs.py) independently regenerates all four inputs and their border targets, without using any diagnostic input in inference. It compares only the finished arrays with copied pinned-source checkpoints; `--require-exact` exits successfully.

| Input comparison per case | LH white | RH white | LH pial | RH pial |
| --- | ---: | ---: | ---: | ---: |
| Initial vertex xyz | 319,866/319,866 | 316,623/316,623 | 319,866/319,866 | 316,623/316,623 |
| Initial normal components | 319,866/319,866 | 316,623/316,623 | 319,866/319,866 | 316,623/316,623 |
| Prepared MRI voxels | 16,777,216/16,777,216 | 16,777,216/16,777,216 | 16,777,216/16,777,216 | 16,777,216/16,777,216 |
| Bright-label voxels | 16,777,216/16,777,216 | 16,777,216/16,777,216 | 16,777,216/16,777,216 | 16,777,216/16,777,216 |
| Surface RAS-to-voxel float32 entries | 16/16 | 16/16 | 16/16 | 16/16 |
| Ripped flags | 106,622/106,622 | 105,541/105,541 | 106,622/106,622 | 105,541/105,541 |
| Initial target intensity | 106,622/106,622 | 105,541/105,541 | 106,622/106,622 | 105,541/105,541 |
| Border target xyz | 319,866/319,866 | 316,623/316,623 | 319,866/319,866 | 316,623/316,623 |

All entries are exact, maximum absolute error zero. Border value, distance, gradient, marked flag, and sigma also match every vertex in all four cases; see the [machine-readable raw-input first-pass report](place_surface_raw_first_pass_report.json). The separate [two-pass LH](lh_white_rip_report.json) and [RH](rh_white_rip_report.json) reports compare both white rip and pre-search intensity arrays with pinned-source diagnostics. Combined read, smoothing, ripping, Numba search, and comparison took 30.3/28.5 s for left/right white and 12.4/13.6 s for left/right pial, including JIT warmup; these timings are not paired against native stage timings. These are source-probe intermediate comparisons, while the previously checked full no-optimizer target matches the installed binary. The later white outer pass and full optimizer remain incomplete.

### Installed-machine-code first optimizer checkpoint

The installed binary is statically linked, stripped, and has no CLI option to set `parms.niterations`. A fixed LH pial diagnostic verified its SHA-256 above and used GDB on the child process to set the already-initialized `parms.niterations` from 100 to 1 at address `0x416970` (`$rbx+0x514`). The ordinary `--no-pin-medial-wall` option removed only post-optimization pinning; a second GDB breakpoint skipped the final `MRISremoveIntersections` call. This changed the child's RAM, never the installed executable on disk. The one-step mesh was the same, all 106,622 vertices and 213,240 ordered faces, whether final intersection removal was skipped or run in this fixed case.

The copied-source [optimizer gradient probe](build_place_surface_optimizer_probe.py) exposes immediate first-step terms without changing the installed binary. The isolated [intensity](../../../src/fnit/recon_all/place_surface_intensity.py), [surface repulsion](../../../src/fnit/recon_all/place_surface_repulsion.py), [signed gradient averaging](../../../src/fnit/recon_all/place_surface_gradient_average.py), and [normal/tangential spring](../../../src/fnit/recon_all/place_surface_spring.py) kernels each match their immediate LH pial source checkpoint exactly: 319,866/319,866 float32 gradient components per term or averaging pass. Intensity is regenerated from original MRI, WM, surface and frozen autodet thresholds; its placement MRI matches 16,777,216/16,777,216 voxels and vertex value/sigma each matches 106,622/106,622. Original-surface repulsion also regenerates all 178,156 spatial hash candidate IDs exactly. The [intensity](place_surface_intensity_report.json), [repulsion](place_surface_repulsion_report.json), [averaging](place_surface_gradient_average_report.json), and [spring](place_surface_spring_report.json) reports contain the exact counts. These are **copied-source internal checkpoints**, not gradient arrays emitted by the installed binary.

The [quadratic-curvature input port](../../../src/fnit/recon_all/place_surface_curvature.py) independently reproduces 639,732/639,732 tangent-basis elements and 1,965,674/1,965,674 ordered two-ring neighbor IDs exactly. Its VNL-order float32 QR inverse also reproduces the quadratic-curvature scalar for all 106,622 vertices exactly. The f2c `snrm2` routine evaluates the final square root in float64 before casting back to float32; reproducing that order removes the earlier 3.82e-4 maximum error. The [curvature report](place_surface_curvature_report.json) records the complete exact comparison. From the raw MRI, WM, aseg, LH white mesh, label, and frozen autodet thresholds, the [first-gradient validator](validate_place_surface_first_gradient.py) independently regenerates all six consecutive gradient checkpoints; each has 319,866/319,866 float32 elements exactly equal to the copied-source probe ([report](place_surface_first_gradient_report.json)). This is a copied-source internal comparison, not an installed-binary gradient measurement. From the source's **final** gradient, the [unconstrained step candidate](../../../src/fnit/recon_all/place_surface_step.py) gives 106,447/106,622 exact first-step vertices; 175 are held by collision checks. The [ordered asynchronous collision step](../../../src/fnit/recon_all/place_surface_collision.py) now reproduces those holds and matches the copied-source `lh.probe0001` at all 106,622 vertices and 213,240 faces. See the [first-step report](place_surface_first_step_report.json). The following section checks the second step separately; neither comparison establishes the full-stage surface output.

The [one-step comparator](compare_place_surface_installed_step.py) and [report](place_surface_installed_one_step_report.json) compare this installed-machine-code one-step mesh with the recompiled pinned-source `lh.probe0001`: 105,195/106,622 vertex coordinates are bitwise exact, P99 distance `2.98e-8` mm and maximum `7.63e-6` mm; ordered faces are exact. Compared with the source pre-step mesh, the installed mesh moves 100,825 vertices with median displacement `0.3000` mm. The native log reports `0.08` min internally for this one-iteration diagnostic. It is an installed-binary **RAM diagnostic**, not a normal FreeSurfer CLI output. The Python first-step result equals this copied-source mesh, and therefore has the same 105,195/106,622 exact vertices against the installed-binary RAM diagnostic. The remaining installed-versus-source difference is at most 7.63e-6 mm at this one-step checkpoint.

### Three-step objective and acceptance

The [placement objective](../../../src/fnit/recon_all/place_surface_objective.py) samples the prepared placement MRI at eligible vertices, sums the tangential-spring energy, and computes float32 surface area from the ordered triangles. The fixed pial call weights intensity by 0.2 and tangential spring by 0.3; the pinned source's quadratic-curvature SSE routine returns zero even though its gradient term is active. The [validator](validate_place_surface_objective.py) regenerates these inputs from the original T1, WM, aseg, LH white mesh, label, and frozen autodet thresholds. Its target array matches the source probe exactly. The [full-precision copied-source probe](build_place_surface_objective_probe.py) prints initial and three successive RMS, total-area, and full-SSE values from the three-iteration command; [runner](run_place_surface_objective_probe.py), [initial two-step report](place_surface_objective_report.json), and [continuous three-step report](place_surface_three_steps_report.json) preserve the comparisons.

| LH pial checkpoint | Python / source total area | Python / source RMS | Python complete SSE | Source complete SSE |
| --- | ---: | ---: | ---: | ---: |
| Initial | 75,302.5 / 75,302.5 | 39.05922669761335 / 39.05922669761335 | 30,943,635.20565021 | 30,943,635.20565021 |
| First accepted step | 76,115.171875 / 76,115.171875 | 32.40564155460752 / 32.40564155460752 | 21,350,287.220679563 | 21,350,287.22067956 |
| Second accepted step | 76,982.5859375 / 76,982.5859375 | 26.553467457855252 / 26.553467457855252 | 14,390,833.45159803 | 14,390,833.45159803 |
| Third accepted step | 77,900.28125 / 77,900.28125 | 21.77106790773631 / 21.77106790773631 | 9,731,189.165331032 | 9,731,189.1653310321 |

All four checks use 100,881 eligible vertices. The maximum complete-SSE absolute difference is 3.73×10⁻⁹; all four float32 surface areas are exact. The fixed source acceptance rule accepts all three steps from the RMS decrease and complete SSE decrease. The continuous three-step validator passed --require-exact and reported no first failure. Later accept/retry iterations and final placed surfaces remain open. The earlier installed-binary RAM one-step geometry comparison still applies, with its 7.63×10⁻⁶ mm maximum source-versus-installed difference. The copied-source objective probe does not establish complete-stage output parity or speed.

### Consecutive pial gradient and collision checkpoints

A copied-source two-iteration diagnostic captures the second LH pial gradient after the first accepted collision-constrained step. The [second-gradient validator](validate_place_surface_second_gradient.py) starts from the same first-step surface and independently regenerates the MRI preparation, fixed target intensity, and six gradient terms from original subject inputs. The [report](place_surface_second_gradient_report.json) passed `--require-exact`:

| Checkpoint | Python versus copied source |
| --- | ---: |
| Second-step starting xyz | 319,866/319,866 float32 components exact |
| Active-vertex normals | 303,000/303,000 components exact |
| Ripped flags | 106,622/106,622 exact |
| Intensity, original-surface repulsion, signed averaging, normal spring, quadratic curvature, tangential spring | Six checks, each 319,866/319,866 gradient components exact |

On 354 ripped vertices, 1,062 normal components differ because the native code preserves stale normals there; none contributes to these gradients. The first collision step marks 175 blocked vertices as `cropped`. The source skips them in the following original-surface repulsion term, so the Python port now carries that collision state into its optional `cropped` mask. Before that change, repulsion disagreed at exactly 175 vertices and the error spread during gradient averaging. The validator derives the cropped mask from the source-validated first gradient and exact first-step mesh; a complete native-free optimizer would carry it directly from its own collision step. The [second-step validator](validate_place_surface_second_step.py) then applies the native-order asynchronous collision constraint to that gradient. It passed the copied-source two-iteration diagnostic at all 106,622/106,622 ordered vertex coordinates and 213,240/213,240 ordered faces, with 182 blocked vertices ([report](place_surface_second_step_report.json)). This establishes the second accepted coordinate snapshot. The full objective SSE and acceptance of these steps also match as shown above. The [continuous three-step validator](validate_place_surface_three_steps.py) separately starts from the original MRI and LH white mesh, carries its own cropped mask from each Python collision step, and checks six ordered gradient arrays plus the next mesh after every step. Its [report](place_surface_three_steps_report.json) shows 319,866/319,866 exact float32 gradient components at each of six checkpoints for each of the three steps; the third step also matches 319,866/319,866 ordered coordinate components and 213,240/213,240 ordered faces. The Python third collision result infers 406 blocked vertices from proposals that stayed fixed. Its coordinates have the same SHA-256 as the copied-source third snapshot. Later step retries and the final placed surface remain unverified.

The four Python calls took 4.60–6.00 s in separate control-host processes with cached JIT compilation, excluding file reads. Pinned source-probe logs report 0.0204–0.0228 min (1.22–1.37 s) for the C++ border-search call with four threads. These are descriptive, unpaired measurements under changing load and do not form a speed comparison. Output parity took priority in this port.

With frozen inputs, the source probe's iteration 0 equals the installed pre-iteration mesh. Its first accepted left white optimizer step moves 101,004 vertices (median 0.300000 mm, maximum 0.300004 mm); left pial moves 100,825 (median 0.300000 mm, maximum 0.300004 mm). At that historical source-probe stage, no complete Python optimizer output existed. The first LH pial collision-constrained step takes 62.09 s including initial Numba JIT and excluding file I/O; the second takes 68.38 s excluding file I/O. The independent three-step run took 265.54 s including input preparation, file I/O, JIT and all comparisons, with each iteration taking 79.71–81.85 s under the then-current control-host load; the earlier installed one-step RAM diagnostic reported 0.08 min internally. These are not paired timings. The frozen installed LH pial log ran 41 numbered optimizer steps across four outer passes before final intersection handling. At roughly 18 s per optimized Python step under the observed load, a direct loop would take about 12 minutes for those 41 steps alone, compared with about 55 minutes for the original Python path; these are extrapolations, not paired benchmarks. The later installed-binary section completes fixed-subject pial geometry with the native accept/retry schedule supplied as diagnostic input; independent scheduling remains open. The [earlier CPU profile](../native_cuda_pilot/CPU_PROFILE.md) places the largest measured time fractions in collision/placement (47.43%), hash-table setup (22.95%), and SSE (16.56%) for its separate clean-source replay.

### Source-order CPU hot-path optimization

The [gradient profile](place_surface_gradient_profile.json) measured the validated third-step gradient and objective twice in one process. Four calls rebuilt the same ordered one-ring table for signed averaging, normal spring, tangential spring, and tangential-spring SSE, each taking about 3.3–3.6 s. The gradient and [objective kernels](../../../src/fnit/recon_all/place_surface_objective.py) now accept one reusable table; [curvature](../../../src/fnit/recon_all/place_surface_curvature.py) can use it for the two-ring construction as well. The [cached profile](place_surface_gradient_profile_cached.json) still matches every third-step gradient element and reduces the repeated-call gradient/objective time from 14.83 to 1.40 s; its first call took 3.62 s with new Numba compilation.

The [collision profile](place_surface_collision_profile.json) found about 604,419 KDTree queries and 6.85 million small NumPy reductions per step. The [collision module](../../../src/fnit/recon_all/place_surface_collision.py) now moves each proposed triangle's geometry and candidate-face filtering into Numba while keeping the vertex, subvolume, and KDTree candidate order. The exact-checked Numba path is the default; the original path remains available with `fast=False` for comparison. The [optimized profile](place_surface_collision_profile_fast.json) matches all 106,622 third-step vertices exactly and takes 27.42 s on its first call including JIT, then 17.87 s on the next call, versus 80.55/79.94 s for the original path. KDTree queries remain the largest sampled cost (7.95 s in the warm optimized call).

The [optimized continuous report](place_surface_three_steps_fast_report.json) passed all three source-level steps again: each six-stage gradient and ordered mesh remains exact, as do area, RMS, and complete SSE. The total three-step runtime fell from 265.54 to 73.63 s, and the optimized iterations took 18.16, 17.62, and 18.00 s. These were separate control-host runs with changing load, so the ratio is descriptive, not a controlled native-versus-Python benchmark. All optimized kernels here run on CPU with Numba; no CUDA placement parity is claimed.

### Ten consecutive diagnostic pial checkpoints

The [ten-step validator](validate_place_surface_three_steps.py) independently replays the fixed LH pial call from the original T1, WM, aseg, LH white mesh, label and frozen autodet thresholds. This **diagnostic command** explicitly sets `--n_averages 2`, with sigma 2 and five target-value averages. Its command, input mesh SHA-256, pinned original and patched source SHA-256, and per-step native objectives are embedded in the [machine report](place_surface_ten_steps_projection_report.json). The frozen official `recon-all` call starts its first pial pass at `n_averages=16`; that pass has a separate validation target.

For each of the ten diagnostic iterations, the six consecutive gradients match 319,866/319,866 float32 components; source-order asynchronous coordinates and the written mesh each match 319,866/319,866 components; the 106,622 native cropped counters and 213,240 ordered faces are exact. Float32 areas and RMS values match, and the maximum complete SSE difference is 3.73e-09. The first ten-step attempt identified one step-seven vertex with a 0.000143 mm coordinate difference ([initial failure report](place_surface_ten_steps_report.json)). FreeSurfer projects displacement away from a neighbor within 0.01 mm before collision testing; [the focused diagnostic](place_surface_step_seven_projection_report.json) confirms that the new ordered projection restores all 319,866 coordinates exactly. The continuous rerun reports `first_failure=null`.

| Diagnostic step | Nonzero cropped counters | Native/Python area (mm²) | Native/Python RMS | Native complete SSE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 175 | 76,115.171875 | 32.405641555 | 21,350,287.220680 |
| 2 | 182 | 76,982.585938 | 26.553467458 | 14,390,833.451598 |
| 3 | 406 | 77,900.281250 | 21.771067908 | 9,731,189.165331 |
| 4 | 1,197 | 79,040.992188 | 18.061530786 | 6,754,050.150879 |
| 5 | 3,443 | 81,664.484375 | 15.240500844 | 4,861,889.091680 |
| 6 | 6,844 | 84,988.742188 | 12.931391911 | 3,552,042.454582 |
| 7 | 9,949 | 88,556.890625 | 11.039023635 | 2,639,215.653879 |
| 8 | 12,312 | 91,772.203125 | 9.507565615 | 2,006,400.275625 |
| 9 | 13,977 | 94,623.820312 | 8.366564225 | 1,596,563.439044 |
| 10 | 15,060 | 96,667.445312 | 7.543103923 | 1,333,514.958896 |

The continuous Python replay took 257.41 s including input reads, JIT, and all checks. The instrumented native command logged 0.74 min for its full ten-step call including later pinning/intersection handling and diagnostic writes. These are descriptive runs with different work and load.

## Official-parameter LH pial control and current acceptance boundary

The [provenance report](place_surface_installed_source_provenance.json) records the complete argv, pinned source commit, build and source file hashes, installed and copied-source binary hashes, compiler flags, and ten scientific input-file hashes. The paired current runs use the same frozen LH white input (SHA-256 `9c88a786c4a571c07d830b83fe647461b09fdacb63ff143670771340ef50b51c`), `--threads 4`, first outer pass `n_averages=16`, sigma 2, and five target-value averages. Both initial printed objectives are SSE 30,943,636 and RMS 39.059. The original `recon-all.log` places **LH** pial at line 5503 with its initial objective at line 5608, also SSE 30,943,636. The 28,453,128 initial SSE at line 5923 belongs to **RH** pial, whose stage starts at line 5818; it was previously misassociated with LH. The current installed normal CLI mesh matches the original `lh.pial.T1` at all 106,622 ordered vertices and all 213,240 ordered faces, although the two surface file hashes differ.

Four output classes are kept separate:

| Output class | Run and scope | Result |
| --- | --- | --- |
| Installed 8.2.0-1 normal CLI | Current frozen input, all four outer passes, pinning and intersection cleanup | 41 accepted steps; `lh.pial.objective` is the paired installed geometry baseline and matches the historical LH mesh vertex for vertex |
| Copied-source pristine normal CLI | Unmodified pinned source and clean built libraries, all four passes and cleanup | 42 accepted steps; final output differs from the installed baseline |
| Installed binary RAM checkpoint | GDB stops immediately after the first outer optimizer and skips later passes, pinning and cleanup | Source comparison at accepted step 1 and natural first-pass end, step 26; not a final surface |
| Copied-source instrumented diagnostic | Same `n_averages=16`/sigma 2, writes gradients, objectives, meshes and next-pass target state | 26 accepted first-pass steps; its printed step 0–26 SSE/RMS/dt trajectory matches the pristine copied-source CLI, but it is not the installed binary |

The [first-pass objective trajectory report](place_surface_official_trajectories_report.json) shows identical `dt` decisions in all three native executions. Each rejects a trial at step 21 and retries at `dt=0.25`; steps 24 and 26 halve the next `dt`. The first source-versus-installed printed SSE difference above 1.0 is step 14, above 10 is step 18, above 100 is step 24, and step 26 differs by 412.2. The source build uses Anaconda GCC 11.2, `-O3`, `-march=nocona`, `-mtune=haswell`, and conda `libgomp`; the installed stripped binary records GCC 4.8.5 and uses system `libgomp`. These toolchain differences are a plausible cause of the first few float32 differences, but no causal compiler ablation has been performed.

| Paired geometry | Exact vertices | Median / P99 / max vertex distance | Vertices >0.1 mm |
| --- | ---: | ---: | ---: |
| Installed RAM versus instrumented source, [step 1](place_surface_official_first_step_mesh_comparison.json) | 105,295 / 106,622 | 0 / 0.000000030 / 0.00000763 mm | 0 |
| Installed RAM versus instrumented source, [step 26](place_surface_official_pass0_mesh_comparison.json) | 13,074 / 106,622 | 0.00000384 / 0.0372 / 0.664 mm | 265 |
| Installed normal CLI versus pristine copied-source [normal CLI](place_surface_official_final_cli_comparison.json) | 8,279 / 106,622 | 0.0158 / 0.229 / 1.669 mm | 5,571 |

### Installed first-pass divergence, selected large-outlier vertices

The intermediate [step 12](place_surface_official_step12_mesh_comparison.json), [step 13](place_surface_official_step13_mesh_comparison.json), and [step 14](place_surface_official_step14_mesh_comparison.json) meshes locate three selected vertices whose installed-versus-copied-source distance exceeds 0.1 mm at step 14. These are **not** the first globally differing vertices: step 10 already has three vertices above 1e−4 mm. The first crossing of 1e−4 mm for each selected vertex is step 13 for v53114 and v55276, and step 14 for v81282. The [source-order collision traces](place_surface_official_step13_branch_report.json) and [step-14 trace](place_surface_official_step14_branch_report.json) reproduce the copied-source accepted coordinates at all three vertices; they do not establish installed collision internals.

The [installed-binary debug report](place_surface_installed_debug_first_branch_report.json) captures `--debug-vertex` pre-move aggregate gradients in RAM-limited installed runs. Each debug run has exactly the same 106,622 ordered coordinates, 213,240 ordered faces, and accepted objective lines as its corresponding no-debug installed RAM checkpoint. The surface files have different metadata bytes, so the raw SHA-256 values differ. The native `%g` text limits gradient precision to roughly six significant digits.

| Vertex / first >1e−4 step | Pre-step mesh distance | Copied-source final gradient | Installed pre-move gradient | Accepted move and copied-source collision result |
| --- | ---: | --- | --- | --- |
| v53114 / 13 | 4.77e−7 mm | [−0.163136, +0.193660, −0.032630] | [−0.163972, +0.194520, −0.033237] | Source rejects candidate at moving face 123585 / other face 123550; installed moves 0.1283 mm away from source |
| v55276 / 13 | 3.00e−5 mm | [−0.107621, +0.256920, +0.234681] | [−0.112047, +0.256725, +0.225397] | Both move; post-step distance 0.00513 mm |
| v81282 / 14 | 4.26e−6 mm | [−0.030246, −0.075789, −0.006323] | [−0.0197776, +0.201596, +0.0666822] | Both move; installed y displacement +0.100800 mm agrees with `dt=0.5` times its pre-move y gradient; post-step distance 0.14351 mm |

For v81282, every source-versus-installed step-13 coordinate within its 16-ring (1,004 vertices) differs by at most 1.91e−5 mm; the center pre-step normal from the independent Python installed mesh matches the installed native debug normal to its printed precision. The [force-term and 16-pass trace](place_surface_installed_step14_force_terms_report.json), generated by [the isolated validator](diagnose_place_surface_installed_force_terms.py), uses the installed step-13 RAM mesh with the original MRI, frozen target values, and copied-source cropped counters. Its final source-order Python gradient is `[−0.019777633, +0.201596007, +0.066682197]`, agreeing with all three installed printed gradient components within 3.33e−8. The source-mesh Python 16-pass averaging array matches every copied-source checkpoint component exactly; the installed-mesh repeated single-pass array matches the direct 16-pass array exactly. The installed debug log also prints its intensity term `(−0.000, 0.000, 0.000)`, post-averaging vector `(−0.020, +0.066, +0.020)`, and curvature term `(−0.022, +0.089, +0.023)` at three decimal places, consistent with the Python replay. The installed binary's cropped counters and exact individual force-term floats were not captured; normal and tangential spring increments are not separately printed.

The first source-order signed-neighbor decision differs in averaging pass 1. At v81282, the pre-average y component is −1.22781e−5 on the copied-source mesh and +2.20756e−5 on the installed mesh. Its first ordered neighbor, v81291, has dot product +2.27176e−6 and is included for the source mesh, but −4.08453e−6 and is excluded for the installed mesh; that neighbor's y input is about −0.170340. Ordered neighbor v80901 at rank 3 is excluded on the source mesh but included on the installed mesh, contributing about +0.097534 in y. Pass 1 therefore produces y=−0.206952 versus +0.0487778; pass 16 produces −0.211253 versus +0.0661283. The remaining normal-spring, curvature and tangential-spring increments at that vertex are close across meshes. These near-zero signed-dot decisions explain the observed step-14 amplification in the independent replay; they do not prove which earlier installed/source floating-point operation first changed the mesh. This copied-source comparison did not establish installed-binary parity; the independent replay and metric results appear below.

The [early 16-ring backtrace](place_surface_early_step_16ring_report.json) checks the same fixed v81282 support at installed/source steps 1, 5, 10, 12, and 13. At step 1 the center is still exact, while four of its 1,004 16-ring vertices differ by 1–2 float32 coordinate ULPs (maximum 3.8147e−6 mm). The nearest, v80852, is ten mesh edges away. Its [installed first-step printf-register capture](place_surface_installed_step1_exact_gradient_report.json) yields all three promoted float32 gradient bits exactly equal to the copied-source checkpoint: `[−3.140500784, +1.844934344, +4.697066784]`. The independent source-order clipped candidate equals the copied-source after-collision coordinate, but the installed accepted y coordinate is 3.8147e−6 mm lower. That earlier capture placed the difference after the matched gradient; the follow-up below now captures installed `odxyz` and resolves this selected first-step displacement. Ring membership remains a possible route to the later signed average, not a proven vertex-to-vertex causal chain. The [consolidated first-difference report](place_surface_installed_first_difference_report.json) preserves the earlier unresolved boundary and step-14 branch as historical evidence.

The capture is reproducible on control-host in the isolated `W=/data/workspaces/freesurfer_synth/work/place_surface_python_20260925` workspace, using the unchanged frozen subject and existing license **path**; the scripts never read or copy license contents. Run [the RAM capture](capture_place_surface_installed_ram.py) with `--installed-log "$W/probe_installed_official_run/objective_probe.log" --out "$W/probe_installed_debug_v81282_step14" --license /data/workspaces/freesurfer_synth/work/reconall_external_models_20260925/private_license.txt --first-pass-iterations 14 --debug-vertex 81282`, then [the debug comparison](summarize_place_surface_installed_debug.py) with `--root "$W" --report "$W/place_surface_installed_debug_first_branch_report.json"`. The force-term validator takes `--subject /data/workspaces/freesurfer_synth/work/reconall_main_20260924/single_subjects/fs_sub01 --module "$W" --source-run "$W/probe_official_pass0_run" --installed-previous "$W/probe_installed_ram_official_step13/surf/lh.pial.ram" --native-debug-report "$W/place_surface_installed_debug_first_branch_report.json" --vertex 81282 --step 14 --trace-averaging --report "$W/place_surface_installed_step14_force_terms_report.json"`.

The earlier selected-vertex check can be reproduced from those frozen checkpoints on control-host with:

```bash
W=/data/workspaces/freesurfer_synth/work/place_surface_python_20260925
L=/data/workspaces/freesurfer_synth/work/reconall_external_models_20260925/private_license.txt
python "$W/trace_place_surface_early_ring.py" --root "$W" --vertex 81282 --radius 16 --report "$W/place_surface_early_step_16ring_report.json"
python "$W/capture_place_surface_installed_ram.py" --installed-log "$W/probe_installed_official_run/objective_probe.log" --out "$W/probe_installed_debug_v80852_step1" --license "$L" --first-pass-iterations 1 --debug-vertex 80852
python "$W/capture_place_surface_exact_debug_gradient.py" --prior-debug-report "$W/probe_installed_debug_v80852_step1/installed_ram_report.json" --out "$W/probe_installed_exact_v80852_step1" --license "$L"
python "$W/summarize_place_surface_installed_first_difference.py" --root "$W" --module "$W" --report "$W/place_surface_installed_first_difference_report.json"
```

### Installed first-step clipped displacement, 2026-09-26

The pinned `mrisAsynchronousTimeStep_update_odxyz` source evaluates the sum of three float32 squared offsets, promotes that sum to double for `sqrt`, divides the float32 `max_mag` by the double result, and stores each scaled offset as float32. An isolated GDB read of the **installed** binary at v80852 captured sum `8.8322410583496094`, double square root `2.9719086557883316`, `max_mag=0.30000001192092896`, raw y offset `0x3f6c26cf`, and clipped y offset `0x3dbeb4ff`. The previous Numba step used a float32 square root, giving `0x3dbeb500` and accepted y `0x4225c12e`. Promoting the squared sum explicitly to float64 before `sqrt` gives installed clipped y `0x3dbeb4ff` and accepted y `0x4225c12d`. The native GDB log is saved at `W/pial_installed_v80852_clip_registers.log` (SHA-256 `c2a498f8bf9b2daeca1c8b249e05564bb81376a42cab97368c86486bc1d5b040`); [the targeted bit regression](../../../tests/recon_all/test_place_surface_step.py) passed 1/1 on control-host.

With the unchanged frozen copied-source first-step gradients, the corrected [step kernel](../../../src/fnit/recon_all/place_surface_step.py) and Python asynchronous collision replay match the installed first-step LH pial RAM surface at **106,622/106,622 ordered vertices, all three float32 coordinate bit patterns**; 213,240 ordered faces also match. The four originally differing vertices in v81282's 16-ring, including v80852, now all match. Before collision handling, 106,439/106,622 candidate vertices equal the installed final coordinates; after handling, all do. The same replay matches only 105,295/106,622 vertices in the GCC11 copied-source checkpoint, so that private build is not the installed baseline. The [focused validator](validate_place_surface_installed_first_step_clip.py) and [structured report](place_surface_installed_first_step_clip_report.json) record input/binary SHA-256 values and selected vertex bits. Only v80852's gradient was independently confirmed bitwise equal to the installed gradient; the full-mesh result is conditional on the frozen copied-source gradient input.

The step-14 signed-neighbor branch remains a later-stage acceptance boundary. On the two frozen step-13 meshes, v81282's first signed-average pass includes v81291 for copied source (`dot=+2.27176e−6`) but excludes it for installed (`dot=−4.08453e−6`); v80901 has the opposite inclusion. This identifies the first nonlinear amplification at step 14, while that early diagnostic had not yet replayed steps 2–13 against installed RAM. The later bilateral replay and per-vertex metric results supersede this provisional boundary; white placement remains separate.

### Installed bilateral pial steps 1–2, continuous replay

The [bilateral validator](validate_place_surface_installed_second_step.py) now starts separately from each hemisphere's original `white` surface, `brain.finalsurfs.mgz`, `wm.mgz`, `aseg.presurf.mgz`, cortex/hippocampus/amygdala label, and frozen autodet stats. It recomputes the target, six gradient terms, float32/double clipping, and ordered asynchronous collision at each step. The second step takes the **Python first-step output**, including its blocked-vertex counters; installed RAM coordinates and copied-source gradients are used only for comparisons. The installed reference is the unchanged FreeSurfer 8.2.0-1 binary, captured in separate `niter=1` and `niter=2` GDB runs that copy the white input into isolated directories and stop after the first outer pass. Both accepted steps use `dt=0.5`.

| Hemisphere | Step 1 ordered vertices / components exact | Step 2 ordered vertices / components exact | Blocked after steps 1 / 2 |
| --- | ---: | ---: | ---: |
| LH | 106,622/106,622; 319,866/319,866 | 106,622/106,622; 319,866/319,866 | 183 / 82 |
| RH | 105,541/105,541; 316,623/316,623 | 105,541/105,541; 316,623/316,623 | 195 / 108 |

Every comparison uses `uint32` coordinate bit patterns, and both hemispheres preserve the input's ordered faces. The [LH](place_surface_installed_lh_two_steps_report.json) and [RH](place_surface_installed_rh_two_steps_report.json) reports record SHA-256 of the installed binary, each input, and each RAM checkpoint. The LH second-step intensity gradient differs from the GCC11 copied-source gradient at 6,540 vertices because that source run starts from a different first-step mesh; this is diagnostic only. These two accepted optimizer iterations are validated against the installed binary. Bilateral later-pass results, the repaired step-37 branch, final cleanup, and per-vertex metrics are detailed below; white optimization remains open.

Reproduce either side with `PYTHONPATH="$W" python "$W/validate_place_surface_installed_second_step.py" --subject "$S" --root "$W" --module "$W" --hemisphere lh --installed-binary /opt/reference/Freesurfer/8.2.0-1/bin/mris_place_surface --report "$W/place_surface_installed_lh_two_steps_report.json" --require-exact`, changing `lh` to `rh` for the other side. Here `S=/data/workspaces/freesurfer_synth/work/reconall_main_20260924/single_subjects/fs_sub01`, and `python` is the repository virtual-environment interpreter.

### Installed LH pial four-pass replay and final surface, 2026-09-26

The independent [replay validator](validate_place_surface_installed_second_step.py) starts from the frozen official white surface, MRI volumes, cortex label and autodet thresholds. It computes target values, all six gradient terms, source-order clipping and collision updates from its own state. The installed CLI log supplies accepted `dt` and reject/retry decisions as **diagnostic controls**. Native RAM meshes and copied-source gradients are comparison inputs only. The installed binary is FreeSurfer 8.2.0-1, SHA-256 `67139c3590f961f879a4214cd01f173b42986c7b178d1851dfa3f08e8acf12aa`; input and checkpoint hashes are in the linked reports.

| Outer pass | Accepted steps | Border sigma / gradient averages | Installed endpoint exact float32 coordinates |
| --- | ---: | ---: | ---: |
| 0 | 1–26 | 2 / 16 | 319,866/319,866 at [step 26](place_surface_installed_lh_first_outer_26_report.json) |
| 1 | 27–32 | 1 / 8 | 319,866/319,866 at [step 32](place_surface_installed_lh_outer1_32_report.json) |
| 2 | 33–36 | 0.5 / 4 | 319,866/319,866 at [step 36](place_surface_installed_lh_outer2_36_report.json) |
| 3 | 37–41 | 0.25 / 2 | 319,866/319,866 at [step 37](place_surface_installed_lh_step37_stale_mht_report.json) and [step 41](place_surface_installed_lh_outer3_41_stale_mht_report.json) |

The first fourth-pass step rejects a `dt=0.5` trial and retries at `dt=0.25`. The [earlier failing diagnostic](place_surface_installed_lh_step37_trace_report.json) rebuilt a face hash table for the retry and first differed at v27154. The installed source restores coordinates after rejection but retains the rejected trial's face hash table. The [face-pair and voxel trace](place_surface_installed_lh_step37_collision_branch.json) isolates the affected query: the rejected-trial bucket lacks the other face that a fresh-coordinate hash would contain. The collision replay now carries that source-order stale hash state, restoring all step-37 and step-41 coordinate bits. The fixed-case candidate broadphase still uses a KDTree; arbitrary subject geometry has not been validated. The installed pass-3 log prints sigma as `0.2` through a one-decimal format; the actual source parameter is 0.25.

The [final boundary and vertex-metric report](place_surface_installed_lh_final_boundary_with_metrics_report.json) compares the independent step-41 mesh with the installed optimizer RAM: 106,622/106,622 ordered vertices and 319,866/319,866 float32 coordinate components exact. Medial-wall pinning to the official white surface makes 106,616 final vertices exact. Two remaining intersecting faces mark six vertices; one source-order 100-pass soap-bubble repair removes those intersections and makes all 106,622 final vertices exact. The normal installed CLI, archived `lh.pial.T1`, and archived `lh.pial` have identical ordered coordinates and faces. Their file SHA-256 values may differ because surface metadata differs.

| Map from independently reconstructed final `lh.pial` and frozen official `lh.white` | Maximum absolute difference | Outliers |
| --- | ---: | ---: |
| `lh.area.pial` | 9.54e−7 mm² | 0 at 0.001 + 0.001 × reference |
| `lh.thickness` | 4.77e−7 mm | 0 at 0.005 + 0.001 × reference |
| `lh.curv.pial` | 4.54e−4 | 0 at 0.005 + 0.001 × reference |
| `lh.volume` | 1.91e−6 mm³ | 0 at 0.005 + 0.001 × reference |

Each row is a per-vertex comparison to the installed archived morphometry file; exact bit counts and SHA-256 values are in the report. The standalone metric implementations run on CPU for this paired check. The final Python surface has been written to an isolated directory and read back with all 319,866 coordinate bits and ordered faces unchanged.

The [RH installed normal CLI control](place_surface_installed_rh_normal_cli_report.json) ran in 83.76 s from the frozen `rh.white` input and produced 105,541/105,541 final vertices, all 316,623 coordinate components and ordered faces exact against archived `rh.pial`. Its accepted outer-pass endpoints are 26/31/35/41. The independent [RH four-pass replay](place_surface_installed_rh_full_replay_report.json) carries its own coordinates and cropped counters through all 41 steps, using only the native dt/reject schedule as a diagnostic control. Steps 1–2 and the step-41 optimizer mesh match installed RAM at all 316,623 float32 coordinate bits. Intermediate RH steps have not all been captured from installed RAM.

The [RH final boundary and vertex-metric report](place_surface_installed_rh_final_boundary_with_metrics_report.json) shows that the independent step-41 mesh matches installed optimizer RAM exactly. Medial-wall pinning leaves 45 final vertices differing. The installed CLI reports 25→9→0 intersecting faces; source-order Python smoothing takes the same two 100-pass cycles and then matches all 105,541 final ordered vertices, 316,623 coordinate components, and faces exactly. The independently written final `rh.pial` also survives readback bitwise. Its per-vertex maps have **zero outliers**: area maximum error 9.54e−7 mm², thickness 4.77e−7 mm, pial curvature 2.03e−4, and volume 1.91e−6 mm³. The area tolerance is 0.001 + 0.001 × |reference|; the other three use 0.005 + 0.001 × |reference|. Full hashes and exact bit counts are in the report.

The RH diagnostic replay took 956.1 s on shared control-host CPU, versus 83.76 s for the separate installed normal CLI control. Its wall time by outer pass was 483.8, 150.9, 133.4, and 187.9 s. Across all passes, accepted collision/update work took 730.9 s, rejected trial collision work 60.1 s, gradient computation 52.7 s, and repeated input/target setup 109.7 s. The replay writes all 41 checkpoints and compares selected RAM meshes; these runs had differing concurrent jobs, so the ratio is descriptive, not a controlled performance benchmark. Placement remains CPU NumPy/Numba/KDTree; speed optimization and independent accept/retry scheduling are open.

Reproduce the fixed LH optimizer and final metrics on control-host using the unchanged subject and isolated work directory. The license is passed only as an environment path to the installed control; the Python replay does not read it:

```bash
W=/data/workspaces/freesurfer_synth/work/place_surface_python_20260925
S=/data/workspaces/freesurfer_synth/work/reconall_main_20260924/single_subjects/fs_sub01
PY=/data/workspaces/freesurfer_synth/work/reconall_main_20260924/venv/bin/python
BIN=/opt/reference/Freesurfer/8.2.0-1/bin/mris_place_surface
PYTHONPATH="$W" "$PY" "$W/run_place_surface_installed_pial_replay.py" --subject "$S" --root "$W" --module "$W" --hemisphere lh --installed-binary "$BIN" --native-log "$W/probe_installed_official_run/objective_probe.log" --output-dir "$W/independent_lh_pial_first_outer_20260926" --report "$W/place_surface_installed_lh_full_replay_report.json" --checkpoint "41:$W/probe_installed_ram_official_pass3/surf/lh.pial.ram"
PYTHONPATH="$W" "$PY" "$W/validate_place_surface_installed_final_boundary.py" --subject "$S" --root "$W" --module "$W" --hemisphere lh --report "$W/place_surface_installed_lh_final_boundary_with_metrics_report.json" --metrics-output-dir "$W/independent_lh_pial_metrics_20260926"
```

For RH, use the same two commands with `--hemisphere rh`, `--native-log "$W/probe_installed_rh_official_run/installed_pial.log"`, `--output-dir "$W/independent_rh_pial_first_outer_20260926"`, `--checkpoint "41:$W/probe_installed_rh_ram_official_pass3/surf/rh.pial.ram"`, and the RH report/metrics paths. The replay command overwrites only its own isolated step files and report. A clean, uncontended LH replay timing was deliberately not repeated; the earlier LH checkpoint reports and independent final report establish its geometry and metric comparisons. The [bilateral acceptance report](place_surface_installed_bilateral_pial_acceptance_report.json) collects the fixed-input SHA-256 values, exact vertex counts, per-vertex tolerances, RH timing, and test log hashes.

### Installed LH pial independent optimizer decisions, 2026-09-26

The [bounded independent validator](validate_place_surface_independent_decision.py) computes the fixed LH pial target, gradients, collision proposals, SSE, and intensity RMS from the frozen official `lh.white` and MRI inputs. Its [decision kernel](../../../src/fnit/recon_all/place_surface_decision.py) applies the pinned `MRISpositionSurface` tolerance, reduction, and rejection conditions to those Python values. The installed log is parsed only to assert the completed decision; it never supplies a trial `dt`, rejection, or accepted coordinate to the Python optimizer. The Python step-21 state is saved and reused for steps 22–41, so this validation does not repeat a full reconstruction. The installed FreeSurfer 8.2.0-1 binary and every input/checkpoint are SHA-256 pinned in the [first](place_surface_installed_lh_independent_decision_step21_report.json) and [continuation](place_surface_installed_lh_independent_decision_step22_41_area_reset_report.json) reports.

At each new `MRISpositionSurface` outer pass, the pinned native source calls `MRIScomputeMetricProperties` followed by `MRISstoreMetricProperties`. The Python validator likewise resets `orig_area` to the current mesh area before computing that pass's SSE; it recomputes the border target and halves sigma and gradient averaging at the same boundary. The [source-order summary](place_surface_installed_lh_independent_decision_full_summary.json), produced by [this focused audit](summarize_place_surface_independent_decision.py), finds **41/41 independent `dt`/accept/reject decisions** equal to the installed run, with outer-pass endpoints 26/32/36/41. All 14 available RAM checkpoints (steps 1, 2, 5, 10, 12, 13, 14, 20, 21, 26, 32, 36, 37, 41) match every one of the 106,622 ordered vertices and 319,866 float32 coordinate components. The initial SSE/RMS of all four passes and the 41 accepted-step values match the installed log at its printed precision after float32 conversion. Nine of twelve reduction lines also match exactly; the three pass-2 SSE reductions print 0.1 higher in Python while their RMS values match to four decimals. Installed full-precision SSE was not captured, so these printed-value checks do not establish bitwise SSE equality. They do establish the same decisions and bitwise geometry for this fixed LH input.

An earlier comparison accidentally selected `probe_installed_ram_official_step37` (SHA-256 `d77700e1ac3a4315b48d2fb9f2301e3836d00cd8eeaf3c74581c0f7a1acac3d1`), which is not the same-parameter step-37 reference and matched only 14,607 vertices. The correct same-input, same-parameter `probe_installed_ram_official_step37_limited` (SHA-256 `959e68c639c09d63646b65865883db2c8e228183d800fcfe6a71ee2516562119`) matches all 106,622; both references and the original diagnostic are preserved in the audit. The step-37 correction changes the comparison fixture, not the Python optimizer state.

The [direct endpoint check](place_surface_installed_lh_independent_decision_final_metrics_report.json) consumes the independent step-41 state, pins the medial wall, repairs its two intersecting faces in one 100-step smoothing cycle, and compares the resulting `lh.pial` with the installed normal CLI: 106,622/106,622 vertices, 319,866/319,866 coordinate components, and ordered faces exact. Per-vertex `lh.area.pial`, `lh.thickness`, `lh.curv.pial`, and `lh.volume` have zero outliers under the established tolerances; maximum absolute differences are 9.54e−7 mm², 4.77e−7 mm, 4.54e−4, and 1.91e−6 mm³, respectively. The final check uses [the same isolated cleanup implementation](../../../src/fnit/recon_all/place_surface_final_cleanup.py) and the [final-boundary validator](validate_place_surface_installed_final_boundary.py) with `--optimizer-candidate` pointing to the independent output. The [decision branch tests](../../../tests/recon_all/test_place_surface_decision.py) passed 3/3 on control-host.

This establishes a native-log-independent **LH pial optimizer and final pial endpoint** for one frozen subject. RH pial final geometry and vertex maps remain validated through the earlier log-guided control, and white placement and the complete recon-all chain have separate acceptance boundaries. These placement modules run on CPU NumPy/Numba; this result does not establish a GPU speedup.

### Installed RH pial independent optimizer decisions, 2026-09-26

The [bounded decision validator](validate_place_surface_independent_decision.py)
now accepts `--hemisphere rh`, selects the RH frozen white/label/stats/MRI
inputs, and compares only the three available installed RAM checkpoints at
steps 1, 2, and 41. It computes its own SSE, RMS, `dt`, accept/reject
outcome, and coordinates. The native log supplies only assertions after each
decision; none of its decisions enters the candidate state. The RH path uses
the [source-order collision retry and accepted-offset implementation](../../../src/fnit/recon_all/place_surface_collision.py)
ported into this package. All 16 placement modules called by the validator
now match the previously validated source modules byte for byte. The focused
decision/step tests pass 4/4; the prior LH full replay was not repeated.

The first RH step alone matched all 105,541 installed RAM vertices and
316,623 coordinate components ([report](place_surface_rh_independent_decision_step1_control-host.json)).
The fresh [41-step report](place_surface_rh_independent_decision_full_control-host.json)
sets `first_mismatch=null`: **41/41 independently chosen decisions** agree
with the installed log, including three rejected trials and outer-pass
endpoints 26/31/35/41. Steps 1, 2 and 41 match every installed RAM vertex
and float32 coordinate bit. The run uses frozen official `rh.white`, MRI,
autodet thresholds and labels; it does not create the white surface.

The [final-boundary check](place_surface_rh_independent_decision_final_metrics_control-host.json)
starts from this independently chosen step-41 state, pins the medial wall,
repairs intersections, then matches all 105,541 final installed `rh.pial`
vertices, all 316,623 coordinate components and every ordered face exactly.
The independently regenerated maps have zero outliers under the existing
rules: `area.pial` maximum error 9.54e-7 mm², thickness 4.77e-7 mm,
pial curvature 2.03e-4, and vertex volume 1.91e-6 mm³. This replaces the
prior log-guided RH control for the decision/endpoint gate. White placement,
connected orchestration and subject-level T1-to-metrics validation remain open.

## Time boundary

The installed diagnostic calls took 4.20–9.56 s each on control-host; these include input loading, MRI preprocessing, rip handling, and target work where enabled. The PyTorch smoothing call took 3.29–3.31 s per hemisphere on the same host, including Python topology construction; the marked-value calls took 4.20–4.49 s. These are **not** paired full-stage benchmarks and do not establish a placement speedup. Two source-probe runs took 12.05–13.23 s for two one-step white outer passes and 7.13–8.56 s for one one-step pial pass, with diagnostic writes and a different build. Their corresponding step snapshots were exactly equal. LH pial final output parity is established for this fixed input. Full white-plus-pial timing and a paired speed benchmark remain unmeasured.
