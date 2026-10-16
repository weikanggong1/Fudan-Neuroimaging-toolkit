# Fixed FreeSurfer 8.2 `mris_register`: partial exact checkpoints and default-path boundary

## Reference call and inputs

The completed `fs_sub01` log runs the two commands through `rca-surfreg`:

```text
mris_register -curv -threads 4 surf/lh.sphere average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif surf/lh.sphere.reg
mris_register -curv -threads 4 surf/rh.sphere average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif surf/rh.sphere.reg
```

The actual paths are under
`/data/workspaces/freesurfer_synth/work/reconall_main_20260924`.
The pinned source is FreeSurfer 8.2.0 commit
`d932c45b7941662ea380a05efef580568b98d41a`. The active path is
`mris_register/mris_register.cpp` → `utils/mrisurf_integrate.cpp` and
`utils/mrisurf_deform.cpp`. It also reads `surf/{lh,rh}.sulc` and
`surf/{lh,rh}.smoothwm`; the default registration starts at sulc, then uses
smoothwm curvature. The two atlas TIFFs are 2,857,642 and 2,858,470 bytes
and are already in the external asset manifest. Each contains nine 512×256
32-bit frames. `tifffile` exposes their signed integer storage; viewing the
same bits as float32 yields plausible mean, variance and degree-of-freedom
frames, including a maximum degree-of-freedom value of 40 in LH frame 2.

The archived full native calls took **243.41 s LH** and **233.33 s RH** on
`accelerator-a-1` with four CPU threads (`@#@FSTIME` wall seconds). Their initial
rigid search took about 0.15/0.16 min according to the log. These are
archived native timings, not a matched candidate comparison.

## Exact isolated projection

`mris_register_kernels.py` implements bounding-box centering and one radial
projection in PyTorch with the source's float32 stored coordinates and
float64 distance arithmetic. It also implements mean curvature normalization
with a ripped-vertex mask. This module does **not** implement atlas matching
or registration.

For each hemisphere, a **fresh native diagnostic** ran on the frozen input
with `-N 0 -norot -remove_negative 0`; it did not run full recon-all. Source
control flow applies one initial radial projection, then one at the start of
each zero-step integration epoch. The first pass has nine averaging scales;
the four registration sigmas have seven each, for **38 projections** total.
The Python benchmark repeated the single-step tensor kernel 38 times and
compared its ordered vertex coordinates and faces with the native diagnostic:

| Hemisphere | Vertices | Faces | Exactly equal vertices | Maximum coordinate error | Torch CPU projection only |
| --- | ---: | ---: | ---: | ---: | ---: |
| LH | 106,622 | 213,240 | 106,622 | 0 mm | 0.174 s |
| RH | 105,541 | 211,078 | 105,541 | 0 mm | 0.132 s |

The native zero-step diagnostics reported about 8.3/6.5 s and also read the
atlas, build parameterizations and evaluate the registration objective. The
Torch numbers are one warm measurement per side, exclude I/O and those other
operations, and are **not** a full-stage speed comparison. An earlier run
measured 0.258/0.306 s, indicating run-to-run variation. The focused tests
passed 3/3. The CPU reports
include SHA-256 values for both frozen inputs and fresh native outputs:
[`LH`](mris_register_lh_zero_step_control-host.json),
[`RH`](mris_register_rh_zero_step_control-host.json). Reproduce the tensor check with
[`benchmark_mris_register_kernels.py`](benchmark_mris_register_kernels.py).

The source sulc values before normalization have mean/std
`2.979e-9 / 5.52030677` LH and `-5.488e-8 / 5.52782359` RH; the native
diagnostic prints `0.000 / 5.520` and `-0.000 / 5.528`. The native command
does not expose its normalized per-vertex vector here, so curvature
normalization of source sulc has **scalar-log parity only**. The target atlas
normalization is checked per vertex below.

## Exact unblurred atlas sampling on the canonical sphere

Fresh native diagnostics used `-N 0 -norot -remove_negative 0 -W 0` in
isolated LH/RH directories. They wrote a per-vertex atlas curvature fixture
named `sno1_target_blur4.00`. Despite that suffix, the source writes this
curve from the **unblurred** `mrisp_template`; each of the four
`target_blur{4.00,2.00,1.00,0.50}` diagnostic files has the same hash within
each hemisphere. The TIFF's signed 32-bit storage was reinterpreted as
float32, and frame 3 was sampled at the frozen canonical sphere vertices.

`mris_register_atlas.py` follows `MRISfromParameterization` float32 spherical
coordinates, pole reflection, azimuth wrap and bilinear arithmetic. It matches
the native diagnostic **exactly at every vertex**. Applying the existing
PyTorch `normalize_mean_curvature` to these values also matches the separately
written native normalized `target1` curve at every vertex:

| Hemisphere | Exact raw / normalized values | Maximum absolute error | Torch CPU sampler only | Native diagnostic |
| --- | ---: | ---: | ---: | ---: |
| LH | 106,622 / 106,622 each | 0 | 0.0217 s | 10.40 s |
| RH | 105,541 / 105,541 each | 0 | 0.0683 s | 12.34 s |

The native diagnostic runs all four scales and writes multiple curves; its
time is **not** directly comparable with one Torch sampling call. Torch
timings are single CPU observations excluding file I/O and startup. Paired
input/output SHA-256 values and vertex errors are in
[`LH`](mris_register_lh_atlas_control-host.json) and
[`RH`](mris_register_rh_atlas_control-host.json). Reproduce with
[`probe_mris_register_atlas.py`](probe_mris_register_atlas.py). The analytical
sphere-axis test passed 1/1. This checkpoint does not cover source curvature
parameterization, blurred mean-frame sampling or angle search. Atlas variance
blur is checked separately below.

## Exact spherical blur of atlas variance frame

The installed native binary cannot write its `.hipl` debug grids (`hips is not
supported`). For validation only, a copy of that binary changed **12 bytes**
in three debug filename strings from `.hipl` to `.mgz`; no arithmetic or
official binary was changed. The original binary SHA-256 is
`75137b92fcbed63b441e6214b5b923d7664f05d00c454b62da724dec26636f80`;
the diagnostic copy is
`0dadb9864bb3225ac4557b0dbc37bbccb06a5b6c0e851ca8f310ad7fe0edb1fe`.
This exposed `mrisp_template_blur.mgz` after the last, sigma 0.5, pass.
Its frame 4 is the atlas variance frame blurred directly by `MRISPblur_new`;
the source does not apply target-curve normalization to this variance frame.

`mris_register_blur.py` implements the latitude-dependent kernel, pole
reflection, azimuth wrap and double-precision accumulation in PyTorch. It
matches the diagnostic variance grid at **every pixel**:

| Hemisphere | Exact variance pixels | Maximum absolute error | Torch CPU blur only |
| --- | ---: | ---: | ---: |
| LH | 131,072 / 131,072 | 0 | 0.337 s |
| RH | 131,072 / 131,072 | 0 | 0.229 s |

Paired atlas/grid hashes and single-run timings are in
[`LH`](mris_register_lh_blur_control-host.json) and
[`RH`](mris_register_rh_blur_control-host.json). Reproduce with
[`probe_mris_register_blur_frame.py`](probe_mris_register_blur_frame.py);
the constant-frame unit test passed. The source and target mean-frame grids
are checked next.

## Complete rigid-search input grids at sigma 4

The first rigid search uses sigma 4. A fresh, isolated `-N 0 -W 0` diagnostic
with rigid alignment enabled wrote the source sulc grid (frame 0), normalized
target atlas mean grid (frame 3), and atlas variance grid (frame 4) through
the validation-only MGZ debug copy. The native search again selected LH
`(10.5, -10.5, -8.0)` and RH `(19.5, -9.5, -2.0)` degrees. Python starts from
the frozen sphere, applies one radial projection, then reproduces curvature
normalization, spherical rasterization, sigma-4 blur, grid-to-vertex sampling,
normalization and reparameterization. Each grid matches **all 131,072 native
pixels exactly** on both sides:

| Search input | LH Torch CPU | RH Torch CPU | Exact pixels per side |
| --- | ---: | ---: | ---: |
| Source sulc grid | 7.746 s | 7.257 s | 131,072 / 131,072 |
| Target atlas mean grid | 6.923 s | 6.936 s | 131,072 / 131,072 |
| Target atlas variance grid | 5.613 s | 7.114 s | 131,072 / 131,072 |

These are three separate Python calls and include repeated sigma-4 blur work;
they are not a combined-stage benchmark. The native rigid diagnostics took
7.21/7.73 s and include the angle search, so their timing is also a different
workload. Input and native-grid hashes, all-pixel errors and timings are in
[`LH source`](mris_register_lh_source_sigma4_control-host.json),
[`LH target`](mris_register_lh_target_sigma4_control-host.json),
[`LH variance`](mris_register_lh_variance_sigma4_control-host.json),
[`RH source`](mris_register_rh_source_sigma4_control-host.json),
[`RH target`](mris_register_rh_target_sigma4_control-host.json), and
[`RH variance`](mris_register_rh_variance_sigma4_control-host.json). The analytical
atlas, blur and parameterization tests passed 3/3. The independent rigid search is verified below; nonlinear deformation remains open.

## Exact rigid transform with native-selected angles

A second fresh native diagnostic kept the default rigid search and set
`-N 0 -remove_negative 0`. It produced registered spheres after the rigid
search, without nonlinear deformation. The native logs selected angles
`(10.5, -10.5, -8.0)` degrees LH and `(19.5, -9.5, -2.0)` degrees RH.
`rotate_sphere` implements the source's float32 `MRISrotate` coefficient and
arithmetic order. Given those native-selected angles, `rigid_grid_angle`, one
projection and `rotate_sphere` match **every ordered native vertex exactly**:

| Hemisphere | Exact vertices | Faces equal | Native rigid diagnostic | Torch CPU projection + rotation only |
| --- | ---: | --- | ---: | ---: |
| LH | 106,622 / 106,622 | yes | 7.51 s | 0.0057 s |
| RH | 105,541 / 105,541 | yes | 7.17 s | 0.0062 s |

These are different workloads: native timing includes atlas reads, curvature
parameterization, blurring and global angle search; Torch timing excludes I/O,
startup and angle search. It is **not** a registration speedup measurement.
The frozen input and fresh rigid output SHA-256 values, angle inputs, per-vertex
errors and kernel timings are in the paired
[`LH`](mris_register_lh_rigid_control-host.json) and
[`RH`](mris_register_rh_rigid_control-host.json) reports. The native rigid outputs
have SHA-256 `b22b3be1f006848b4dc616cf05d68351064f4e7055d341ea5c817d4b4b2c33e4`
LH and `4918b98f7edb6499378a8c7863d006e41528f76bbe1ca47146084d9eab342d47`
RH. Reproduce the vertex check with
[`probe_mris_register_rigid.py`](probe_mris_register_rigid.py).

## Independent rigid objective and angle search

`mris_register_objective.py` now reproduces the FreeSurfer lookup-table
`atan2`, atlas mean/variance interpolation, absolute standardized curvature
objective, 16-partition summation and one-center coarse-to-fine angle grid.
The search uses no angles or intermediate grids from the native run.
Python computes the source curve from frozen `sphere` and `sulc`, and both
target grids from the atlas TIFF. The native source curve and all three grids
are used only as parity references. Both hemispheres match every source-curve
vertex and all 131,072 pixels in each of the source, target-mean and
target-variance grids before angle search.

| Hemisphere | Source curve exact | Independent angle choice | Objective | Angle evaluations | Search + rotation CPU | Ordered rigid vertices exact |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| LH | 106,622 / 106,622 | (10.5, -10.5, -8.0)° | 193495.33136 | 3,337 | 30.10 s | 106,622 / 106,622 |
| RH | 105,541 / 105,541 | (19.5, -9.5, -2.0)° | 208867.47737 | 3,172 | 20.90 s | 105,541 / 105,541 |

Both face arrays match the native rigid diagnostic. Native logs print the
objective to one decimal place, respectively 193495.3 and 208867.5. The
source-curve preparation took 21.32/16.56 s LH/RH and both target grids
14.01/10.78 s in these CPU runs. The full Python rigid preparation, search
and rotation therefore took about 65.43/48.25 s excluding file I/O. The
paired native `-N 0` rigid diagnostic took 7.21/7.73 s including file I/O;
this Python rigid stage is currently slower. These are single observations on
a shared host, not a controlled speed benchmark or a GPU result. Input hashes, ordered-vertex checks and
single-run times are in [`LH`](mris_register_lh_search_control-host.json) and
[`RH`](mris_register_rh_search_control-host.json); reproduce with
[`probe_mris_register_search.py`](probe_mris_register_search.py). Focused
objective tests passed 2/2 in the remote PyTorch environment.

The search arithmetic follows pinned FreeSurfer 8.2 source
`utils/MRISrigidBodyAlignGlobal.cpp`, `utils/mrisp.cpp`, `utils/utils.cpp` and
`utils/vertexRotator.cpp` at commit
`d932c45b7941662ea380a05efef580568b98d41a`.

The reusable tensor API is
[`register_rigid`](../../../src/fnit/recon_all/mris_register_rigid.py).
It takes `vertices`, `sulc`, and the float32-reinterpreted atlas mean and
variance frames, and returns rigid vertices, angles, SSE and search count.
Independent API validation on both frozen hemispheres matched every ordered
native rigid vertex and all faces again: LH **44.33 s**, RH **45.27 s** on
CPU excluding I/O. It omits the diagnostic source-grid reconstruction and
comparison, so these times are shorter than the probe above. The paired API
reports are [`LH`](mris_register_lh_rigid_api_control-host.json) and
[`RH`](mris_register_rh_rigid_api_control-host.json); reproduce with
[`probe_mris_register_rigid_api.py`](probe_mris_register_rigid_api.py).

## First nonlinear checkpoints

A separate official-binary diagnostic used `-N 1 -W 1 -remove_negative 0`
on frozen LH/RH inputs in isolated scratch directories. Its saved
`sno1.rotated` surface equals the independently matched rigid diagnostic at
every ordered vertex and face. The first integration epoch then applies one
more sphere projection, forms distance, angle/area, and curvature-correlation
forces, averages the gradient **16,384** times, and performs quadratic line
minimization. Its first saved `debug0002` surface provides an ordered-vertex
reference:

| Hemisphere | Observed line-search `dt` | Unchanged ordered vertices | Median / 95th percentile displacement from rigid |
| --- | ---: | ---: | ---: |
| LH | 70.93703089944263 | 0 / 106,622 | 4.36 / 6.32 mm |
| RH | 69.6503677368164 | 0 / 105,541 | 5.21 / 6.67 mm |

Both face arrays remain equal. A second LH diagnostic with `-V 0` reproduced
the first saved surface exactly. Reference surface hashes and displacements
are in [`LH`](mris_register_lh_first_step_control-host.json) and
[`RH`](mris_register_rh_first_step_control-host.json), generated by
[`probe_mris_register_first_step.py`](probe_mris_register_first_step.py).
These diagnostics were stopped after saving that surface, with no full
`recon-all` rerun.

The installed binary's `DIAG_VERBOSE=1` per-vertex gradient writer crashes in
`MRISwriteIntoVolume`, so read-only GDB breakpoints captured the native
`MRI_SURFACE` gradients. The breakpoints are pinned to the installed
`mris_register` Build ID `a7727d3bbfa92bd11932ea9981c2399344c217a2`:
`MRISintegrate` calls the first gradient average at `0x480b46`, and the
first line minimization begins at `0x47e890`. The validation scripts
[`capture_mris_register_first_force_stages.gdb`](capture_mris_register_first_force_stages.gdb),
[`capture_mris_register_distance_inputs.gdb`](capture_mris_register_distance_inputs.gdb),
[`capture_mris_register_one_average.gdb`](capture_mris_register_one_average.gdb), and
[`capture_mris_register_first_average.gdb`](capture_mris_register_first_average.gdb)
read arrays at those checkpoints without modifying the native binary. GDB
must `continue` to a return breakpoint when OpenMP workers are active;
stepping over the averaging call stalls those workers.

The first native gradient is zero before the distance force. Distance changes
it, angle/area changes it again, and curvature correlation gives the
pre-average gradient. The polar-correlation and subsequent zero-weight terms
do not change that first pre-average gradient. Exact per-stage SHA-256 hashes,
first vectors, scalar values and native diagnostic times are in the
[`LH`](mris_register_lh_first_force_stages_control-host.json) and
[`RH`](mris_register_rh_first_force_stages_control-host.json) reports. These are
observed native checkpoints. Python now generates the distance and
angle/area forces independently; curvature correlation remains open.

## Independent first distance force

[`first_distance_gradient`](../../../src/fnit/recon_all/mris_register_nonlinear.py)
takes the original input sphere, the subject's `smoothwm`, the rigidly aligned
sphere, and ordered faces. It reproduces the extra pre-integration sphere
projection, source-order one-ring, default face-corner vertex normals, current
sphere arc lengths, original `smoothwm` chord lengths, and the legacy
three-hop average-neighbor count. It derives the original area from input
sphere triangles and the total area from the theoretical 100 mm sphere.
All geometry is stored as float32 before these operations, as in FreeSurfer.
It then applies the source distance-force arithmetic. The rigidly aligned
sphere was provided as the native fixed checkpoint in this isolated probe;
the independent [`register_rigid`](../../../src/fnit/recon_all/mris_register_rigid.py)
API already reproduces that checkpoint on both sides.

| Hemisphere | Exact ordered normals | Exact ordered current/original edges | Exact distance-force vectors | Python CPU stage excluding I/O |
| --- | ---: | ---: | ---: | ---: |
| LH | 106,622 / 106,622 | 639,720 / 639,720 each | 106,622 / 106,622 | 7.78 s |
| RH | 105,541 / 105,541 | 633,234 / 633,234 each | 105,541 / 105,541 | 7.72 s |

The exact first-distance API hashes and timings are in
[`LH`](mris_register_lh_first_distance_api_control-host.json) and
[`RH`](mris_register_rh_first_distance_api_control-host.json); reproduce with
[`probe_mris_register_first_distance_api.py`](probe_mris_register_first_distance_api.py).
The separately checked edge-length and distance-force components are in
[`LH`](mris_register_lh_distance_from_surfaces_control-host.json) and
[`RH`](mris_register_rh_distance_from_surfaces_control-host.json), reproducible with
[`probe_mris_register_distance_from_surfaces.py`](probe_mris_register_distance_from_surfaces.py).
The CPU stage times include Python topology construction and three-hop
counting, but exclude input reads and the preceding rigid search. They are
single observations on a shared host. The native gradient-checkpoint runs
were 8.66/6.54 s LH/RH from program startup and include atlas reads and
rigid preparation, so those are not matched step timings.

## Independent first angle/area force

[`first_area_gradient`](../../../src/fnit/recon_all/mris_register_nonlinear.py)
continues from the distance gradient. It regenerates each current sphere
triangle area and unit face normal, each original `smoothwm` triangle area,
and then applies the native nonlinear-area and percentage-area contributions
in face order. The Python result matches every face input and every ordered
post-area gradient in both hemispheres. The paired probe also passes the
independent `first_distance_gradient` output into the area API, so the
combined distance-plus-area result has no native intermediate force input:

| Hemisphere | Exact current/original areas and face normals | Exact post-area vertex gradients | Python CPU area only / distance plus area |
| --- | ---: | ---: | ---: |
| LH | 213,240 / 213,240 each | 106,622 / 106,622 | 4.92 / 12.18 s |
| RH | 211,078 / 211,078 each | 105,541 / 105,541 | 4.98 / 12.17 s |

The fixed input, native checkpoint and predicted gradient SHA-256 values are
in [`LH`](mris_register_lh_first_area_api_control-host.json) and
[`RH`](mris_register_rh_first_area_api_control-host.json); reproduce with
[`probe_mris_register_first_area_api.py`](probe_mris_register_first_area_api.py).
Times exclude file reads and the preceding rigid search. The native face
arrays were read at the start of the area force with
[`capture_mris_register_area_inputs.gdb`](capture_mris_register_area_inputs.gdb).

## Independent first curvature-correlation force

`sample_correlation_atlas` reproduces `MRISPfunctionVal` pole wrapping,
bilinear sampling and off-sphere projection. `tangent_basis` follows the
native float32 cross-product and normalization order. The scalar correlation
force then samples the sigma-4 target mean along both tangent axes and uses
the target variance and source curvature to update the distance-plus-area
gradient. In this fixed T1 case, the later polar and vector terms do not
change the first pre-average gradient.

The paired probe builds the source curve from input `sphere` and `sulc`, the
target mean/variance grids from the atlas TIFF, and the full first distance,
area and correlation forces from input `sphere`, `smoothwm` and the rigidly
aligned sphere. The isolated rigid producer is independently exact on both
sides, but this probe uses its frozen native rigid surface as the starting
position to keep the force check separate. Every source-curve vertex,
tangent-axis vertex, target-grid pixel and final pre-average gradient equals
the native checkpoint:

| Hemisphere | Exact source curve / e1 / e2 vertices | Exact target mean / variance pixels | Exact complete pre-average gradient | Python CPU curve / grids / forces, excluding I/O and rigid |
| --- | ---: | ---: | ---: | ---: |
| LH | 106,622 / 106,622 each | 131,072 / 131,072 each | 106,622 / 106,622 | 5.32 / 10.48 / 14.84 s |
| RH | 105,541 / 105,541 each | 131,072 / 131,072 each | 105,541 / 105,541 | 7.24 / 14.26 / 18.89 s |

The predicted/native float32 gradient hashes are identical:
`6d26291ee71423747b186fe46d8eb53204c441c8989c88f2f98db4b86d352622`
LH and
`373bae6caa23f0f5df85fa6825188e1563a17028c3a65789163a17116074fb26`
RH. Paired input hashes, exact counts and timings are in
[`LH`](mris_register_lh_first_correlation_api_control-host.json) and
[`RH`](mris_register_rh_first_correlation_api_control-host.json), reproducible with
[`probe_mris_register_first_correlation_api.py`](probe_mris_register_first_correlation_api.py).
The native arrays were read by
[`capture_mris_register_correlation_inputs.gdb`](capture_mris_register_correlation_inputs.gdb)
on the pinned ELF, and the reference sigma-4 target grid was captured by the
validation-only debug-copy run described above. The PyTorch inference uses
neither array. These CPU timings are single observations on a shared host;
the archived native full-registration calls have a different scope.

## Complete first gradient average and conditional first update

`ordered_neighbors_from_faces` recovers the native one-ring order from the
surface's face order. It matches all **639,720 LH** and **633,234 RH** native
ordered neighbor entries. A debugger diagnostic changed only the first
average count from 16,384 to one, then stopped at the return instruction.
`average_gradients_once` matched every native resulting vector byte for byte.
The paired one-iteration reports are
[`LH`](mris_register_lh_first_average_control-host.json) and
[`RH`](mris_register_rh_first_average_control-host.json); reproduce with
[`probe_mris_register_first_average.py`](probe_mris_register_first_average.py).

With the **actual 16,384** iterations, `average_gradients` independently
matched every native post-average vector exactly when given the captured
pre-average gradient: 106,622/106,622 LH and 105,541/105,541
RH, maximum error zero. The native post-average hashes also equal the
independently captured gradients at the following line-search entry. Paired
hashes and timings are in
[`LH`](mris_register_lh_full_average_control-host.json) and
[`RH`](mris_register_rh_full_average_control-host.json); reproduce with
[`probe_mris_register_full_average.py`](probe_mris_register_full_average.py).
The new source-input probe now reproduces that same pre-average gradient
exactly, so the isolated average and source-input force checks compose to
the native first post-average gradient. PyTorch CPU averaging alone took
**44.18 s LH** with its default thread count
and **76.31 s RH** with four threads, excluding I/O and adjacency setup.
The native GDB runs through that checkpoint took 13.06/11.42 s from program
startup, including earlier work. These configurations and scopes differ, so
this is a parity result and not a speedup claim.

Finally, `apply_spherical_gradient` reproduces the first saved native surface
at **every ordered vertex** on both sides when supplied the observed native
pre-line-search gradient and selected `dt`; see
[`LH`](mris_register_lh_first_gradient_apply_control-host.json) and
[`RH`](mris_register_rh_first_gradient_apply_control-host.json), reproducible with
[`probe_mris_register_first_gradient_apply.py`](probe_mris_register_first_gradient_apply.py).
That conditional check supplied the native-selected `dt`; the independent
selection is validated next. The focused PyTorch nonlinear-kernel tests passed
**8/8** on the remote validation environment.

## Independent first nonlinear line search

[`mris_register_line_search.py`](../../../src/fnit/recon_all/mris_register_line_search.py)
now computes the four active first-pass SSE terms and tests the native decade
steps, three-point bracket and float32 quadratic candidate. The original
`smoothwm` face areas and ordered chord distances are regenerated from the
frozen source surfaces and equal the native checkpoint at every entry. The
post-average gradient and starting positions are frozen native checkpoints;
the independent force and averaging probes above reproduce that gradient
exactly from source inputs on both sides. No native trial SSE or selected
`dt` enters the Python search.

The pinned `mrisLineMinimize` sums float32 gradient squares and then their
double-precision square roots in vertex order. Its three-point fit stores the
normal equations and inverse in float32 matrices. Matching those reduction
and matrix orders resolves the prior tiny LH step discrepancy and the larger
RH quadratic-candidate discrepancy. Both independently selected steps now
equal the native double values, and the projected output matches every
ordered vertex and face of the first native saved surface:

| Hemisphere | Native and PyTorch selected `dt` | Exact first-step vertices | Maximum coordinate error | PyTorch CPU setup / search and final projection |
| --- | ---: | ---: | ---: | ---: |
| LH | 70.93703089944263 | 106,622 / 106,622 | 0 mm | 1.34 / 0.346 s |
| RH | 69.6503677368164 | 105,541 / 105,541 | 0 mm | 1.23 / 0.358 s |

A separate read-only GDB diagnostic set `FREESURFER_logSSE=1` and stopped
after the first line search. It captured all nine corresponding native SSE
trials and their active percentage-area, nonlinear-area, distance and
curvature-correlation terms. The native logger prints six decimal places per
term. Across those nine trials, the maximum absolute differences are:

| Hemisphere | Area | Nonlinear area | Distance | Correlation | Total SSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| LH | 8.32e-5 | 2.81e-6 | 4.98e-7 | 4.44e-7 | 8.60e-5 |
| RH | 8.41e-5 | 5.23e-7 | 3.68e-7 | 4.30e-7 | 8.40e-5 |

The native float32 quadratic fit yields `(a,b)=(25.81787109375,-1667.25)`
LH and `(45.60546875,-3176.4375)` RH. Both coefficients and the predicted
candidate steps are reproduced exactly from the three native bracket points;
LH selects the lower bracket endpoint and RH selects its quadratic candidate.
The component differences are numerical SSE differences, not bitwise SSE
parity. The native GDB script prints the full buffered term log before a
post-flush debugger register-state error; all 12 log records, including the
nine line-search trials, are present in each retained log. These GDB wall
times and the CPU-only PyTorch kernel times are not matched stage benchmarks.

Paired reports are [`LH`](mris_register_lh_first_line_parity_control-host.json)
and [`RH`](mris_register_rh_first_line_parity_control-host.json). The raw PyTorch
reports are [`LH`](mris_register_lh_first_line_search_control-host.json) and
[`RH`](mris_register_rh_first_line_search_control-host.json); the native
[`LH fit`](mris_register_lh_native_first_line_fit_control-host.log),
[`RH fit`](mris_register_rh_native_first_line_fit_control-host.log),
[`LH terms`](mris_register_lh_native_first_line_terms_control-host.log), and
[`RH terms`](mris_register_rh_native_first_line_terms_control-host.log) logs retain
the reference values. Reproduce the frozen PyTorch trial with
[`probe_mris_register_first_line_search.py`](probe_mris_register_first_line_search.py),
the native captures with
[`capture_mris_register_first_line_fit.gdb`](capture_mris_register_first_line_fit.gdb)
and [`capture_mris_register_first_line_terms.gdb`](capture_mris_register_first_line_terms.gdb),
and the paired summary with
[`summarize_mris_register_first_line_search.py`](summarize_mris_register_first_line_search.py).
The focused float32 fit test passed **1/1** on the remote PyTorch environment
and checks both captured hemispheres.

## Connected second sigma-4 integration call

The next `MRISintegrate` call uses 4,096 gradient averages and saves `debug0003`
under an isolated installed-FreeSurfer `-N 1 -W 1 -remove_negative 0` diagnostic.
The read-only [checkpoint script](capture_mris_register_second_epoch.gdb) captured
its starting positions/normals, distance, area and correlation gradients, and
post-average gradient from the **installed** FreeSurfer 8.2 binary (SHA-256
`75137b92fcbed63b441e6214b5b923d7664f05d00c454b62da724dec26636f80`).
The [Python probe](probe_mris_register_second_epoch.py) independently prepares
the sigma-4 source curve and target grids from the original sphere, sulc and
atlas; then it performs both updates in sequence. It starts from the frozen
rigid surface, which the independent `register_rigid` API already matched at
every ordered vertex on both sides. It does **not** read the native first-step
surface, gradient or selected `dt` as a computation input; those are comparison
references only. Its face-order assertions pass for all input and reference
surfaces.

| Hemisphere | First saved vertices exact | Second start / normals / each force / 4,096-average exact | Second saved vertices exact | Python second `dt` |
| --- | ---: | ---: | ---: | ---: |
| LH | 106,622 / 106,622 | 106,622 / 106,622 at each checkpoint | 106,622 / 106,622 | 85.39686584472656 |
| RH | 105,541 / 105,541 | 105,541 / 105,541 at each checkpoint | 105,541 / 105,541 | 66.45890045166016 |

Every listed coordinate/gradient comparison has maximum absolute error **0**;
the raw paired reports retain per-input, official file and capture hashes:
[LH](mris_register_lh_second_epoch_control-host.json),
[RH](mris_register_rh_second_epoch_control-host.json). The installed log prints the second `dt` as 85.397 LH and 66.459 RH. A
read-only [quadratic-fit capture](capture_mris_register_second_line_fit.gdb)
confirmed the exact values above and retained each side's three SSE samples:
[LH](mris_register_lh_second_line_fit_control-host.log),
[RH](mris_register_rh_second_line_fit_control-host.log).

The first independent attempt matched all second-call gradients exactly but
chose LH `dt=85.38683319091797`, leaving only 3/106,622 saved vertices exact.
The first divergent operator was the float32 **3×3 VXL inverse** inside the
quadratic fit. The sampled `dt`, SSE values after float32 conversion, normal
matrix and right-hand side were identical to native. Our determinant summed
its six terms in a different order (`33837056` versus native-order `33835008`).
Correcting the term order reproduced native coefficients and `dt`, and the
complete sequential rerun then matched all vertices. The regression test
covers both original first fits and both second fits. The focused nonlinear
and line-fit tests passed **9/9** in the same remote PyTorch environment after
the correction.

Python CPU stage times, four threads and one observation on a shared host,
exclude file I/O and the preceding rigid search:

| Stage | LH seconds | RH seconds |
| --- | ---: | ---: |
| Prepare source curve and target grids | 21.31 | 21.29 |
| First force / 16,384 averages / line search | 15.49 / 98.75 / 0.26 | 17.80 / 102.97 / 0.36 |
| Second force / 4,096 averages / line search | 16.46 / 25.18 / 0.30 | 17.91 / 25.87 / 0.40 |

These CPU probe times are not comparable to the archived **full** native
registration times above, because the probe covers only two integration calls
and omits rigid search, I/O, later scales and negative-face repair. There is
no GPU result or full-stage speed comparison for registration yet.

## Connected third sigma-4 integration call

The following `MRISintegrate` call uses **1,024** gradient averages and saves
`debug0004` in the same isolated `-N 1 -W 1 -remove_negative 0` diagnostic.
The official installed-binary [checkpoint script](capture_mris_register_third_epoch.gdb)
was stopped after its third post-average gradient; its LH/RH GDB wall times were
26.90/26.93 s, including program startup, the first two calls, GDB and file
writes. The expanded [Python probe](probe_mris_register_second_epoch.py)
propagates from the matched rigid seed through all **three** updates without
reading native update coordinates, gradients or `dt` as calculation inputs.
It uses the official `debug0002`, `debug0003` and `debug0004` only for
comparison. Ordered faces are asserted equal throughout.

| Hemisphere | Third start, normals, each force and post-average gradient | Third saved ordered vertices | Exact native and Python `dt` | Native negative faces after update |
| --- | ---: | ---: | ---: | ---: |
| LH | 106,622 / 106,622 each | 106,622 / 106,622 | 29.758642196655273 | 1 |
| RH | 105,541 / 105,541 each | 105,541 / 105,541 | 25.513355255126953 | 0 |

All listed maximum absolute errors are **0**. The paired
[LH](mris_register_lh_third_epoch_control-host.json) and
[RH](mris_register_rh_third_epoch_control-host.json) reports include source,
original-input, reference-surface and capture-script SHA-256 values. Exact
installed-binary quadratic fit inputs, coefficients and steps are in the
[LH](mris_register_lh_third_line_fit_control-host.log) and
[RH](mris_register_rh_third_line_fit_control-host.log) logs, captured by
[the read-only script](capture_mris_register_third_line_fit.gdb). The corrected
VXL inverse already reproduced both fits; no further computational fix was
needed. The focused nonlinear and first-three-fit tests passed **9/9** after
adding both third-fit reference cases.

PyTorch CPU timings for only the third call (four threads, one shared-host
observation, excluding I/O and earlier registration work) were **17.89 s**
force, **7.36 s** averaging and **0.41 s** line search LH; **16.03 s**,
**6.00 s** and **0.32 s** RH. These timings do not include the earlier source
curve/atlas setup, first two updates or rigid search, and the installed GDB
capture has a different scope. They are not a native-versus-Python speed
benchmark. No GPU timing or final `sphere.reg` parity is established here.

## Connected fourth sigma-4 integration call

The next installed-FreeSurfer `MRISintegrate` call averages the gradient
**256** times and saves `debug0005`. The read-only
[checkpoint capture](capture_mris_register_fourth_epoch.gdb) recorded native
positions, normals, distance/area/correlation forces and the post-average
gradient for each hemisphere. The [quadratic-fit capture](capture_mris_register_fourth_line_fit.gdb)
recorded all three native SSE samples, the float32 fit matrices and exact
selected step. Both used the same installed 8.2 binary (SHA-256
`75137b92fcbed63b441e6214b5b923d7664f05d00c454b62da724dec26636f80`)
and frozen inputs. The Python probe started from the matched rigid surface and
propagated all four updates without reading native update coordinates, forces
or selected steps as calculation inputs; native files were comparisons only.

| Hemisphere | First, second and third saved vertices | Fourth start, normals, each force, 256-average and saved vertices | Fourth native and Python `dt` | Maximum coordinate/force error |
| --- | ---: | ---: | ---: | ---: |
| LH | 106,622 / 106,622 each | 106,622 / 106,622 at every checkpoint | 20.325061798095703 | 0 |
| RH | 105,541 / 105,541 each | 105,541 / 105,541 at every checkpoint | 16.469606399536133 | 0 |

The first fourth-call attempt matched LH distance force but diverged at the
area force on precisely the three vertices of face **11154** (41302, 41312,
42106), with maximum gradient error **5.44e-4**. This was the sole inward
face after the third LH update. The source `mrismp_OrientEllipsoid` stores its
area as negative and reverses its face normal. Applying that rule to current
spherical faces made the LH area force exact at all 106,622 vertices. The RH
force was already exact, but its line fit differed slightly because trial
surfaces can contain inverted faces; applying signed spherical areas in the
SSE objective reproduced the installed fit and saved surface. Smoothwm
original-face areas remain unsigned. No other computational change was
needed. The regression test includes an inverted spherical face and both
fourth-fit matrices; the targeted nonlinear and line-fit tests passed **10/10**
in the remote PyTorch environment.

The paired [LH](mris_register_lh_fourth_epoch_control-host.json) and
[RH](mris_register_rh_fourth_epoch_control-host.json) reports retain original-input,
reference-surface, installed-binary, GDB-script and Python-source SHA-256
values, together with every checkpoint's exact-vertex count and maximum
error. Native [LH](mris_register_lh_fourth_capture_control-host.log) and
[RH](mris_register_rh_fourth_capture_control-host.log) checkpoint logs and
[LH](mris_register_lh_fourth_line_fit_control-host.log) and
[RH](mris_register_rh_fourth_line_fit_control-host.log) fit logs are retained. The
reference `debug0005` file hashes are
`5ffbebe84b0ddee66ba84bdd65b01481a01525a24e029825904e84ae83b35a72`
LH and `ef4f059d1542b2934d4535be5e304603a2a4b9d0d32077b676f598b6fce24bae`
RH. The initial divergent reports are also retained:
[LH](mris_register_lh_fourth_epoch_initial_control-host.json) and
[RH](mris_register_rh_fourth_epoch_initial_control-host.json).

Fourth-call Python CPU force / 256-average / line-search timings were
**15.35 / 1.35 / 0.30 s LH** and **17.14 / 1.64 / 0.46 s RH**, four threads,
excluding I/O and preceding setup/updates. The full connected Python probe
wall times were 219.67/231.49 s LH/RH. The native GDB checkpoint captures
took 25.68/32.03 s, but each includes startup, all prior calls, debugger
stops and file writes; these times are not a matched speed comparison. No
GPU timing is established.

## Connected fifth sigma-4 integration call

The installed FreeSurfer 8.2 `MRISintegrate` call next uses **64** gradient
averages and saves `debug0006`. A read-only
[checkpoint capture](capture_mris_register_fifth_epoch.gdb) records native
positions, normals, distance/area/correlation forces and the post-average
gradient. A separate [line-fit capture](capture_mris_register_fifth_line_fit.gdb)
records native SSE samples and float32 quadratic matrices. Both use the same
installed binary and frozen inputs as the earlier captures. The
[Python probe](probe_mris_register_second_epoch.py) starts at the independently
matched rigid surface and propagates all **five** updates in sequence. Native
intermediate coordinates, gradients and selected steps are only comparison
references, not inputs to its calculation. All reference faces have the
original ordered topology.

| Hemisphere | First four saved surfaces exact | Fifth start, normals, each force, 64-average and saved surface exact | Native and Python fifth `dt` | Maximum error |
| --- | ---: | ---: | ---: | ---: |
| LH | 106,622 / 106,622 each | 106,622 / 106,622 at each checkpoint | 11.0 | 0 |
| RH | 105,541 / 105,541 each | 105,541 / 105,541 at each checkpoint | 9.652631759643555 | 0 |

The signed spherical-face rule introduced for the fourth update also covers
the negative faces at the fifth start; no new force or step-size fix was
needed. The native reference `debug0006` hashes are
`b1a3c4b2226cdf7efa48dd09b825cda5d2f0c85e135be382d7524b99fc198f2e`
LH and `bf93a5906486176ba783acd42f54200c2b13bb39b752f60536fab9eabb7b0af6`
RH. The paired [LH](mris_register_lh_fifth_epoch_control-host.json) and
[RH](mris_register_rh_fifth_epoch_control-host.json) reports retain all input,
source, checkpoint and output SHA-256 values and all-vertex errors. Native
[LH](mris_register_lh_fifth_capture_control-host.log) and
[RH](mris_register_rh_fifth_capture_control-host.log) checkpoint logs, plus
[LH](mris_register_lh_fifth_line_fit_control-host.log) and
[RH](mris_register_rh_fifth_line_fit_control-host.log) fit logs, retain the official
values. Both captured fifth-fit cases were added to the focused regression;
the nonlinear and line-fit tests passed **10/10** remotely.

PyTorch CPU force / 64-average / line-search times for this update alone were
**16.70 / 0.39 / 0.35 s LH** and **14.52 / 0.33 / 0.31 s RH**, with four threads,
excluding I/O and the earlier updates. The connected five-update probe wall
times were 264.30/204.48 s. Official GDB capture wall times were 31.62/24.47 s,
but include startup, all preceding native updates, debugger stops and file
writes. These are not matched stage benchmarks; no GPU timing is established.

The fifth surface is still distinct from the full official `sphere.reg`. The
[fifth-to-final comparison script](compare_mris_register_fifth_to_final.py)
asserted identical face order and compared every ordered vertex against the
archived same-subject FreeSurfer final surface:

| Hemisphere | Exact final vertices | Median / 95th percentile fifth-to-final distance | Maximum coordinate difference |
| --- | ---: | ---: | ---: |
| LH | 0 / 106,622 | 2.383 / 5.058 mm | 10.648 mm |
| RH | 0 / 105,541 | 2.408 / 5.369 mm | 8.620 mm |

The paired [LH](mris_register_lh_fifth_vs_final_control-host.json) and
[RH](mris_register_rh_fifth_vs_final_control-host.json) reports include both file
hashes. The final reference hashes are
`801d345eb11c4b02e64aa453bc388f18e8cd47e8bb0beed7d75868d92276b8f8`
LH and `0e2a10093721f2f75f7460c559aa982560abd44662a9dac16865d42cb419fa8b`
RH.

## Official schedule replay through the first smoothwm update

The [schedule probe](probe_mris_register_schedule.py) continued from the
independently matched `debug0006` surface. It parsed each official
`-N 1 -W 1 -remove_negative 0` diagnostic update's averaging count and sigma,
recomputed the source sulc and atlas parameterizations when the sigma changed,
and compared the output to every saved official surface **in original vertex
order**. At the next segment boundary, it started from the independently
matched `debug0017` surface. Saved native surfaces were comparison references
and segment seeds only; within each segment, the Python state and selected
step were propagated independently. The first five updates, `debug0002` to
`debug0006`, were validated above. Thus `debug0002` through `debug0038`
contain **37 consecutive exact updates on both hemispheres**.

| Saved updates | Source / sigma | LH exact updates | RH exact updates | LH/RH Python CPU kernel seconds, excluding I/O |
| --- | --- | ---: | ---: | ---: |
| 0002–0006 | sulc / 4 | 5/5 | 5/5 | connected five-update probes 264.30 / 204.48 s wall |
| 0007–0017 | sulc / 4 | 11/11 | 11/11 | 168.12 / 160.33 s |
| 0018–0024 | sulc / 2 | 7/7 | 7/7 | 114.28 / 135.94 s |
| 0025–0031 | sulc / 1 | 7/7 | 7/7 | 114.63 / 106.84 s |
| 0032–0038 | sulc / 0.5 | 7/7 | 7/7 | 114.99 / 106.07 s |
| 0039 | smoothwm / 4 | 0/1 | 0/1 | 37.66 / 34.30 s attempted |

Every passing update matched all 106,622 LH or 105,541 RH ordered vertices,
with maximum coordinate error **0 mm**. The per-update full-precision Python
`dt`, native log's three-decimal `dt`, native surface SHA-256, exact count,
maximum error and force/average/line timing are in the compact
[LH](mris_register_lh_schedule_summary_control-host.json) and
[RH](mris_register_rh_schedule_summary_control-host.json) manifests. Full objective
trial samples, input hashes, seed hashes, reference surface hashes and timings
are in the [LH 7–17](mris_register_lh_schedule_sigma4_only_control-host.json),
[RH 7–17](mris_register_rh_schedule_sigma4_only_control-host.json),
[LH 18–39](mris_register_lh_schedule_to_sno2_control-host.json) and
[RH 18–39](mris_register_rh_schedule_to_sno2_control-host.json) reports. The
archived [LH](mris_register_lh_n1_debug_out_control-host.log) and
[RH](mris_register_rh_n1_debug_out_control-host.log) official diagnostic logs have
SHA-256 `d03b2eab334facdef1710f11d0cff1648898ddb9412637e202c245b3e8b83cde`
and `8fa7d833f346771618a2389cd16de3771c7ace31112c714ca523ba00700668b6`.
The first LH 7–17 run used an earlier native log with a different file hash;
its averaging counts were checked again against this archived official log.
The current schedule probe has SHA-256
`7cc9f6c80183fb6bed6fc0a9990770bc52dcf7ce965ee7e3693f0214e55c8a06`.
The installed official binary has SHA-256
`75137b92fcbed63b441e6214b5b923d7664f05d00c454b62da724dec26636f80`.

The installed complete `-N 1` diagnostic took **92.97 s LH** and **120.89 s
RH** with four CPU threads, including rigid and all later calls. Its
[LH](mris_register_lh_n1_native_run_control-host.log) and
[RH](mris_register_rh_n1_native_run_control-host.log) logs retain the complete diagnostic through `debug0071`. The segmented Python CPU replay has different setup and incomplete
work, so these numbers are **not** a matched full-stage speed benchmark or a
GPU result. The selected Python `dt` after update 6 is recorded at full
precision, but the official log rounds it to three decimals; exact native
floating-point `dt` has not been separately captured for updates 7–38.
The exact ordered output surface is the parity criterion for those updates.

### First differing operator at update 39

The installed binary switches from sulc to `smoothwm` curvature and atlas
frames 6/7 at this boundary. Its source recomputes the subject mean curvature
on `smoothwm`, normalizes it, reduces `l_corr` by 20, and adds a spring term of
0.5 after gradient averaging. The original Python schedule continuation still used sulc curvature,
atlas frames 3/4, the original correlation weight and no spring term. A read-only [GDB capture](capture_mris_register_sno2_first_epoch.gdb)
compared its first `MRISintegrate` call with the Python
[diagnostic probe](probe_mris_register_schedule_diagnostic.py):

| Checkpoint at update 39 | LH exact vertices / maximum error | RH exact vertices / maximum error |
| --- | ---: | ---: |
| Starting positions | 106,622 / 0 | 105,541 / 0 |
| Normals | 106,622 / 0 | 105,541 / 0 |
| Curvature input | 0 / 3.113 | 0 / 2.870 |
| Distance force | 106,622 / 0 | 105,541 / 0 |
| Area force | 106,622 / 0 | 105,541 / 0 |
| Correlation force | 0 / 2.164 | 0 / 2.359 |
| After 1,024 gradient averages, before spring | 0 / 0.0205 | 0 / 0.0293 |

Errors in this table are maximum absolute float32 values, in millimeters for
coordinates and in native numeric units for curvature and force. The first
different input is the smoothwm curvature; the first differing force is the
correlation term. The stage comparison, both predicted/native array hashes
and NPZ hashes are retained in the [LH](mris_register_lh_sno2_first_operator_control-host.json)
and [RH](mris_register_rh_sno2_first_operator_control-host.json) reports, with
[LH](mris_register_lh_sno2_capture_control-host.log) and
[RH](mris_register_rh_sno2_capture_control-host.log) GDB logs. The comparison
script is [here](compare_mris_register_sno2_checkpoint.py). The native
update-39 log prints `dt=2.827` LH and `dt=2.580` RH, whereas the incomplete
Python branch selects 0.003348 and 0.048077 respectively. The candidate
saved surface matches zero ordered vertices in either hemisphere; maximum
coordinate errors are **1.061 mm LH** and **0.672 mm RH** in the continuous
schedule run. A separate smoothwm probe now isolates this boundary, as reported below.
Continuous later updates and negative-face repair remain open.

### Paired first smoothwm update, with independent raw curvature boundary

The new `mris_register_smoothwm.py` computes three-hop quadratic-form mean
curvature from each frozen `smoothwm` in PyTorch. The probe compares **every
raw H value** with the installed FreeSurfer 8.2 array captured immediately
before normalization. Both input surface SHA-256 values and native/predicted
raw-array hashes are in the linked [LH](mris_register_lh_sno2_raw_sequential_control-host.json)
and [RH](mris_register_rh_sno2_raw_sequential_control-host.json) reports. Their source
surface hashes are `36e199e4d971a39457a4ad1d0b8eed3a84feb236ada29c05e297ffdd865a5002`
LH and `b67ceb95069d865bb11f6c2b20bac99eab9f19982b73d3c9c43735b7bedbe7ca`
RH. The native raw-array hashes are `319c755b667a04331ef0c545714aaf5897c9a04e0c08a882b69b48542f0ab39c`
and `820582a7731addfb50c7453bbce534ed4fb314a590b4749191fee06af04031a1`.

| Side | Raw H exact | Raw H median / max absolute error | PyTorch CPU raw H only |
| --- | ---: | ---: | ---: |
| LH | 14,692 / 106,622 | 1.86e-8 / 0.000165939 | 6.48 s |
| RH | 14,713 / 105,541 | 1.49e-8 / 0.000091553 | 6.59 s |

`mris_register_nonlinear.py` now applies `l_corr=0.05` in the source's
float32 arithmetic order and provides the reusable post-average spring
operator. Given the exact native post-average gradient as an operator-level
input, that PyTorch spring matches **all 106,622 LH and 105,541 RH vertices
exactly**, maximum error 0. The paired
[LH](mris_register_lh_sno2_spring_production_control-host.json) and
[RH](mris_register_rh_sno2_spring_production_control-host.json) reports include
input/output hashes and 1.01/0.87 s single-run CPU times.

The complete one-update probe started from the native-exact `debug0038`
geometry and recomputed the atlas frames 6/7, sigma-4 source/target grids,
weighted correlation force, 1,024 gradient averages, spring, line search and
saved `debug0039`. **Conditional diagnostic:** when its only injected native
intermediate was raw smoothwm H, both hemispheres matched native positions,
normals, blurred source curvature, area and correlation force, post-average
and post-spring force, and saved surface at every ordered vertex with maximum
error zero. The selected `dt` was exactly `2.826995849609375` LH and
`2.5797934532165527` RH. Paired input/output SHA-256, all checkpoint
errors and per-operation single-run CPU timings are in the
[LH](mris_register_lh_sno2_epoch_native_raw_control-host.json) and
[RH](mris_register_rh_sno2_epoch_native_raw_control-host.json) reports. This
conditional result proves the downstream operators at the same checkpoint;
it does not establish an independent full-stage match.

With the independently computed raw H, the first non-exact checkpoint is the
sigma-4 blurred source curve (maximum absolute error 3.34e-6 LH and
5.84e-6 RH). The resulting `debug0039` coordinate maximum error is
**0.000123 mm LH / 0.000355 mm RH** across all ordered vertices. Only
3,996/106,622 LH and 228/105,541 RH vertices are bitwise exact; the selected
`dt` changes to `2.8273253440856934` and `2.578439712524414`.
The [LH](mris_register_lh_sno2_epoch_sequential_control-host.json) and
[RH](mris_register_rh_sno2_epoch_sequential_control-host.json) reports retain
these full-array comparisons, all input/native-output hashes and individual
setup, force, average, spring and line-search CPU times. Those times cover
one isolated update, exclude I/O and were measured under shared-host load;
no matched native per-update timing or GPU speedup is asserted. The precise
remaining first difference is the raw H float32 neighborhood/matrix solution.

The `-N 1` diagnostic changes the official iteration count and is distinct
from the default recon-all call. Even the **complete official** `-N 1` surface
matches zero ordered vertices of the archived default official `sphere.reg`:
its median per-vertex distance is 1.602 mm LH and 1.844 mm RH. The paired
[LH](mris_register_lh_n1_final_vs_default_control-host.json) and
[RH](mris_register_rh_n1_final_vs_default_control-host.json) reports retain both
file hashes and all-vertex comparison. The default final reference hashes are
`801d345eb11c4b02e64aa453bc388f18e8cd47e8bb0beed7d75868d92276b8f8`
LH and `0e2a10093721f2f75f7460c559aa982560abd44662a9dac16865d42cb419fa8b`
RH.

## Paired default-parameter official registration reference

A fresh, isolated installed FreeSurfer 8.2 run used the **default**
`mris_register -curv -threads 4 -W 1` schedule on the same frozen sphere,
sulc, smoothwm and atlas inputs. `-W 1` saved every registration checkpoint;
the output geometry, ordered vertices and faces were then compared with the
archived default `sphere.reg` from the completed official recon-all.
The binary SHA-256 is again
`75137b92fcbed63b441e6214b5b923d7664f05d00c454b62da724dec26636f80`.

| Side | Default saved checkpoints | Installed native wall time | Final ordered vertices exact vs archived default | Max coordinate error |
| --- | ---: | ---: | ---: | ---: |
| LH | 107 | 154.01 s | 106,622 / 106,622 | 0 mm |
| RH | 97 | 147.07 s | 105,541 / 105,541 | 0 mm |

Both face arrays are equal. The newly written surface files have different
SHA-256 hashes because their headers differ; the geometry comparison above is
on decoded original-order coordinates. The paired
[LH](mris_register_lh_default_reference_check_control-host.json) and
[RH](mris_register_rh_default_reference_check_control-host.json) reports include
all frozen input, binary, native log, checkpoint log and final-output hashes.
These native wall times were observed on a shared control-host host and should not
be interpreted as a matched speed comparison with the earlier accelerator-a-1 archive.

Default registration retains up to 25 iterations per integration call,
whereas `-N 1` uses one. Their first saved update (`debug0002`) is identical
at every vertex on both sides, but `debug0003` has zero exactly matching
vertices between the two official schedules (maximum difference 4.347 mm LH,
3.176 mm RH). This is a control-flow difference, so the earlier 37 exact
`-N 1` updates do not establish default final parity.

The initial default LH PyTorch continuous replay matched native `debug0002`
at all 106,622 ordered vertices, selected `dt=70.93703089944263`. At
`debug0003`, its original implementation selected `dt=198.3500518798828`
versus the native log's rounded `196.340` and differed by a maximum
0.0734673 mm. The [initial report](mris_register_lh_default_first_difference_control-host.json)
retains paired input, output and candidate-array hashes plus timings.
Read-only [GDB checkpoints](mris_register_lh_default_second_capture_control-host.log)
showed the first different operator: native `MRISintegrate` projects only once
at the start of a call, whereas the candidate projected again on every
iteration. At the second iteration, native pre-force positions equal the
saved `debug0002` at all vertices; the extra candidate projection matched
only 101,966/106,622 vertices with maximum start error 7.63e-6 mm.
The GDB capture took 21.81 s to reach the second update on the shared host.

The registration force APIs now accept an explicit `project=False` for
iterations inside the same `MRISintegrate` call, and the default schedule
probe tracks those call boundaries. A fixed conditional second-update probe
used the exact native `debug0002` coordinates as its starting array; those
coordinates had already been produced independently and matched all ordered
vertices in the first update. **Every** checkpoint at `debug0003` then matched
all 106,622 vertices exactly: starting positions, normals, distance, area and
correlation gradients, 16,384-average gradient, and saved surface. Its
`dt=196.33958435058594` agrees with the installed log's rounded value.
The paired [before](mris_register_lh_default_second_operator_initial_control-host.json)
and [after](mris_register_lh_default_second_operator_fixed_control-host.json)
reports retain all native/predicted array hashes and per-operation CPU times.
The post-fix LH 16,384-average kernel took 89.57 s under shared-host load.

The same paired default RH second update also matched **all 105,541 ordered
vertices exactly** at each GDB checkpoint and saved `debug0003`, selecting
`dt=135.03231811523438` (installed log: `135.032`). Its
[checkpoint report](mris_register_rh_default_second_operator_fixed_control-host.json)
and read-only [capture](mris_register_rh_default_second_capture_control-host.log)
retain source, reference and predicted hashes, each force/gradient comparison
and individual CPU timings. This confirms two default updates per side, not
the full default sequence. Focused nonlinear and line-search tests passed
10/10 in the same remote PyTorch environment after the projection-state fix.

The next bounded LH default continuation started from the independently
matched `debug0003` geometry and stayed inside the same 16,384-average
integration call. It selected `dt=64.76956939697266` and matched the installed
`debug0004` at **106,622/106,622 ordered vertices**, maximum error 0 mm.
The saved Python candidate surface was reopened and also matched all vertices
and faces exactly. Its [report](mris_register_lh_default_epoch0004_control-host.json)
records the frozen input, native output, predicted array and serialized
candidate SHA-256 values. Single-run CPU times excluding I/O were 13.93 s
force, 89.42 s averaging and 0.33 s line search under shared-host load.
The next bounded LH continuation used the reopened independent candidate
`debug0004` and stayed in the same 16,384-average call. It selected
`dt=170.02264404296875`; the installed `debug0005` and the saved Python
candidate match at **106,622/106,622 ordered vertices and all faces**, maximum
error 0 mm. The [epoch-5 report](mris_register_lh_default_epoch0005_control-host.json)
includes input/native-output, predicted-array, serialized-output SHA-256,
readback comparison and per-operation time: force 16.19 s, averaging 105.22 s,
line search 0.38 s excluding I/O. This run shared control-host CPU with the RH
pial validation; its timing is an observation, not a matched speed ratio.

### Default sulcal continuation and first smoothwm difference

The bounded schedule probe then propagated each saved candidate surface to the
next native checkpoint and stopped at the first mismatch. LH `debug0006` starts
a new 4,096-average integration call; RH `debug0004` stays inside the original
16,384-average call. The RH `debug0003` seed was the installed surface whose
complete predicted coordinate array had independently matched at every vertex;
all later segment seeds were serialized Python candidate surfaces. The probe
reopened each seed, retained native face order, and recorded every full-precision
`dt`, comparison, input/reference/candidate SHA-256 and CPU force, averaging,
line-search and scale-setup time in its JSON reports. Native `dt` lines are
stderr-buffered ahead of the stdout scale log; the probe now assigns sigma by
the **ordered saved checkpoint writes**. Its scale map covered all 107 LH and
97 RH installed saved checkpoints, including the actual sulc-to-smoothwm switch.

| Hemisphere | Continuous exact sulc checkpoints | Exact ordered vertices at every update | Last sigma-0.5 checkpoint |
| --- | --- | ---: | --- |
| LH | `debug0002`–`debug0056` (55 updates) | 106,622 / 106,622; maximum error 0 mm | `debug0056` |
| RH | `debug0002`–`debug0055` (54 updates) | 105,541 / 105,541; maximum error 0 mm | `debug0055` |

The newly paired LH segment reports are [0006](mris_register_lh_default_epoch0006_control-host.json),
[0007–0009](mris_register_lh_default_epoch0007_0009_control-host.json),
[0010–0012](mris_register_lh_default_epoch0010_0012_control-host.json),
[0013–0026](mris_register_lh_default_epoch0013_0026_control-host.json),
[0027–0033](mris_register_lh_default_epoch0027_0033_control-host.json),
[0034–0041](mris_register_lh_default_epoch0034_0041_control-host.json),
[0042–0048](mris_register_lh_default_epoch0042_0048_control-host.json) and
[0049–0056](mris_register_lh_default_epoch0049_0056_control-host.json).
The RH reports are [0004](mris_register_rh_default_epoch0004_control-host.json),
[0005](mris_register_rh_default_epoch0005_control-host.json),
[0006–0009](mris_register_rh_default_epoch0006_0009_control-host.json),
[0010–0026](mris_register_rh_default_epoch0010_0026_control-host.json),
[0027–0033](mris_register_rh_default_epoch0027_0033_control-host.json),
[0034–0041](mris_register_rh_default_epoch0034_0041_control-host.json),
[0042–0048](mris_register_rh_default_epoch0042_0048_control-host.json) and
[0049–0055](mris_register_rh_default_epoch0049_0055_control-host.json).
The 53 LH and 52 RH newly paired numbered updates have no gaps and every
comparison is exact. The last serialized sulc seeds were separately reopened and matched
all ordered vertices and faces exactly on both sides; their file hashes are in
the [readback report](mris_register_default_sulc_last_seed_readback_control-host.json).
These CPU times were observed under varying shared control-host load and are not an
end-to-end matched benchmark.

The next installed update changes the source from sulc to `smoothwm` curvature:
LH `debug0057`, RH `debug0056`. The same `smoothwm` files and the previously
captured native raw-curvature arrays were paired by SHA-256. Starting from the
independent exact last sulc candidate, substituting **only native raw H** into
the PyTorch smoothwm force, 1,024-average, spring and line search gives the
installed saved output at every ordered vertex: LH 106,622/106,622 with
`dt=2.721714973449707`, RH 105,541/105,541 with
`dt=2.542891502380371`; maximum error 0 mm. The conditional reports are
[LH](mris_register_lh_default_epoch0057_native_raw_control-host.json) and
[RH](mris_register_rh_default_epoch0056_native_raw_control-host.json).

Replacing that one input with the independent PyTorch raw H identifies the
first native-free default output difference. LH `debug0057` matches
106,455/106,622 vertices exactly, with maximum coordinate error
7.62939453125e-6 mm and unchanged `dt`. RH `debug0056` matches 253/105,541
exactly; 12,639 vertices are within 1e-5 mm, the maximum coordinate error is
3.452301025390625e-4 mm, and `dt` becomes 2.544266700744629. The paired
independent-input reports are [LH](mris_register_lh_default_epoch0057_independent_raw_control-host.json)
and [RH](mris_register_rh_default_epoch0056_independent_raw_control-host.json).
This conditional intervention located the initial difference at the raw
`smoothwm` curvature input. The RH correction below supersedes that failed
independent-input trial; later default-stage and final `sphere.reg` parity
remain open.

## RH correction and remaining registration work

The following source SHA-256 values identify the pre-correction probe:
`9944484056d08df18c95fef8dc70262c18bd44e2695fdea950ab93d812e16b1c`
for `mris_register_smoothwm.py`,
`4b9f8faff6da19abb6fb6e1c687ad7d631d38630b47cf34018b9dc583967b9e9`
for `mris_register_nonlinear.py`, and
`29172196286139b740391a2995f764a95199e6a2ac7b6aea37d3ecd5cf7b840f`
for the bounded default schedule probe, and
`e5d40a7279c6d3bec240b76f9c001c052d0d7424b8a98611f217da8a4ee3d705`
for the conditional smoothwm probe. Input and native/predicted output SHA-256
values are recorded per hemisphere in the linked JSON reports.

The [RH raw H SVD audit](MRIS_REGISTER_RH_RAW_H_SVD_MATCH.md) locates the
first numerical difference at the 3×3 SVD inverse: both selected vertices
match the installed neighborhood, tangent axes, design, Gram and RHS bitwise.
The source-order VNL inverse plus FreeSurfer's 2×2 Hessian eigenvalue mean
now matches **105,541/105,541** RH raw H values, identical array SHA-256.
From the independently exact last sulc seed, the corrected first RH default
smoothwm update (`debug0056`) matches **105,541/105,541** native saved vertices
with maximum error 0 mm and identical `dt=2.542891502380371`.

The corrected raw H fit uses CPU Numba inside the Python/PyTorch stage.
The [LH continuation audit](MRIS_REGISTER_LH_SMOOTHWM_BOUNDARY.md) matches
all 106,622 raw H values and all 45 consecutive normal smoothwm surfaces,
`debug0057`–`debug0101`, at every ordered vertex and face. The stage schedule
is read from the native log; independent stopping remains unverified.

### RH `debug0058` line-search correction

The prior RH `debug0056`–`debug0057` continuation was exact but selected
`dt=1.7613636255264282` at `debug0058`, versus the installed
`1.8571428060531616`. A accelerator-a-1 FreeSurfer 8.2.0-1 GDB run captured the
three native bracket SSEs at the actual fit, using the same binary, input
sphere, atlas and four threads. The [capture and comparison](mris_register_rh_default_epoch0058_native_line_fit_accelerator-a-1.json)
retain the native matrix, inverse, RHS, coefficients, input hashes, source
script and full log hashes. The [raw fit log](mris_register_rh_default_epoch0058_native_line_fit_accelerator-a-1.log)
and [compressed FreeSurfer SSE-term log](mris_register_rh_default_epoch0058_native_sse_terms_accelerator-a-1.log.gz)
are included for audit.

| Trial dt | Native SSE | Old Python SSE | Corrected Python SSE | Native/corrected float32 SSE |
| ---: | ---: | ---: | ---: | ---: |
| 0.2409516472 | 1,425,277.594681 | 1,425,277.602511 | 1,425,277.594677 | 1,425,277.625 |
| 0.4819032944 | 1,424,246.309728 | 1,424,246.317522 | 1,424,246.309692 | 1,424,246.250 |
| 0.7228549416 | 1,423,340.796671 | 1,423,340.804469 | 1,423,340.796641 | 1,423,340.750 |

The old middle score rounded to `1,424,246.375` in float32, one 0.125 ULP
above the native value. FreeSurfer's SSE-term log attributes almost all of
the roughly 0.0078 double-score difference to the spring term. Its area
ratio is stored as float32 before that term accumulates; the Python probe
used a double ratio. Casting that one ratio to float32 changes neither the
previously exact force nor its float32 `dist_scale`, but aligns all three
stored scores and restores the native quadratic candidate. This is a
source-derived and empirically checked precision fix, not a change to the
matrix inverse order. The [fixed RH report](mris_register_rh_default_epoch0056_0058_area_scale_fixed_accelerator-a-1.json)
shows `debug0056`, `debug0057`, and `debug0058` each match **105,541/105,541
ordered native vertices**, maximum coordinate error 0 mm, with the same
native-selected step at each update. The [resumed RH report](mris_register_rh_default_epoch0059_0066_area_scale_fixed_accelerator-a-1.json)
then matches all eight consecutive `debug0059`–`debug0066` checkpoints at
105,541/105,541 ordered vertices each, with maximum coordinate error 0 mm.
The fixed probe uses the native raw H array as a controlled input; its
independent production computation was separately shown to match the same
full-array hash. The integration schedule is still read from the native log.

The earlier [unfixed continuation](mris_register_rh_default_epoch0056_0061_divergence_control-host.json)
shows why this 0.125 ULP mattered: its coordinate error reached 0.1981 mm
by `debug0061`. It is retained as pre-fix evidence.

### RH `debug0067`–`debug0097`: complete normal smoothwm updates

The [continuous continuation](mris_register_rh_default_epoch0067_0083_first_difference_accelerator-a-1.json)
resumes from the exact `debug0066` checkpoint and matches each of the 16
`debug0067`–`debug0082` saved surfaces at **105,541/105,541 ordered vertices**,
maximum coordinate error 0 mm. This crosses the sigma-2 switch at
`debug0077`. The [sigma-2 resume check](mris_register_rh_default_epoch0083_sigma2_resume_accelerator-a-1.json)
reconstructs that switch from the exact `debug0076` seed and reproduces the
continuous `debug0083` trial SSEs and selected step exactly. The probe now
restores this stage state when resuming after a sigma switch.

At `debug0083`, the [native GDB fit and comparison](mris_register_rh_default_epoch0083_native_line_fit_accelerator-a-1.json)
show the first new decision mismatch. The first bracket SSE is
1,333,351.6875719244 native versus 1,333,351.6874348717 Python; these
round to 1,333,351.75 and 1,333,351.625 in float32. The middle and upper
bracket SSEs round identically. Native selects `dt=0.04196085984684431`;
Python selects `0.04296875`. The Python saved surface matches 488/105,541
vertices, with maximum error 0.0008392334 mm. Applying the **native** step
to the Python projected surface and gradient produces the native checkpoint
at all 105,541 vertices exactly. This isolates the observed output mismatch
to the selected step; the score terms were audited next.

The [native SSE log](mris_register_rh_default_epoch0083_native_sse_terms_accelerator-a-1.log.gz)
and [source-order summation check](mris_register_rh_default_epoch0083_sse_order_accelerator-a-1.log)
localize the small score difference to percentage area, spring and correlation
terms. Sequential Python sums of the already computed per-face and per-vertex
terms did not close it. The native-step replay is a diagnostic, not an
independent registration result.

FreeSurfer's `INTEGRATION_PARMS::l_corr` is a `float`; `MRIScomputeSSE`
promotes that stored value to double when weighting the term. The probe had
multiplied by Python's double `0.05`. Using the float32-stored coefficient
changes the three `debug0083` bracket scores to 1,333,351.6876297435,
1,333,308.4455033988 and 1,333,274.7667974494. Their float32 values
now match the three native fit inputs exactly; the remaining double-score
differences do not alter the decision. The [independent corrected run](mris_register_rh_default_epoch0083_0097_lcorr_f32_accelerator-a-1.json)
selects native `dt=0.04196085984684431` at `debug0083`, then matches every
`debug0083`–`debug0097` surface at **105,541/105,541 ordered vertices and all
faces**, maximum coordinate error 0 mm. No native step is injected. Its
15 decisions and complete saved-surface comparisons match the earlier
native-step diagnostic trajectory.

The weight correction cannot change the 27 already matched `debug0056`–
`debug0082` updates: the [saved-sample attestation](mris_register_rh_lcorr_f32_prior_epochs_attestation.json)
checks all 243 prior line-search samples and finds no change in any float32
fit input or pairwise score ordering. This is a source-equivalent decision
audit, not a repeated 27-update run. Thus the complete bounded RH smoothwm
path, `debug0056`–`debug0097`, matches every saved vertex and face.

The [native final-surface check](mris_register_rh_final_surface_vs_debug0097_accelerator-a-1.json)
shows that native `rh.sphere.reg` and `debug0097` have identical coordinate
and face bytes. Their [volume geometry fields](mris_register_rh_final_volume_info_compare_control-host.json)
also match; only trailing provenance bytes differ. The native RH run had
zero negative triangles at the final repair call, which makes that loop a
no-op. The bounded Python trajectory therefore reaches the official RH
final geometry on this subject. The [LH fold and overlap report](MRIS_REGISTER_LH_SMOOTHWM_BOUNDARY.md)
now covers its six additional updates and final repair at exact ordered
vertices. The source-stopping audit below resolves the average/sigma decisions
on the saved trajectory; a continuous native-free registration invocation,
vertex/ROI metrics, and connected T1 reconstruction still require acceptance.


## Source-derived smoothwm stopping and sigma schedule

Pinned FreeSurfer `MRISintegrate` compares
`100 × (starting_sse − ending_sse) / ending_sse` with
`tol × sqrt((navgs + 1) / 1024)`, and advances when the value is smaller,
`dt` is zero, or 25 updates have run at the scale. Smoothwm registration uses
`tol=1.0`; LH fold cleanup uses `tol=0.1`. The
[Python scheduler](../../../src/fnit/recon_all/mris_register_schedule.py)
advances `1024 → 256 → 64 → 16 → 4 → 1 → 0`, then switches to the next
blur sigma `4 → 2 → 1 → 0.5`. After the last sigma, the source enters
64-average fold cleanup only if negative faces remain.

The [saved-score audit](mris_register_source_schedule_audit.json) reads the
already exact bilateral Python trajectory reports. It uses their **selected
float32 trial SSE**, not the older float64 reevaluation of a reopened native
surface. It predicts all 50 LH and 41 RH saved update states, including
whether the next update projects at a new scale; both final updates
independently satisfy the source stopping rule. The LH/RH fold branch counts
(193/0) in this retrospective audit come from the pinned native logs. The
[reproducible auditor](experimental/audit_mris_register_source_schedule.py)
records every input-report hash and predicted decision. No deformation was
rerun for this 91-decision audit.

The existing [registration probe](probe_mris_register_sno2_epoch.py) now has
an opt-in `--source-schedule` mode. It computes the next average count,
sigma, projection boundary and fold branch from its own scores and current
mesh. A bounded LH run started from the native `debug0056` file, whose
ordered geometry had separately matched the Python sulc seed exactly. It
used the independently computed raw smoothwm curvature and matched native
`debug0057`–`debug0060` at all 106,622 vertices; the three new decisions were
`1024 → 256 → 64 → 64`. See the
[LH runtime report](mris_register_lh_source_schedule_first4_control-host.json).
Three focused scheduler tests passed at this intermediate checkpoint. The
continuous bilateral run and final one-call validation are reported below.
The older bounded geometry replays retain their logged schedule by default.

## Continuous source-scheduled smoothwm stage and callable API

The [`run_register_smoothwm` API](../../../src/fnit/recon_all/mris_register_smoothwm_run.py)
now takes an ordered conventional `sphere`, `smoothwm`, an exact sulc-pass
seed, and the 2.86 MB folding-atlas TIFF. It independently computes raw
`smoothwm` curvature, selects every scale from its own SSE, runs the nonlinear
updates, performs negative-face repair, and writes the final sphere and
volume geometry. It invokes no FreeSurfer executable. The same module has a
`python -m fnit.recon_all.mris_register_smoothwm_run` CLI; the caller supplies
the completed sulc update count as `--seed-iteration`.

On the frozen `fs_sub01`, the independent raw-curvature arrays matched the
installed FreeSurfer float32 arrays by SHA-256 and at every vertex. Starting
from the saved LH `debug0056` and RH `debug0055` sulc meshes, whose geometry
was separately reproduced by Python, a **single continuous** source-scheduled
run matched all 51 LH and 42 RH official smoothwm/fold checkpoints at every
ordered vertex, with maximum error 0 mm. Both schedules terminated from the
Python SSE rather than an official log. The [LH trajectory](mris_register_lh_source_schedule_full_independent_control-host.json)
and [RH trajectory](mris_register_rh_source_schedule_full_independent_accelerator-a-1.json)
retain every selected step, scale decision, native reference hash, coordinate
hash and timing. Native checkpoint meshes were read for comparison only.

Separate [LH](mris_register_lh_smoothwm_api_control-host.json) and
[RH](mris_register_rh_smoothwm_api_control-host.json) calls of the installed API
computed raw curvature again and wrote final `sphere.reg` files. The
[independent LH comparator](mris_register_lh_smoothwm_api_final_compare_control-host.json)
and [RH comparator](mris_register_rh_smoothwm_api_final_compare_control-host.json)
found 106,622/106,622 and 105,541/105,541 exact ordered vertices,
identical ordered faces and volume geometry, and all 51/42 selected steps
identical to the separate continuous probes. LH ran all 102 overlap repair
updates; RH had no negative triangles. File hashes differ because the Python
writer uses its own provenance stamp.

The two API calls took 876.99 s LH and 797.29 s RH on control-host, including input
and output I/O and with four PyTorch CPU threads. Their integration portions
were 860.03 s and 782.46 s; final repair took 2.86 s and 0.01 s. The
separate LH checkpoint probe spent 756.39 of 816.29 s of recorded update
compute in force construction. Earlier official logs report approximately
154.08 s LH and 146.88 s RH for **complete** native registration, which also
includes rigid and sulc passes. These are different stage boundaries and
shared-host load conditions; they do not establish a matched speed ratio.

A [paired first-force CPU/CUDA0 diagnostic](mris_register_lh_first_force_cpu_cuda0_accelerator-a-1.json)
on accelerator-a-1 with float32 and TF32 enabled took 32.76 s CPU and 33.58 s CUDA,
excluding tensor transfer. The maximum force difference was 1.55e-5;
106,615/106,622 vertices were within 1e-5. This direct CUDA substitution
has not been adopted. Force construction repeatedly builds ordered topology
and is the measured optimization target.

This isolated validation starts from a sulc seed. The independent sulc runner
and one-call `sphere`-to-`sphere.reg` result are reported below. The upstream
topology, white/pial and whole-subject acceptance gates remain open.

## Source-scheduled sulc stage and sphere-to-registration connection

The source rule for the default sulc pass starts with one sigma-4
16,384-average epoch, then a normal 1,024-average epoch at sigma 4 and
normal epochs at sigmas 2, 1 and 0.5. Each average count divides by four
through zero; the native tolerance is `0.5 × sqrt((averages + 1) / 1024)`.
The [`next_sulc_scale` implementation](../../../src/fnit/recon_all/mris_register_schedule.py)
uses its own selected Python SSE and step, including the 25-update cap.
The [reproducible source audit](experimental/audit_mris_register_sulc_source_schedule.py)
reads the newly published exact sulc reports from this directory. Its
[result](mris_register_sulc_source_schedule_audit.json) matches all 53 LH
and 52 RH saved post-initial decisions, including next-scale projection and
termination. These reports cover `debug0004`–`debug0056` LH and
`debug0004`–`debug0055` RH; the initial two nonlinear updates and rigid
search have their separate exact checks above.

The new [`run_register_sulc` API and CLI](../../../src/fnit/recon_all/mris_register_sulc_run.py)
starts from `sphere`, `smoothwm`, `sulc`, and the external folding-atlas TIFF.
It performs the independent rigid search and every source-scheduled sulc
update without reading an installed FreeSurfer mesh or log. Its bilateral
[LH](mris_register_lh_sulc_api_control-host.json) and
[RH](mris_register_rh_sulc_api_control-host.json) control-host calls selected 55 and 54
updates and terminated at native iteration numbers 56 and 55. The
[LH final comparator](mris_register_lh_sulc_api_final_compare_control-host.json)
and [RH final comparator](mris_register_rh_sulc_api_final_compare_control-host.json)
found all 106,622 and 105,541 ordered vertices exact, all ordered faces and
volume geometry equal, and all 53/52 previously saved selected steps equal.
The file hashes differ due to the Python writer's provenance stamp.

The [LH](mris_register_lh_stage_connection_attestation_control-host.json) and
[RH](mris_register_rh_stage_connection_attestation_control-host.json) connection
checks prove the new sulc seeds have the same ordered geometry, face order,
source `sphere`/`smoothwm`/atlas file hashes and iteration numbers as the
inputs of the previously exact smoothwm API. The one-call
[`run_register_sphere` wrapper](../../../src/fnit/recon_all/mris_register_run.py)
has now also passed a fresh bilateral numerical run, detailed below. It is
not a T1-to-metrics reconstruction.

The standalone sulc calls took 1,465.67 s LH and 1,336.97 s RH on control-host
with four PyTorch CPU threads; the prior standalone smoothwm calls took
876.99 s and 797.29 s. Their separate-call sums are 2,342.67 s LH and
2,134.26 s RH. The earlier installed FreeSurfer logs report about 154.08 s
LH and 146.88 s RH for the complete native registration. These observations
were under different shared-host loads and the Python values include two
separate process setups, so they are not a controlled full-stage benchmark.
The first four 16,384-average updates alone took 477.96 s LH and 429.03 s
RH. Gradient averaging and repeatedly rebuilt force topology are the next
speed targets; no registration-wide GPU speedup is established.

## Fresh one-call bilateral registration with exact CPU gradient averaging

The production sulc and smoothwm runners now call
[`average_gradients_exact_cpu`](../../../src/fnit/recon_all/mris_register_average_numba.py).
Its Numba loop retains each ordered neighbor and float32 addition/multiplication.
On a saved real LH gradient, 16,384 iterations matched the PyTorch source-order
result at all 106,622 vertices, with zero maximum error, in 28.67 s versus
98.16 s for the previous PyTorch loop (3.42× for this kernel alone). The
1,024-iteration checks were also exact at all 106,622 LH and 105,541 RH
vertices; the production wrapper passed an additional exact LH 64-iteration
check. The [benchmark script](experimental/benchmark_mris_register_average_numba.py)
and [16,384-iteration report](mris_register_lh_average_numba_16384_control-host.json)
retain input shape, timings and error. These are CPU measurements on control-host;
no GPU speed claim follows from them.

A fresh **one-call** `run_register_sphere` execution then started from the
unchanged conventional `sphere`, `smoothwm`, `sulc` and external folding atlas
for each hemisphere. Both independently generated temporary sulc seeds were
observed before cleanup to have byte hashes identical to the previously
validated seeds: LH `26c48f477b9596c6128cf2d138d0590379e23979cde61e2186db0914e39a46ab`,
RH `92b6e45701cc946a5274750f9bccd6743098dcb252ea7458e86855107739de96`.
The snapshot wrapper used for this execution predates the repository wrapper's
report-only SHA fields; its numerical stage calls are identical by direct
source diff. Consequently the machine comparison records the seed-hash field
as `null`, rather than claiming it read a missing report field.

A new unmodified FreeSurfer 8.2 `mris_register -curv -threads 4` run on the
same inputs and control-host host generated fresh LH/RH references. They reproduced
the archived official ordered geometry and faces exactly. The independent
[full-stage comparator](experimental/compare_mris_register_full_api.py)
checked the Python output against those fresh references and the previously
validated PyTorch stage schedules:

| Hemisphere | Exact ordered vertices | Faces / volume geometry | Sulc / smoothwm selected steps | Python one-call | Fresh native |
| --- | ---: | --- | --- | ---: | ---: |
| LH | 106,622 / 106,622 | equal / equal | 55 / 51, all equal | 2,029.73 s | 157.62 s |
| RH | 105,541 / 105,541 | equal / equal | 54 / 42, all equal | 2,037.06 s | 161.62 s |

The Python sulc and smoothwm portions took 1,102.74/926.98 s LH and
1,273.81/763.09 s RH. Full reports are
[LH](mris_register_lh_numba_full_control-host.json) and
[RH](mris_register_rh_numba_full_control-host.json); exact comparisons are
[LH](mris_register_lh_numba_full_compare_control-host.json) and
[RH](mris_register_rh_numba_full_compare_control-host.json). The native
[LH](mris_register_native_pair_lh_control-host.log) and
[RH](mris_register_native_pair_rh_control-host.log) logs retain the commands and
`/usr/bin/time` wall times. Both Python jobs ran concurrently, and the native
controls ran during those jobs on a shared host. These numbers are descriptive
same-host observations, not controlled paired speed ratios. The current Python
registration is substantially slower than native despite the exact averaging
kernel improvement.

A [real LH force profile](mris_register_lh_force_profile_control-host.json) found
8.03 s for one distance-force call and 5.22 s for one area-force call. The
distance force repeatedly rebuilt vertex normals (5.18 s), three-hop
neighbor counts (1.42 s) and ordered neighbors (1.30 s) even though face
order is fixed. The exact cached-force implementation and its full validation
follow. This one-call registration still starts from supplied surface inputs;
full topology repair, white/pial placement and T1-to-metrics native-free
orchestration remain open.

## Exact ordered-topology cache and accelerated one-call validation

The sulc and smoothwm runners now build one immutable
`RegistrationForceCache` per hemisphere from the fixed input faces and
original surface. It holds ordered neighbors, ordered incident faces/corners,
original distances and areas, the three-hop neighbor count, and the original
sphere area. Every force call still performs the same float32 additions in
the same vertex/face order; only static topology construction is reused.
Default uncached force APIs remain callable.

The [bilateral real-mesh benchmark](experimental/benchmark_mris_register_force_cache.py)
compared the old and cached normal, distance-force and area-force arrays
with projection enabled and disabled. **Every vertex in every array matched
exactly** (106,622 LH; 105,541 RH; maximum absolute error 0). For one
normal+distance+area call on control-host, the LH old/cached times were
17.74/0.165 s without projection and 17.58/0.129 s with projection; RH was
17.21/0.131 s and 18.24/0.154 s. Building each cache took 7.55 s LH and
7.82 s RH. These are isolated CPU calls under shared-host load, not complete
stage speed ratios. The [LH](mris_register_lh_force_cache_control-host.json) and
[RH](mris_register_rh_force_cache_control-host.json) reports retain all comparisons.

A first integrated run generated byte-identical bilateral sulc seeds, then
stopped in smoothwm because `face_area_normals` was mistakenly removed from
its imports. The import was restored, `pyflakes` checked all four registration
modules without errors, and the corrected **fresh one-call** run completed
both hemispheres from the original `sphere`/`smoothwm`/`sulc`/atlas inputs.
The independent [full-stage comparator](experimental/compare_mris_register_full_api.py)
used the same fresh unmodified FreeSurfer 8.2 references as above:

| Hemisphere | Exact ordered vertices | Faces / volume geometry | Seed SHA / all sulc and smoothwm steps | Cached Python total | Previous Python total | Fresh native |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| LH | 106,622 / 106,622 | equal / equal | equal / equal | 372.85 s | 2,029.73 s | 157.62 s |
| RH | 105,541 / 105,541 | equal / equal | equal / equal | 347.36 s | 2,037.06 s | 161.62 s |

The cached sulc/smoothwm portions took 283.70/89.10 s LH and
267.49/79.81 s RH, including their stage I/O and setup. Relative to the
previous one-call Python observations, total time fell by factors of
5.44 LH and 5.86 RH. Runs occurred at different times on a shared node, so
those factors are descriptive. Both versions executed each hemisphere in a
separate concurrent process. Current cached Python registration remains
slower than the native program; the native controls ran while the earlier
Python pair was active, so these measurements do not establish controlled
Python/native speed ratios. The complete [LH](mris_register_lh_force_cache_full_control-host.json)
and [RH](mris_register_rh_force_cache_full_control-host.json) reports include input,
seed and output SHA-256 values, full selected-step trajectories and stage
times. The strict comparisons are [LH](mris_register_lh_force_cache_full_compare_control-host.json)
and [RH](mris_register_rh_force_cache_full_compare_control-host.json).

This exact accelerated stage still starts from supplied conventional surface
inputs and runs its numerical core on CPU PyTorch/Numba. It is not a complete
GPU `recon-all` or a T1-to-metrics subject reconstruction.
