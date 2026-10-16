# Conventional `mris_sphere` parity boundary

The frozen `fs_sub01` commands are `mris_sphere -threads 4 -seed 1234
../surf/{lh,rh}.inflated ../surf/{lh,rh}.sphere` from FreeSurfer 8.2.0,
source commit `d932c45`. This nonquick stage reads the matching `smoothwm`
surface for its original metric. The Python projection is in
`sphere_standard_python.py`; the full sparse original metric sampler is in
`sphere_standard_metric.py`. The bounded first `MRISunfold` epoch is in
`sphere_standard_unfold.py` and `sphere_standard_line_search.py`. The first
one-ring nonlinear-area fold-repair update is in `sphere_standard_nonlinear.py`.

| Fixed projection checkpoint | LH | RH |
| --- | ---: | ---: |
| `before`: exact ordered vertices | 106,622 / 106,622 | 105,541 / 105,541 |
| `after`: exact ordered vertices | 106,622 / 106,622 | 105,541 / 105,541 |
| `after`: exact ordered faces | 213,240 / 213,240 | 211,078 / 211,078 |
| Python scale + projection, control-host | 0.0206 s | 0.0148 s |

The metric sampler constructs the source-order 1–7 ring candidates on the
frozen `smoothwm`, carries FreeSurfer's seeded VNL random stream through all
vertices, keeps the first three complete rings, samples eight entries per
ring from rings 3–7, and performs FreeSurfer's serial reciprocal-distance
averaging. The third ring intentionally occurs both in the complete prefix
and in the sampled suffix. The CSR rows preserve those duplicates.

An isolated native `FS_MEASURE_DISTANCES` diagnostic wrote the complete
**post-averaging** distance table at four decimal places immediately after
the first `MRISsampleDistances` call, before any `MRISunfold` optimization.
The Python and native tables have the same number of entries, and every
entry agrees at all four printed decimal places in source row order:

| Whole metric validation on control-host | LH | RH |
| --- | ---: | ---: |
| Vertices | 106,622 | 105,541 |
| Sampled entries | 8,268,920 | 8,183,354 |
| Native printed distances matching | 8,268,920 / 8,268,920 | 8,183,354 / 8,183,354 |
| Max absolute error against printed values | 0.000050 mm | 0.000050 mm |
| Native isolated sample + text dump | 14.33 s | 14.44 s |
| Python topology / sample incl. JIT / average incl. JIT | 4.49 / 10.98 / 0.66 s | 3.42 / 8.52 / 0.51 s |

The native diagnostic logs use four decimal places, so the complete table
check establishes agreement at that precision rather than bitwise identity.
Separate native `-v` logs include neighbor IDs and six-decimal distances:
LH vertex 0 and vertex 1 each match all 79 ordered IDs and 79 printed
distances; RH vertex 0 matches all 76 ordered IDs and 76 printed distances.
The whole-table dump omits IDs, so ordered IDs have only been checked for
those three queried vertices. The native time includes writing 8 million
text lines, whereas Python time excludes that write; these are not paired
sampler speed measurements. The sampler currently runs on CPU through
Numba, not GPU.

Structured reports are [LH](standard_sphere_full_metric_lh.json) and
[RH](standard_sphere_full_metric_rh.json). Earlier candidate-ring diagnostics
([LH](standard_sphere_metric_rings_lh.json),
[RH](standard_sphere_metric_rings_rh.json)), seeded neighbor comparisons
([LH](standard_sphere_vnl_seed_lh.json),
[RH](standard_sphere_vnl_seed_rh.json)), and the
[LH continued RNG](standard_sphere_vnl_continuity_lh.json) record the
pre-averaging checks. Native large text tables remain under
`$W/tessellate/sphere/standard_{lh,rh}/full_metric_native/distance.log`.

## Bounded first `MRISunfold` epoch

Separate isolated native probes used copied `fs_sub01` inputs, one thread,
seed 1234, `-a 0 -n 1 -p 1`, and one saved coordinate snapshot per update.
The initial negative-area repair took five updates on LH and zero on RH;
Python starts from the native checkpoint **after** that repair. The original
`smoothwm` copies have the same SHA256 as the read-only official inputs.
Each [LH](standard_sphere_first_epoch_lh.json) and
[RH](standard_sphere_first_epoch_rh.json) gradient report includes SHA256 for
the `smoothwm`, before/after snapshots, and native verbose log.

The current spherical distance used in the force has its own six-decimal
native log check ([LH](standard_sphere_current_arc_lh.json),
[RH](standard_sphere_current_arc_rh.json)): LH vertex 0 matches 79/79
ordered distances and RH vertex 0 matches 76/76. The Numba implementation must accumulate the norm and dot
product in double precision, then perform the C++ float division before the
angular approximation. This is a CPU float32/float64 arithmetic match, not a
GPU benchmark.

| First epoch, whole surface | LH | RH |
| --- | ---: | ---: |
| Vertices / negative faces | 106,622 / 2,305 | 105,541 / 491 |
| Python gradient L2 / native printed | 449.855171 / 449.855 | 10.984243 / 10.984 |
| Python maximum force / native printed | 147.603461 / 147.603 | 2.153895 / 2.154 |
| Python starting SSE / native printed | 2,627,487.011 / 2,627,487.05 | 2,660,370.600 / 2,660,370.64 |
| Python and native selected candidate | 2 / 2 | 4 / 4 |
| Independent selected `dt` / native printed | 0.014806829 / 0.015 | 17.633028 / 17.633 |
| Exact updated coordinate components | 319,863 / 319,866 | 316,581 / 316,623 |
| Updated vertices within 0.00001 mm | 106,622 / 106,622 | 105,541 / 105,541 |
| Maximum vertex error | 0.00000381 mm | 0.00000769 mm |
| Matrix / gradient / warm line search CPU time | 16.09 / 2.71 / 0.76 s | 16.23 / 2.71 / 0.66 s |

The independent search evaluates all native-form trial scores. Its float32
3×3 quadratic fit reproduces the native printed `(a,b,c)` exactly: LH
`(21495808,-306176,2627632)`, RH `(436,-7688,2661536)`. The chosen trial
SSE differs from native printed values by 0.051 mm² LH and 0.041 mm² RH.
The selected dt above comes only from Python scores and fit; the earlier
coordinate-fitted dt in the gradient reports is diagnostic. The independent
search and every updated vertex satisfy the 0.00001 mm first-step checkpoint
criterion. Reports are [LH](standard_sphere_line_search_lh.json) and
[RH](standard_sphere_line_search_rh.json). The focused float32 fit regression
passes on control-host (`1 passed`). Timing includes Numba JIT for the gradient,
while line-search timing is warm; no native matched-time claim follows.

## Paired repair and next-epoch checkpoints

The isolated native run reports a 0.005% negative-area threshold. The projected
LH input has 2,185 negative faces (0.0221117%) and enters five repair updates;
RH has 491 (0.00331086%) and skips repair. For each listed update, Python
starts from that update's **native saved input**, independently computes the
whole-surface gradient and line search, and compares every ordered output
vertex with the next native snapshot. These checks therefore establish
paired-checkpoint parity; they do not yet establish a continuously propagated
Python `mris_sphere` run.

| Hemisphere and update | Distance weight | Native / Python selected trial | Python `dt` | Exact xyz components | Vertices within 0.00001 mm | Max vertex error (mm) | Python gradient / search CPU time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LH repair 0 | 0.000001 | 4 / 4 | 0.967522740 | 319,856 / 319,866 | 106,622 / 106,622 | 0.000007629 | 2.347 / 1.058 |
| LH repair 1 | 0.00001 | 0 / 0 | 0.013019243 | 319,866 / 319,866 | 106,622 / 106,622 | 0 | 1.349 / 0.834 |
| LH repair 2 | 0.001 | 3 / 3 | 0 | 319,866 / 319,866 | 106,622 / 106,622 | 0 | 1.426 / 0.821 |
| LH repair 3 | 0.01 | 2 / 2 | 0.250300967 | 319,861 / 319,866 | 106,622 / 106,622 | 0.000008530 | 1.498 / 0.796 |
| LH repair 4 | 0.1 | 4 / 4 | 6.571062565 | 319,810 / 319,866 | 106,622 / 106,622 | 0.000008741 | 1.337 / 0.793 |
| LH first unfold 5 | 0.1 | 2 / 2 | 0.014806829 | 319,863 / 319,866 | 106,622 / 106,622 | 0.000003815 | 1.360 / 0.733 |
| LH next unfold 6 | 1 | 4 / 4 | 0.008593750 | 319,866 / 319,866 | 106,622 / 106,622 | 0 | 1.501 / 0.791 |
| RH first unfold 0 | 0.1 | 4 / 4 | 17.633028030 | 316,581 / 316,623 | 105,541 / 105,541 | 0.000007689 | 2.367 / 1.164 |
| RH next unfold 1 | 1 | 4 / 4 | 0.004401409 | 316,623 / 316,623 | 105,541 / 105,541 | 0 | 1.379 / 0.764 |

The original-distance matrix takes 16.070 s LH and 15.714 s RH on CPU,
including Numba compilation/cache loading. The per-update times are measured
inside the Python probe and exclude input loading and the matrix build. Native
logs give only whole isolated-command time (0.0111 h LH, 0.0085 h RH), so
these timings do not support a full-stage speed ratio.

A first LH repair-3 comparison missed the 0.00001 mm threshold at the three
vertices of face 167119 (largest error 0.000195576 mm). Its cross-product
normal has length about 8.37e-8, below float32 epsilon. Native `V3_NORMALIZE`
keeps that short cross product unscaled; the former Python implementation
unitized it and changed the local area force. Applying the native threshold
reduces the maximum error to 0.000008530 mm across all 106,622 vertices.
The [before-correction diagnostic](standard_sphere_repair_step3_diagnostic_lh.json)
and [corrected step-3 report](standard_sphere_repair_step3_epsilon_corrected_lh.json)
record this first divergence and its resolution. The corrected full
[LH](standard_sphere_repair_next_corrected_lh.json) and
[RH](standard_sphere_next_corrected_rh.json) reports include SHA256 of each
input, native verbose log, native before/after surface, implementation module,
and ordered step specification, plus selected scores and timings. The focused
near-degenerate-face regression passes (`1 passed`).

## Continuous state propagation from original input

The [continuous probe](experimental/probe_standard_sphere_continuous.py)
starts from the copied original `inflated` and `smoothwm` files, samples the
original metric once, and carries its own coordinates through each update.
Native snapshots are read only for comparison; no native state is injected.
A force-accumulation-order correction made all seven LH and both RH earlier
updates exact in every coordinate component. The older
[pre-correction LH](standard_sphere_continuous_lh.json) failure at repair 3
is historical; the current [LH](standard_sphere_continuous_nonlinear_lh.json)
and [RH](standard_sphere_continuous_nonlinear_rh.json) runs have no divergence
through the first nonlinear-area fold-repair update.

| Continuous chain from original input | LH | RH |
| --- | ---: | ---: |
| Initial projection exact components | 319,866 / 319,866 | 316,623 / 316,623 |
| Earlier repair / main-unfold updates | 7 / 7 exact | 2 / 2 exact |
| First nonlinear update index | 7 | 2 |
| First nonlinear input exact components | 319,866 / 319,866 | 316,623 / 316,623 |
| First nonlinear output exact components | 319,866 / 319,866 | 316,623 / 316,623 |
| First nonlinear maximum vertex error | 0 mm | 0 mm |
| First nonlinear Python selected trial / native | 4 / 4 | 2 / 2 |
| First nonlinear Python selected `dt` | 0.01120199170 | 0.00147362326 |

The native switch before this nonlinear stage calls
`MRISresetNeighborhoodSize(mris, 1)`. The Python implementation takes the
first-ring rows from the full original metric (639,720 LH and 633,234 RH
entries) while retaining native `avg_nbrs` from the prior full matrix
(77.553604 LH and 77.537201 RH). The first fold-repair coefficient has
`l_nlarea=1` and `l_dist=1e-6`. Its full-surface SSE agrees with native
`FREESURFER_logSSE=1` to all six printed decimals:

| First nonlinear starting SSE | LH native / Python | RH native / Python |
| --- | ---: | ---: |
| Nonlinear area | 4648.798864 / 4648.798864 | 4266.923948 / 4266.923948 |
| Weighted one-ring distance | 0.182286 / 0.182286 | 0.178350 / 0.178350 |
| Total | 4648.981150 / 4648.981150 | 4267.102298 / 4267.102298 |

The [paired LH](standard_sphere_nonlinear_one_ring_lh.json) and
[paired RH](standard_sphere_nonlinear_one_ring_rh.json) reports reproduce
the same first nonlinear outputs independently from each native input
checkpoint. Their Python coordinate input and output SHA256 also equal those
in the continuous reports. All four reports contain input/reference and
implementation SHA256, ordered candidate scores, selected time step, and
whole-surface coordinate errors. The isolated native SSE log SHA256 values
are `696c0643647b89e9f2c0b868563e6d634a58ca49e813b8f38c5ec2677b8bc927`
(LH) and `f2fc28cb37f0ac643f7ff088ea47e7303593e43ccf73fde5b7b0b67487a034bb`
(RH). The native logs remain in the isolated `unfold_first_{lh,rh}/nlarea_sse_probe`
directories; the official subject was never modified.

The latest continuous CPU run took 17.03 s LH / 16.13 s RH to build the
full metric, 4.81 / 4.70 s to select one-ring rows, then 1.74 / 1.96 s for
the first nonlinear gradient, 0.235 / 0.279 s for its line search, and
0.004 / 0.004 s to apply the update. These measurements include Numba
compilation or cache loading and server-load variation. The separate native
`logSSE` commands took 0.0103 h LH and 0.0073 h RH end-to-end with one CPU
thread; they perform multiple stages and diagnostics, so they are not matched
first-update timing comparators. The current Python port runs on CPU with
Numba; this is numerical checkpoint validation, not a GPU speed claim.

## Full isolated first-pass coefficient sweep and final file

The next bounded probe continues the **same Python coordinates** from each
original `inflated` input through every update saved by an isolated native
`mris_sphere -threads 1 -seed 1234 -a 0 -n 1 -p 1` run. It never injects a
native coordinate snapshot. After the initial 7 LH / 2 RH updates, both
hemispheres execute five nonlinear-area updates with `l_nlarea=1` and ordered
`l_dist` values `1e-6, 1e-5, 1e-3, 1e-2, 0.1`. The native log omits `1e-4`
and has no subsequent `l_dist=1` update in this isolated first pass. Python
uses the same one-ring metric throughout these five updates.

| Isolated first pass from original input | LH | RH |
| --- | ---: | ---: |
| Saved updates checked | 12 / 12 | 7 / 7 |
| Nonlinear candidate selections matching native | 5 / 5 | 5 / 5 |
| Saved coordinate components exactly matching at every update | 319,866 / 319,866 | 316,623 / 316,623 |
| Final isolated output coordinate components exactly matching | 319,866 / 319,866 | 316,623 / 316,623 |
| Final maximum ordered-vertex error | 0 mm | 0 mm |
| Final ordered faces matching | 213,240 / 213,240 | 211,078 / 211,078 |
| Raw vertex / face / volume-info payloads matching byte-for-byte | yes / yes / yes | yes / yes / yes |
| Full output file SHA matching | no | no |
| Python metric build / one-ring setup / all gradients / all searches | 14.12 / 4.32 / 15.58 / 5.76 s | 16.39 / 5.12 / 9.65 / 2.61 s |
| Sum of timed Python substeps including projections, excluding file reads and write | 41.92 s | 34.62 s |

The isolated native output applies one last radial projection after the
final saved update. Applying that projection in Python produces identical
float32 coordinates in source order. The
[Python surface writer](../../../src/fnit/recon_all/sphere_standard_python.py)
keeps the original `inflated` volume-geometry text block byte-for-byte;
Nibabel's default writer rounds that text and caused up to
`2.15e-9` difference in `cras` on the LH trial. The generated and native
files have identical raw vertex arrays, face arrays, and volume-geometry
blocks. Their whole-file SHA values differ because the creation stamps and
FreeSurfer provenance tags differ; the Python file omits those tags. The
[raw LH](standard_sphere_file_audit_lh.json) and
[raw RH](standard_sphere_file_audit_rh.json) reports give payload and file
SHA256 values. The [continuous LH](standard_sphere_continuous_first_pass_lh.json)
and [continuous RH](standard_sphere_continuous_first_pass_rh.json) reports
give frozen input and implementation hashes, all selected candidate scores,
per-step timings, and ordered-vertex comparisons. The native candidate SSEs
were logged to two decimals; Python differs from those printed values by at
most 0.0051 LH / 0.0049 RH. Every selected candidate matches and every
Python time step rounds to the native three-decimal printout.

These timings are from CPU Numba on control-host and include cache/JIT loading.
The native log reports only the whole isolated command in coarse hundredths
of an hour and includes other work. A matched stage speed ratio is therefore
not established. The focused tests, including the writer's exact
volume-geometry check, pass (`6 passed`) in the remote Python environment.

**Open boundary:** this validates one deliberately shortened, isolated
`MRISunfold` pass and its written LH/RH surfaces. The standard recon-all
`mris_sphere` run uses up to 25 integrations at each average level and a
subsequent overlap-removal stage; its final conventional `sphere` surfaces
and downstream cortical vertex metrics have not been matched by Python.
The Python path here runs on CPU with Numba, not GPU.

To repeat the isolated diagnostic, use the frozen `inflated` and `smoothwm`
files from one subject. The native diagnostic deliberately returns status 1
after writing `distance.log`; the capture script checks for that file and
never writes into the official subject:

```bash
export FS_LICENSE=/path/to/private/license.txt
bash -lc 'source validation/recon_all/python_gpu_port/experimental/capture_standard_sphere_native_metric.sh /path/to/lh.inflated /path/to/temporary/metric_native'
PYTHONPATH=src python validation/recon_all/python_gpu_port/experimental/benchmark_standard_sphere_metric.py \
  /path/to/lh.smoothwm \
  --native-log 0:/path/to/native/v0.log \
  --native-table /path/to/temporary/metric_native/distance.log
```

Use the corresponding right-hemisphere paths for RH.

## Standard-parameter full native baseline and exact original metric

A separate isolated run uses the recon-all command parameters (`-threads 4
-seed 1234`) on copied `fs_sub01` inputs. `-w 25 -v 0` saves sparse
checkpoints but does not alter the final ordered geometry. Its two final
surfaces agree with the read-only official archived `sphere` files in every
float32 vertex coordinate, every ordered face, and every volume-geometry
byte. Creation stamps and provenance tags differ, so whole-file SHA256
values differ. [LH audit](full_native_audit_lh.json) and
[RH audit](full_native_audit_rh.json) include the copied-input SHA256 values,
final-section hashes, checkpoint indices, native log SHA256, and wall time.

| Full conventional native baseline on control-host, four threads | LH | RH |
| --- | ---: | ---: |
| Ordered coordinate components equal to official | 319,866 / 319,866 | 316,623 / 316,623 |
| Ordered faces equal to official | 213,240 / 213,240 | 211,078 / 211,078 |
| Volume geometry bytes equal to official | yes | yes |
| Isolated native wall time | 252.20 s | 113.20 s |

For an exact original-distance audit, a **diagnostic copy** of the native
executable changes only the unique `FS_MEASURE_DISTANCES` format string from
`%2.4f  %2.4f` to `%2.9g  %2.9g`. Nine significant decimal digits round-trip
each float32; the installed binary, official subject, and algorithm are
unchanged. Reading the second column in native source order and casting to
float32 gives a bitwise comparison with the complete Python CSR distance
array **after reciprocal averaging**:

| Original metric, all entries | LH | RH |
| --- | ---: | ---: |
| Native and Python float32 values bitwise equal | 8,268,920 / 8,268,920 | 8,183,354 / 8,183,354 |
| Maximum absolute/ULP difference | 0 / 0 | 0 / 0 |
| Native isolated sample and text export | 15.35 s | 13.94 s |
| Python metric build including JIT | 15.36 s | 16.13 s |

The [LH exact-metric report](standard_sphere_exact_metric_lh.json) and
[RH exact-metric report](standard_sphere_exact_metric_rh.json) contain native
text and diagnostic-binary SHA256 values. Native timing includes writing
roughly 180 MB of text, while Python timing excludes that write; it is not a
paired algorithm speed benchmark. The target distances are no longer an
unresolved source of the first optimization difference.

## Historical bounded default-average optimization prefix

The reports in this section were generated before the float32-stored
`l_dist` correction below. They document the earlier diagnostic branch and
are not current parity claims.

The `-n 1 -w 1 -remove_negative 0` native diagnostic preserves the default
1024 initial gradient averages but advances each average level after one
integration. It is a separate branch from the 25-iteration standard command:
its first update agrees with the full native run, while its second saved
surface differs from the full run on both hemispheres. The
[bounded LH](standard_sphere_default_continuous_lh.json) and
[bounded RH](standard_sphere_default_continuous_rh.json) reports start from
the original `inflated` files, propagate only Python coordinates, and hash
all inputs, implementation modules, and saved native checkpoints.

FreeSurfer's `mrisLineMinimize` omits the predicted quadratic trial when
`|a| < FLT_EPSILON`. Enforcing this source gate removes an LH branch
error at `initial_repair` average 64: 15 consecutive bounded LH updates
(indices 0–14) now match every float32 coordinate component. The next LH
bounded mismatch is index 15, `initial_repair` average 256 in the third
ratio sweep: selected Python dt 60496.0078125 versus native 60487.953,
maximum vertex error 0.002994 mm. Holding the Python gradient fixed and
fitting only dt to the native checkpoint brings all 106,622 vertices within
0.00001 mm (max 0.000008543 mm). RH matches the first bounded update exactly;
its second update, `unfold_epoch_1` average 256, selects Python dt
905.4383545 versus native 905.111, producing maximum vertex error
0.004337 mm. Fitting only dt gives maximum 0.000010957 mm, with 105,537 /
105,541 vertices within 0.00001 mm. These fitted steps are diagnostics,
not inputs to the continuous Python chain.

At the RH second bounded update, the Python negative-area SSE agrees with
native `FREESURFER_logSSE` to six printed decimals for the starting state
and three bracket trials. Distance SSE differs by approximately 0.014–0.017
across those trials. Independent C++ code using the pinned `XYZApproxAngle`
formula produces the same 8,183,354 current spherical arc float32 values as
Python, bitwise, at the saved starting surface. The remaining difference
is under investigation in native update-state or distance-error evaluation;
it is large enough to change the float32 quadratic fit. The bounded Python
path remains on CPU. It does not provide a full-stage speed or accuracy claim.

## Standard 25-iteration continuous prefix

The default `mris_sphere -threads 4 -seed 1234` performs up to 25 updates
inside one integration at each average level. An isolated `-w 1` native
capture saved consecutive checkpoints on copied inputs without changing the
integration count. The Python replay projects at integration entry and after
each update, without adding an entry projection inside the same integration.
The source-hashed [LH four-update](standard_sphere_full_default_prefix_lh_256.json)
and [RH seven-update](standard_sphere_full_default_prefix_rh_seven.json)
reports propagate Python coordinates from the original `inflated` and
`smoothwm` inputs; they do not inject native checkpoints. The earlier
[LH three-update](standard_sphere_full_default_prefix_lh.json) and
[RH five-update](standard_sphere_full_default_prefix_rh.json) reports
remain as historical bounded snapshots.

The native `INTEGRATION_PARMS::l_dist` is float32. The Python SSE originally
multiplied by a double `0.1`, adding 0.013–0.021 to the RH second-update
bracket comparison and changing its quadratic dt. Casting the weight to
float32 before the double SSE product makes the starting and three bracket
scores agree with native to at most `2.3e-7` in the earlier second-update
diagnostic. The original trial distances at vertex 0 also matched bitwise in
a diagnostic binary that printed hexadecimal floats. The
[resolved-weight audit](standard_sphere_full_default_rh_weight_resolution.json)
retains the before/after values and hashes.

| Full-default 1024-average continuous prefix | LH | RH |
| --- | ---: | ---: |
| Complete original target-distance matrix exact | 8,268,920 / 8,268,920 | 8,183,354 / 8,183,354 |
| Exact ordered float32 coordinates, updates | 0–3 (4 steps) | 0–6 (7 steps) |
| First unverified update | 4, not yet compared | 7, not yet compared |

Before the source-order correction, **RH update 5** was the first numerical
mismatch. Its input and original target-distance matrix were exact, but the
third line-search bracket had native/Python distance SSE `764608.958705` /
`764608.958528` and the selected Python dt was `2797.545166015625` versus
native printed `2796.162`. The historical
[first-difference](standard_sphere_full_default_rh_first_difference.json) and
[reduction](standard_sphere_full_default_rh_update5_reduction.json) audits
retain that failed run.

An isolated GDB probe stopped the installed native binary at its 65th
`logSSE` call, immediately after computing the third bracket trial. It read
all 105,541 ordered trial vertices and all 8,183,354 ordered current and
target distances from native memory. The target distances match Python
bitwise. Only vertex 83,527's x coordinate differs by one float32 ULP
(`0xc186c00b` native versus `0xc186c00c` Python); 10 current distances
differ, and each touches that vertex. Recomputing those 10 spherical arcs
from each trial mesh reproduces both sides exactly. They account for the
distance-SSE discrepancy, before the quadratic fit.

The pinned `mrisLineMinimize` source accumulates gradient magnitudes in
vertex order using a scalar double sum. The Python code used NumPy's pairwise
sum. The native and corrected scalar mean are both
`0x1.e47e052aab40fp-12`; the old NumPy mean was
`0x1.e47e052aab380p-12`. This changes the third bracket dt by
`5.46e-11`, exactly crossing the float32 rounding boundary for that one
coordinate. With the native dt, the independently projected trial mesh
matches all 316,623 coordinates bitwise. The correction is a source-order
sum in `sphere_standard_line_search.py`; no dt or vertex value is hardcoded.
The [native memory audit](standard_sphere_full_default_rh_update5_trial_memory.json)
records input and checkpoint hashes, captured binary-array hashes, all
entry counts, and the first differing values. Its
[GDB probe](experimental/capture_standard_sphere_trial_memory.py) runs on a
copy of the surfaces and stops before the full command finishes.

With the correction, the Python gradient at RH update 5 matches all
316,623 native gradient components. The three bracket SSE scores agree with
native to six printed decimals, the selected dt is
`2796.162353515625`, and the independently updated mesh matches the native
saved checkpoint in all 316,623 coordinate components. The
[continuous six-update report](standard_sphere_full_default_prefix_rh_serial_mean.json)
replays updates 0–5 from the original `inflated` and `smoothwm` inputs,
without injecting any native checkpoint; every update is bitwise exact.
The bounded GDB capture took 20.91 s, the isolated corrected line search
1.25 s, and the continuous probe's original metric build 15.87 s on a
shared control-host node. These are observations with different workloads, not a
matched speed comparison.

The next saved RH update, index 6, also matches: its input and output are
both exact in 316,623 / 316,623 float32 coordinate components; Python
selects `dt=4875.181640625`. Index 7 was the next boundary for that
no-injection run.
For LH, the native verbose log switches from `navgs=1024` to `navgs=256`
after update 2 (`tol=2.505e-01`). Treating index 3 as another 1024-average
update incorrectly selected `dt=0` and missed the native surface by up to
1.12164 mm. With the logged 256-average stage and its entry projection,
the continuous Python chain matches all 319,866 / 319,866 float32 coordinate
components through LH update 3; Python selects `dt=34098.9043711711`
versus native printed `34098.904`. The bounded four-decision native capture
used copied inputs, ended after the fourth `sses:` line in 17.10 s, and its
log SHA256 is `d48f9399baa9c577c51815303fedbffcfcb1b9da068a010a3a2941b110e3de7c`.
The paired LH index-3 checkpoint SHA256 values are `850f08e2be639d2fe2c2f617d32bb31ec9732d154505831ee70cae332132145b`
and `71a897f12406d5600e425f1b1fdd15a33a3d2c0d232d65518870cb14d8360a9f`.
The next bounded update uses a checkpoint whose file SHA256 and ordered
coordinate SHA256 equal the previous verified Python output. The probe
rebuilds the same original metric from `smoothwm`, then computes only that
one update; it does not rerun the already verified prefix. The
[LH index-4](standard_sphere_full_default_resume_lh_index4.json) and
[RH index-7](standard_sphere_full_default_resume_rh_index7.json) reports
record this explicit resume boundary and each source hash. The
[candidate and decision audit](standard_sphere_full_default_resume_next_decisions.json)
pairs every native `FREESURFER_logSSE` candidate with Python:

| Checkpoint-resumed update | Native / Python average count | Selected Python dt / native printed | Maximum absolute candidate SSE difference against native six-decimal log | Exact output components |
| --- | ---: | ---: | ---: | ---: |
| LH index 4 | 256 / 256 | 40628.629297542 / 40628.629 | 3.42e-7 | 319,866 / 319,866 |
| RH index 7 | 1024 / 1024 | 2491.848144531 / 2491.848 | 3.95e-7 | 316,623 / 316,623 |

The isolated native captures stopped after five LH and eight RH line-search
decisions, respectively. No complete native sphere was rerun for this audit.
The next [checkpoint-resumed LH index-5](standard_sphere_full_default_resume_lh_index5.json)
and [RH index-8](standard_sphere_full_default_resume_rh_index8.json) updates
also pass. Their original `inflated` and `smoothwm` SHA256 hashes match the
prior reports; each resumed checkpoint file SHA256 and ordered-coordinate
SHA256 match the previous Python output. Native logs confirm LH remains at
`navgs=256` and RH at `navgs=1024`. The [candidate audit](standard_sphere_full_default_resume_indices5_8_decisions.json)
records every native `logSSE` block, the matching Python candidate, capture
and input hashes, and per-step timings:

| Checkpoint-resumed update | Native / Python average count | Selected Python dt / native printed | Maximum absolute candidate SSE difference against native six-decimal log | Exact output components |
| --- | ---: | ---: | ---: | ---: |
| LH index 5 | 256 / 256 | 0 / 0.000 | 4.42e-7 | 319,866 / 319,866 |
| RH index 8 | 1024 / 1024 | 6436.5712890625 / 6436.571 | 4.32e-7 | 316,623 / 316,623 |

The bounded native captures ended after six LH and nine RH line-search
decisions, in 18.89 s and 20.86 s under shared control-host load. At that
point, LH index 6 and RH index 9 were unverified. The initial no-injection
propagation from the original input reached LH index 3 and RH index 6;
the later one-step replays had SHA-proven exact coordinate entry states. None of these timings is a matched benchmark.

The corrected Python CPU observations under shared control-host load include the
original distance-matrix construction and one-ring setup times plus each
step's gradient, averaging, line-search, and projection times in the linked
reports. These bounded prefixes do not establish a full-stage speed ratio.
The isolated native full-stage LH/RH reference took `252.20 s` / `113.20 s`
on control-host and has bitwise-identical final ordered vertices, faces, and volume
geometry to the archived official surfaces. The checkpoint-resumed path
above was a previous boundary. The complete final-sphere validation below
supersedes it. Source-ordered sphere optimization runs on CPU Numba.

The focused line-search and nonlinear SSE tests pass in the remote validation
Python environment, including the float32 coefficient regression. No full
recon-all reconstruction has yet passed the release gates.

## Source-driven default schedule from the original inputs

The Python scheduler now applies the pinned FreeSurfer 8.2 source rule from
[`MRISintegrate`](https://github.com/freesurfer/freesurfer/blob/d932c45/utils/mrisurf_integrate.cpp):
for each gradient-average scale, finish when
`100 * (old_sse - sse) / sse < 0.5 * sqrt((navgs + 1) / 1024)`, the selected
time step is zero, or the scale reaches 25 updates. The next scale divides
`navgs` by four. The initial negative-area coefficients come from
[`mrisRemoveNegativeArea`](https://github.com/freesurfer/freesurfer/blob/d932c45/utils/mrisurf_deform.cpp).
The independent scheduler is in
[`sphere_standard_schedule.py`](../../../src/fnit/recon_all/sphere_standard_schedule.py).
No native checkpoint supplies a coordinate or a scale decision during this
replay; saved native snapshots are read only after each Python update to
compare ordered vertices and faces.

| Original-input accelerator-a-1 CPU replay | LH | RH |
| --- | ---: | ---: |
| Consecutive saved updates checked | 33 (indices 0–32) | 49 (indices 0–48) |
| Exact coordinate components at every update | 319,866 / 319,866 | 316,623 / 316,623 |
| Exact ordered faces at every update | 213,240 / 213,240 | 211,078 / 211,078 |
| First/last stage | initial repair / initial repair | first unfold / second unfold |
| Original metric + one-ring setup, including JIT | 33.39 s | 33.62 s |
| Sum of measured per-update CPU work | 123.85 s | 205.23 s |

The LH repair scales switch at update indices 3, 6, 8, 10, 11, and 12;
its next distance coefficient begins at index 13, and its third coefficient
begins at index 30. RH first-unfold scales switch at indices 12, 19, 28,
38, 41, and 43; its second unfold begins at index 48. These boundaries
come from Python's own SSE and stopping rule. The complete per-update
input/checkpoint hashes, coordinate comparisons, decisions, and timings are
in the [LH report](standard_sphere_lh_auto_from_input_accelerator-a-1.json) and
[RH report](standard_sphere_rh_auto_from_input_accelerator-a-1.json). The reported
probe SHA-256 matches the committed validator, and the scheduler module's
SHA-256 is `58e101cefcc9c25688d14794f021c0f598db53c6968d7c79e1835d023ac85b8d`.

The timings exclude native-checkpoint reads, report serialization, and
process startup; they are not paired with a native run on accelerator-a-1. The
original frozen captures stopped at those limits; the complete validation
below uses newly captured native stage checkpoints.


## Complete conventional spheres and independent Python API

The complete isolated native 8.2 runs used the same frozen original inputs
with `-threads 4 -seed 1234 -w 1`. Their 244 LH and 135 RH saved checkpoints
extend the earlier captures without a coordinate or face difference in the
shared prefix. Their final sphere geometry also equals the archived official
output exactly: ordered vertices, faces, and volume-geometry bytes. See the
[LH capture audit](standard_sphere_lh_complete_native_audit.json) and
[RH capture audit](standard_sphere_rh_complete_native_audit.json).

The standalone [Python API](../../../src/fnit/recon_all/sphere_standard_run.py)
reads only `inflated` and `smoothwm` and writes the final `sphere`; it does
not read native checkpoints or invoke FreeSurfer. It chooses the initial
negative-area branch from the projected input and all later scales from its
own SSE. On the frozen `fs_sub01` subject, both independent control-host CPU runs
match the complete native output payload exactly:

| Standalone API and exact final payload | LH | RH |
| --- | ---: | ---: |
| Vertices / ordered faces | 106,622 / 213,240 | 105,541 / 211,078 |
| Source-scheduled optimization updates | 243 | 134 |
| Initial negative-area percentage | 0.0221117% | 0.00331086% |
| Ordered vertex bytes match | yes | yes |
| Ordered face bytes match | yes | yes |
| Volume geometry bytes match | yes | yes |
| Total control-host CPU time, including I/O | 372.82 s | 220.17 s |
| Original metric setup, including JIT | 20.39 s | 15.85 s |
| All optimization updates | 349.62 s | 202.39 s |
| Final projection and overlap cleanup | 1.91 s | 1.32 s |

The [LH API report](standard_sphere_lh_api_control-host.json),
[RH API report](standard_sphere_rh_api_control-host.json), and byte-level
[LH](standard_sphere_lh_api_control-host_file_audit.json) /
[RH](standard_sphere_rh_api_control-host_file_audit.json) audits retain update
counts, stage times, input paths, and final payload hashes. Whole-file hashes
differ because the creation stamp and provenance tags differ. No comparison
uses those tags as a numerical criterion.

The continuous accelerator-a-1 [RH probe](standard_sphere_rh_full_python_accelerator-a-1.json)
propagated its own coordinates from original input through all 134 updates;
every native snapshot had all 316,623 float32 components exact, and the H100
final cleanup produced the exact final 105,541-vertex sphere. The equivalent
LH run reached update 242 without a reported checkpoint divergence, but its
GPU0 final cleanup raised CUDA out-of-memory before its report could be
written. The isolated [LH GPU1 cleanup](standard_sphere_lh_finish_cuda1_accelerator-a-1.json)
from the exact native last checkpoint matches all 106,622 final vertices and
all 20 native negative-triangle counts. Separate CPU final-cleanup reports
([LH](standard_sphere_lh_finish_cpu_control-host.json),
[RH](standard_sphere_rh_finish_cpu_control-host.json)) are also exact.

Reproduce the independent stage with:

```bash
python -m fnit.recon_all.sphere_standard_run \
  surf/lh.inflated surf/lh.smoothwm surf/lh.sphere \
  --finish-device cuda:0 --report lh_sphere_report.json
```

`--finish-device` affects only the final overlap cleanup. Metric sampling,
gradients, line search, and schedule currently use CPU/Numba; this is not a
full-GPU implementation. The native complete-capture times (381.18 s LH, 200.28 s RH on accelerator-a-1)
include writing every intermediate snapshot and are not speed comparators.
A separate same-input, same-host control-host native run without snapshot writes
passed the same byte-level geometry audit for each hemisphere:

| control-host conventional sphere, including I/O | LH | RH |
| --- | ---: | ---: |
| Native FreeSurfer 8.2, four threads | 240.64 s | 109.72 s |
| Independent Python CPU API | 372.82 s | 220.17 s |
| Observed Python/native ratio | 1.55× | 2.01× |

The [benchmark report](standard_sphere_stage_benchmark_control-host.json) links
input hashes, native log/time-record hashes, bilateral
[LH](standard_sphere_native_benchmark_lh_control-host_audit.json) and
[RH](standard_sphere_native_benchmark_rh_control-host_audit.json) output audits,
and Python times for metric construction, each optimization stage, and final
cleanup. The calls ran sequentially on a shared host. Native used four
threads; Python used the environment's default CPU thread settings and
Numba JIT. The ratios are observed stage timings, not a controlled
thread-matched speed claim. Python currently spends 349.62 s LH and
202.39 s RH in optimization updates, the largest cost.

These two exact sphere outputs do not establish a native-free
T1-to-surface-to-thickness reconstruction; the upstream topology and
white/pial stages and downstream registration and vertex-metric release
gates remain open.
