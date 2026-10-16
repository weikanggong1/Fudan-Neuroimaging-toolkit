# `mris_fix_topology`: first-defect fitness and GA selection boundary

> 历史冻结阶段记录。2026-10-10 源核对发现早期排名 helper 的 `1e-5`
> 同分阈值与实际翻译单元 `FLT_EPSILON` 不符；现版字段写回、权重及
> 同分规则见 [中文说明](../../../../docs/recon_all/TOPOLOGY_PATCH_FITNESS.md)。
> 下列四位小数日志及历史数值保留原验证范围，不改标为当前完整 fitness 回归。

The pinned FreeSurfer 8.2 source is
`d932c45b7941662ea380a05efef580568b98d41a`. This report follows the
fixed `fs_sub01` first defect on each hemisphere after the independently
validated initial `select0` patch. It is a partial topology port, not a
replacement for `mris_fix_topology` or an end-to-end recon-all validation.

## Source sequence and reference

`mrisComputeOptimalRetessellation_wkr` builds each candidate ordering,
then `mrisDefectPatchFitness` retessellates, discovers/orients patch faces,
computes patch Euler, smooths and matches original coordinates, evaluates MRI
and curvature likelihoods, applies the face-validity penalty, and restores
the prior surface state. The early helper ranked the initial 10 fitness values
with a `1e-5` tie assumption; the current source translation-unit audit corrected
this to `FLT_EPSILON`. Later generations mutate, cross over,
select elites, and can delete vertices. After search, the saved best ordering
and vertex status are replayed and matched again before cleanup.

A bounded native `-correct_defect 0 -verbose_low -ga -seed 1234 -threads 1`
reference was run on each frozen hemisphere copy. It stops after defect 0
with the tool's diagnostic exit code 252. It did not rerun recon-all. The
no-snapshot verbose calls took 10.66 s LH and 36.63 s RH; these times are not
comparable to a complete Python topology stage. Native log and snapshot
SHA-256 values are in the bilateral machine-readable reports below.

## Independently computed structural fitness gates

Python starts with `qsphere.nofix`, `orig.nofix`, and `brain.mgz`, deriving
defect labels/status, MRI-scored candidate edges, and the first patch without
a captured native EDGE table. The packaged
[`topology_first_candidate.py`](../../../../src/fnit/recon_all/topology_first_candidate.py)
selects nonintersecting edges. The packaged
[`topology_fitness_search.py`](../../../../src/fnit/recon_all/topology_fitness_search.py)
checks the modified patch's edge-to-face incidence used by
`mrisCheckDefectFaces`; the patch probe calculates source
`computePatchEulerNumber` terms.

| First `select0` patch | LH | RH |
| --- | ---: | ---: |
| Retained interior vertices | 54 | 124 |
| Accepted patch edges | 223 | 487 |
| Added patch faces | 170 | 364 |
| Patch Euler (`V−E+F`) | 1 | 1 |
| Modified edges failing two-face attachment | 0 | 0 |
| Ordered vertex / added-face mismatch to native snapshot | 0 / 0 | 0 / 0 |

The RH native verbose source prints `X=1 (v=124,e=487,f=364)` for this
candidate. Source validity does not apply the `−10,000,000` face penalty to
the first candidates, consistent with the Python zero-violation check.
This verifies the first candidate's structural gates, not the MRI or
curvature likelihood fields.

## First-candidate coordinate smoothing

The packaged [`topology_defect_smooth.py`](../../../../src/fnit/recon_all/topology_defect_smooth.py)
now reproduces source `defectSmooth` type 2: 25 iterations at `alpha=0.1`,
source face/vertex normal recomputation, and the retained runtime neighbor
ordering. The source increments its neighbor loop counter in both the loop
body and header; preserving that behavior is required for parity.

The fixed comparison starts from the saved native `select0` geometry, whose
ordered coordinates and added faces were independently generated from the
input surface and MRI. The curvature histograms are **native validation
fixtures**. Their plots print bin positions to six decimals, so the validator
uses the exact float32 bin widths in `mri_k1_k2.mgh`, reconstructs each
histogram's unique first bin and source-order additions, then computes the
source mean. At this earlier candidate-smoothing boundary the histograms had not yet
been generated independently; the later global first-pass result is below.
Each hemisphere uses its own isolated `-correct_defect 0` diagnostic capture;
these calls ended with the expected diagnostic exit code 252 and did not run
full recon-all.

| Native `select0s` coordinate check | LH | RH |
| --- | ---: | ---: |
| Interior vertices moved by source smoothing | 54 | 124 |
| Interior vertices bitwise equal to Python | 54 / 54 | 124 / 124 |
| All vertices bitwise equal to Python | 102,764 / 102,764 | 101,454 / 101,454 |
| Maximum coordinate distance | 0 mm | 0 mm |
| Python smoothing including Numba JIT, final report | 1.31 s | 1.27 s |

The source writes `select0s` before MRI-based coordinate matching. These
numbers establish only that first-candidate smoothing transition; they are
not a paired timing comparison with the native smoothing kernel. Reports:
[LH smoothing](lh.defect_smooth_report.json),
[RH smoothing](rh.defect_smooth_report.json),
[validator](validate_topology_defect_smooth.py), and the expanded
[LH](lh.independent_first_patch.json) / [RH](rh.independent_first_patch.json)
independent patch inputs with runtime neighbor rows.

## Score composition and first population

For this exact native command the source combines the printed components as
`face_ll + vertex_ll + normal_dot_ll + quadratic_curvature_ll + 10 × volume_unmri_ll`,
then subtracts `10,000,000` if the face check fails. The Python composer and
ranking function reproduce that rule. The likelihood components in this
check come from the **native verbose log**; the Python implementation does
not yet generate those five quantities from MRI and geometry. Their native
four-decimal printout limits the composition check's precision.

| Initial population | LH | RH |
| --- | ---: | ---: |
| Fitness rows ranked | 10 | 10 |
| Python / native best candidate index | 2 / 2 | 1 / 1 |
| Best fitness (native six decimals) | −108.173866 | −108.332152 |
| Largest composition error for first two candidates from four-decimal printed terms | 0.000051 | 0.000452 |
| Python / native mean, rounded four decimals | −112.0512 / −112.0512 | −110.5610 / −110.5610 |
| Python / native SD, rounded four decimals | 2.1648 / 2.1648 | 1.3409 / 1.3409 |
| Selected `selectN` vs saved `bestK`, ordered geometry | Exact | Exact |

The source's best snapshot serial counts *improvements*, not candidate
indices. In this initial population the serial happened to equal the winning
index on each hemisphere (LH 2, RH 1). The fixed report checks that the
selected `selectN` and matching `bestK` have identical ordered coordinates
and faces.

## First-candidate MRI coordinate matching

The packaged [`topology_defect_mri_match.py`](../../../../src/fnit/recon_all/topology_defect_mri_match.py)
now implements the 40 synchronous iterations of source
`defectMaximizeLikelihood_new` with `alpha=0.5`. The isolated validator starts
from each native `select0s` surface, uses `brain.mgz`, the independently
derived patch neighbor rows, and the native white/gray histogram plots. It
compares the resulting ordered coordinates with native `select0sm` by float32
bits. A second validator below generates the histograms independently; all
face rows are unchanged.

The previous probe incorrectly assumed `HISTOGRAM.bin_size=1`. In this source
path `HISTOalloc` zero-initializes that field, and
`mrisComputeGrayWhiteBorderDistributions` sets `bins[n]=n` without changing the
bin size. `computeDefectStatistics` therefore uses `bins[n]` directly. The old
probe subtracted `0.5` from both target means; fixing that input removed the
coordinate discrepancy on both hemispheres.

| `select0s → select0sm` coordinate check | LH | RH |
| --- | ---: | ---: |
| Retained interior vertices | 54 | 124 |
| Interior vertices bitwise equal | 54 / 54 | 124 / 124 |
| All vertices bitwise equal | 102,764 / 102,764 | 101,454 / 101,454 |
| Maximum coordinate distance | 0 mm | 0 mm |
| Python kernel wall time including Numba JIT | 1.80 s | 1.32 s |

The kernel timings exclude image/surface reads and the comparison/report;
they are not full stage times. The source inputs and target snapshot SHA-256
values are recorded in the
[LH](lh.defect_mri_match_ported.json) and
[RH](rh.defect_mri_match_ported.json) reports; the
[validator](probe_topology_defect_mri_match.py) reproduces this check.
An isolated source instrumentation of the first LH candidate captured the
source means `74.8134232` and `58.2887955`, agreeing with the corrected
plot-based float32 means. The older failed
[LH](lh.defect_mri_match_probe.json) and
[RH](rh.defect_mri_match_probe.json) reports are retained as error evidence.
The native diagnostic timing includes unrelated topology work and is not a
paired benchmark for this coordinate-matching kernel.

The original test is an exact **first-candidate transition** using frozen
histogram plots and a native `select0s` checkpoint. The independent histogram
check below removes the plot input. Neither test validates `orig.premesh` or
cortical metrics.

## Independent first-defect white/gray intensity histograms

The packaged [`topology_intensity_histogram.py`](../../../../src/fnit/recon_all/topology_intensity_histogram.py)
reproduces the first defect's white and gray histogram selection, zero-bin
pseudo count, normalization, source `HISTOsmooth` with sigma 2, and the
1%-of-peak truncated float32 means. The independent validator reads
`qsphere.nofix`, `orig.nofix`, and `brain.mgz`; it derives defect labels,
original-surface normals, MRI white/gray samples, and two median passes. The
native `.plt` files are opened **after** histogram and `select0sm` prediction,
solely for comparison. The patch neighbor rows and `select0s` geometry remain
frozen first-candidate checkpoints; this is not an independent GA search.

A first attempt visited only two one-ring steps: LH selected 322 vertices
rather than the native 543 and missed 53 of 54 interior output vertices.
The pinned optimal-retessellation path sets `vtotal` to two rings at
`mrisurf_defect.cpp:8008`; `mrisComputeGrayWhiteBorderDistributions` then
nests two `vtotal` traversals at lines 15237–15321. Four one-ring steps
select 543 LH and 806 RH non-defect vertices, matching the native raw
histogram pseudo count `0.1 / nvertices`. This correction is preserved in the
module rather than tuning bins against the reference plots.

| Independent first-defect check | LH | RH |
| --- | ---: | ---: |
| Defect-label differences against native | 0 | 0 |
| White/gray raw histogram bins matching native 10-decimal plots | 256/256 each | 256/256 each |
| White/gray smoothed bins matching native 10-decimal plots | 256/256 each | 256/256 each |
| Largest absolute difference from parsed plot value | 4.98e-11 | 4.98e-11 |
| White mean, float32 | 74.81342315673828 | 78.17576599121094 |
| Gray mean, float32 | 58.288795471191406 | 60.0891227722168 |
| White/gray mean float32 bits equal to native plot-derived mean | Yes / yes | Yes / yes |
| `select0sm` interior bitwise equal with independent means | 54/54 | 124/124 |
| `select0sm` all vertices bitwise equal | 102,764/102,764 | 101,454/101,454 |
| MRI/surface reads, sphere preflight, histogram, Numba JIT | 20.16 s | 24.63 s |
| Subsequent first-candidate MRI match including Numba JIT | 1.31 s | 1.58 s |

`HISTOplot` writes counts with ten decimal places. Those text plots prove
agreement at that precision, but cannot by themselves establish the native
float32 bit pattern of every smoothed bin. Parsing the printed values back to
float32 agrees with only 79/63 LH and 54/55 RH white/gray smoothed bins; the
same text rounds many small values to an adjacent float32 value. This is a
reference serialization limit, not evidence of an underlying histogram
mismatch. The independently computed white/gray means and the complete
first-candidate MRI-matched vertex arrays are bitwise equal. These Python
seconds include JIT and input preparation and are not paired kernel timing
against native FreeSurfer.

Evidence: [LH report](lh.independent_intensity_histogram.json),
[RH report](rh.independent_intensity_histogram.json), stored [LH exact float32 bins](lh.independent_intensity_histogram.tsv)
and [RH exact float32 bins](rh.independent_intensity_histogram.tsv), and the
[validator](validate_topology_intensity_histogram.py). The bin files preserve the exact Python float32 bit patterns as hexadecimal;
their SHA-256 values are recorded in the JSON reports.

## Native first-defect search endpoint

The bounded native logs continue beyond the initial ten candidates. They
record 13 LH / 16 RH best-fitness improvements and end after generation 27 /
53. These are reference milestones for an eventual Python GA replay, not
Python search results.

| Native first-defect endpoint | LH | RH |
| --- | ---: | ---: |
| Last generation logged | 27 | 53 |
| Last best snapshot serial | `best_13` | `best_16` |
| Last optimal fitness, log precision | −100.44 | −97.753 |
| Surface Euler after defect 0 | −11 | −7 |
| Complete official `orig.premesh` Euler | 2 | 2 |

The last saved generation geometry differs in face count from the best
snapshot (LH 202,429 versus 202,477; RH 200,235 versus 200,279 faces).
The search winner must therefore be replayed explicitly. Subsequent defects
and final compaction still separate these checkpoints from `orig.premesh`.

## Remaining to reach `orig.premesh`

The source still needs independent candidate-patch curvature fits, intensity
histograms for subsequent defects, MRI face/vertex likelihoods, normal-dot and
quadratic-curvature likelihoods, volume unlikelihood, mutation, crossover, elite ranking over
later generations, vertex deletion, final best-order replay, subsequent
13 LH / 9 RH defects, and final surface compaction. The native saved
`select0s` checkpoint changes exactly 54 LH / 124 RH interior vertices
from `select0` (maximum displacement 3.643 / 7.280 mm). Subsequent
`select0sm` changes the same vertex counts (maximum displacement relative
to `select0` 4.734 / 11.109 mm); face order remains identical.

The first-defect white/gray intensity histograms and global first-pass curvature
histograms are independently generated and match the installed reference.
The candidate-patch curvature fit and other likelihood terms still prevent
full-candidate fitness verification. First-candidate topology,
smoothing, and MRI matching coordinates are exact against the saved native
checkpoints, but independent full-candidate fitness inputs have not yet been
established. Native saved first-defect best snapshots
are `best_13` LH and `best_16` RH; neither is the completed subject's final
`orig.premesh`.
The RH all-candidate MRI sort still swaps two unused-in-`select0` new edges
at ranks 16,570/16,571 because one Python score is one float32 ULP higher.
Later GA candidates can depend on that order.

Evidence: [LH fitness](lh.initial_fitness_control-host_report.json),
[RH fitness](rh.initial_fitness_control-host_report.json),
[LH independent patch](lh.independent_first_patch_control-host.json),
[RH independent patch](rh.independent_first_patch_control-host.json),
[validator](validate_topology_initial_fitness.py), and
[preceding edge/segment report](TOPOLOGY_INDEPENDENT_SEGMENTS.md).

## Independent first-pass principal-curvature histogram: historical first attempt

The new [principal-curvature fitter](../../../../src/fnit/recon_all/topology_principal_curvature.py)
starts from ordered `orig.nofix` and `qsphere.nofix`, regenerates the canonical
sphere and defect labels, retains the source's sphere normals at ripped defect
vertices, and fits the original surface over two neighbor rings. The
[histogram module](../../../../src/fnit/recon_all/topology_curvature_histogram.py)
then constructs the 100-bin k1/k2 PDFs, float32 truncated means, and joint
100×100 PDF. Neither module reads native curvature or plot files as input.
The [focused validator](validate_topology_curvature_histogram.py) reads native
outputs only after prediction. It uses float32 throughout; no fp16/bfloat16
reduction is involved.

I captured `curv.dat` immediately after the official `mris_fix_topology`
first histogram pass in isolated subject copies with the same
`-ga -seed 1234 -threads 1 -correct_defect 0 -verbose_low` options. Each
capture stopped before the search. The six official k1/k2 plot and joint MGH
SHA-256 values matched the frozen LH/RH fixtures exactly. A private binary
compiled from pinned FreeSurfer source commit `d932c45b` with the same
options produced different curvature at defect vertices (first difference
above 1e-3: LH vertex 2887, RH vertex 3552). The installed official binary,
not that private build, is therefore the numerical reference below.

| First global curvature check | LH | RH |
| --- | ---: | ---: |
| k1 bin positions equal at six printed decimals | 100/100 | 100/100 |
| k1 PDF counts equal at ten printed decimals | 53/100 | 68/100 |
| k2 bin positions equal at six printed decimals | 0/100 | 1/100 |
| k2 PDF counts equal at ten printed decimals | 34/100 | 65/100 |
| Predicted / official k1 mean | −0.0947988033 / −0.0947326869 | −0.0928762481 / −0.0927938297 |
| Predicted / official k2 mean | −0.0327636786 / −0.0333479270 | −0.0354278274 / −0.0355196632 |
| Joint PDF cells bitwise equal | 0/10,000 | 0/10,000 |
| Largest joint cell absolute difference | 0.00113952 | 0.000136911 |
| Native curvature vertices within 1e−3, k1 / k2 | 102,516 / 102,565 of 102,764 | 101,294 / 101,317 of 101,454 |
| Median vertex absolute difference, k1 / k2 | 2.38e−7 / 2.46e−7 | 2.38e−7 / 2.46e−7 |
| Python fit plus histogram, including first-call JIT | 16.74 s | 13.29 s |

The first predicted-versus-official vertex difference above 1e−3 is LH
vertex 24 and RH vertex 344; both exchange nearly equal-magnitude positive
and negative principal curvatures. The first histogram count discrepancy is
k1 bin 5 LH / bin 23 RH. LH k2 differs already at bin 0: predicted start
−1.836925 versus official −1.846650. The 100×100 joint PDF has a global
normalization denominator, so a count difference can change all cells'
float32 bits. In this first attempt, the Python 3×3 normal-equation solve and analytic
two-eigenvalue formula did not reproduce the source `MatrixSVDInverse` and
VNL eigensystem. The installed binary also differed from the private source
build at defect vertices. Later sections trace and resolve these specific
first-pass differences.

The first candidate's `qcurv_ll` consumes this joint PDF and independently
fitted candidate-patch k1/k2. The global PDF matched only after the correction documented below; the
candidate patch fit has not been ported, so `qcurv_ll` and complete candidate
fitness remain unverified. The independently matched white/gray histogram and `select0sm`
coordinate results above are unaffected. Native first-pass curvature dumps
are auxiliary output checks, never inference inputs. [LH report](lh.independent_curvature_histogram.json)
and [RH report](rh.independent_curvature_histogram.json) contain input and
reference hashes, first bin mismatches, means, and vertex-level precision.

### First curvature vertex: source-order SVD inverse discrepancy

The first LH output difference above 1e−3 is vertex 24, outside every defect.
An isolated GDB probe of the pinned-source binary stopped at that vertex's
`MatrixSVDInverse`, its following `MatrixMultiply(Ut,z)`, and
`MatrixEigenSystem`. Independent Python and the native code both supplied
exact float32 `Gram=[[73,42,21],[42,84,42],[21,42,73]]` and
`RHS=[−1,−2,−1]`. NumPy's direct solve yields quadratic coefficients
`A≈−2.76e−18, B=−0.0238095243, C=0`, making the two curvature magnitudes
equal and placing the negative one first. Native VNL float SVD yields small
nonzero inverse terms (for example inverse[0,2]=`+4.65661287e−10` and
inverse[2,0]=`−1.39698386e−9`). Multiplying the captured native inverse by
the independently computed RHS in `MatrixMultiply` float32 order reproduces
all four native Hessian elements exactly:
`[[-9.31322575e−10,−0.0476190522], [−0.0476190522,+1.11758709e−8]]`.
The source then places `+0.047619056` before `−0.0476190485`.

This establishes the earliest observed numerical discrepancy at the
**SVD inverse substitution**, before eigenvalue sorting. NumPy float32/64
direct solves, NumPy/SciPy float32 pseudoinverses, and SciPy float32
`gesvd`/`gesdd` do not reproduce the nine native inverse elements; the best
of these trials matched 2/9 elements bitwise. The LH histogram remained at
53/100 k1 PDF counts, 34/100 k2 counts, and 0/10,000 joint cells bitwise.
No empirical tie rule or native plot data was incorporated into the
prediction. The [structured trace](lh.curvature_first_mismatch_trace.json)
records the matrices, solver comparisons, and scope of this diagnosis.

### Source-order VNL inverse and installed-reference audit

The preceding table records the **pre-SVD** baseline. The new
[Numba SVD inverse](../../../../src/fnit/recon_all/topology_vnl_svd.py)
ports VNL's 3×3 float LINPACK `ssvdc(job=21)` and FreeSurfer's float
`MatrixSVDInverse` multiplication order. At LH vertex 24, the independently
constructed Gram and RHS exactly equal the pinned-source GDB values, and the
port matches all 9 U, 3 singular values, 9 V, 9 inverse elements, and 4
Hessian entries bitwise. The [focused test](../../../../tests/recon_all/test_topology_vnl_svd.py)
passes. A separate isolated C build of vendored ITK LINPACK and BLAS matched
all U/W/V bits for vertex 24 and 100 fixed-seed random SPD matrices; the
Numba port also matched all 101 C cases. The first arithmetic difference in
our earlier prototype was an extra float rounding before the double-returning
`sqrt` product in `snrm2` and `srotg`.

The installed FreeSurfer 8.2 executable is stripped and statically linked.
Its `.comment` section identifies GCC 4.8.5, and embedded paths identify ITK
4.13.2 VNL. The private pinned-source probe links conda ITK 5.3 VNL instead.
The two vendored `ssvdc.c` and `snrm2.c` files are byte-identical; the
`vnl_svd.hxx` numerical loop is unchanged. The build difference is established,
but it alone does not prove a changed curvature result. Pinned-source GDB
values are used for the local SVD operation test; the installed binary remains
the only global curvature acceptance reference.

One temporary private-debug directory also contained `k1.plt`, `k2.plt`, and
`mri_k1_k2.mgh` with hashes **different** from the frozen installed outputs.
A comparison against that directory has been explicitly retained only as a
[private-source LH diagnostic](lh.private_source_curvature_vnl.json) and
[RH diagnostic](rh.private_source_curvature_vnl.json). The separate
`/tmp/topology_curvature_capture_official_20260926/{lh,rh}` reference has
input-surface and all eight histogram/vertex dump SHA-256 values identical to
the original frozen installed-reference reports. The following numbers use
only that SHA-verified installed reference; the [LH](lh.independent_curvature_vnl.json)
and [RH](rh.independent_curvature_vnl.json) structured reports record the hashes.

| Source-order VNL inverse vs installed FreeSurfer | LH | RH |
| --- | ---: | ---: |
| k1 raw 100-bin counts / printed PDF equal | 54/100 | 71/100 |
| k2 raw 100-bin counts / printed PDF equal | 34/100 | 69/100 |
| joint raw count cells equal | 8591/10,000 | 9875/10,000 |
| joint float32 PDF cells equal | 0/10,000 | 0/10,000 |
| first six-decimal k1 difference | vertex 335 | vertex 96 |
| first six-decimal k2 difference | vertex 27 | vertex 237 |
| first k1 difference above 1e−3 | vertex 2887 | vertex 3552 |
| first k2 difference above 1e−3 | vertex 2906 | vertex 4013 |

The earliest printed differences are outside the defect. At LH vertex 27,
independent Gram and RHS match the pinned-source debugger's 9+3 float values,
and source-order SVD produces the same four Hessian float values. The remaining
k2 difference begins at the eigenvalue operation: the float analytic formula
returns `−0.1049724817`, whereas pinned-source VNL returns `−0.1049725041`.
At RH vertex 96 the float formula returns k1 `+0.7804085016`, versus
pinned-source VNL `+0.7804084420`. Installed FreeSurfer prints the same
six-decimal values as the pinned-source probe at both vertices. ITK 4.13.2
VNL copies the float Hessian to double before calling EISPACK `rs_`; a
source-derived double 2×2 calculation cast back to float reproduces both
pinned-source eigenvalues bitwise. The later direct installed-binary check and
bilateral histogram rerun are reported below. Full first-pass curvature PDFs
and later candidate `qcurv_ll` remained unaccepted at this intermediate check.


### Installed-binary eigen correction and next defect-vertex divergence

The installed `mris_fix_topology` executable used for this check has SHA-256
`0ee25cee2760cde9858d2043f1a9d42be9c10cd613dc43d9620203feeadab02c`.
A GDB breakpoint at its first-pass eigen call (`0x4973d0` in this exact
non-PIE binary) captures the 2×2 Hessian before VNL and the float32 eigen
outputs after VNL. The direct **installed** LH v27 Hessian bits are
`bf2c8780/3f132774/3f132774/bf2f8a4e`, eigen bits
`bfa0992a/bdd6fbd3`; RH v96 Hessian bits are
`3ed8e136/be6f5230/be6f5230/3f209899`, eigen bits
`3f47c8d9/3e8a80b5`. All four Hessian and both eigen float32 elements
match the independent Python fit bitwise for each hemisphere. These are not
inferred from the six-decimal `curv.dat` text.

The independent double-eigen fitter was rerun against the immutable installed
reference. Both surface hashes, all native histogram hashes, and the native
vertex dump hash still equal the earlier frozen-reference reports. The
[updated LH](lh.official_vnl_doubleeig_curvature_histogram.json) and
[RH](rh.official_vnl_doubleeig_curvature_histogram.json) reports contain the
full counts and SHA evidence.

| Double-eigen independent fit vs installed FreeSurfer | LH | RH |
| --- | ---: | ---: |
| k1 raw 100-bin counts / ten-decimal PDF equal | 61/100 | 77/100 |
| k2 raw 100-bin counts / ten-decimal PDF equal | 34/100 | 70/100 |
| joint raw count cells equal | 8591/10,000 | 9879/10,000 |
| joint float32 PDF cells equal | 0/10,000 | 0/10,000 |
| first six-decimal k1 and k2 mismatch | v2887 | v2198 |
| first k1 difference above 1e−3 | v2887 | v3552 |
| first k2 difference above 1e−3 | v2906 | v4013 |

The new first printed mismatches are both in independent defect component 1.
The installed binary's `marked`, `marked2`, `marked3`, and `ripflag` fields are
all zero at both vertices during this first global curvature fit. These
runtime marker fields are distinct from the independently reconstructed defect
component membership and cannot be used interchangeably. The [focused probe](probe_topology_curvature_first_divergence.py) rebuilds the
ordered one- and two-ring, defect labels, normal, Gram/RHS, SVD inverse,
Hessian, and eigenvalues from the frozen mesh bytes without using a native
plot or vertex dump for inference. The installed debugger's complete ordered
`vtotal` lists match the independently built lists entry by entry (LH 15/15,
RH 20/20); installed `vnum` is 6 for both. The first verified numerical
split is already in the vertex normal before Gram construction:

| First remaining vertex | Installed nx,ny,nz | Independent nx,ny,nz | matching float32 bits |
| --- | --- | --- | ---: |
| LH v2887 | `−0.458992809, 0.821101964, 0.339289188` | `−0.459228843, 0.820654571, 0.340051502` | 0/3 |
| RH v2198 | `−0.146728858, −0.888523579, −0.434737116` | `−0.146715656, −0.888523579, −0.434741795` | 1/3 |

Installed and independent Gram, RHS, inverse, Hessian, and eigen bits then
all differ at these vertices. The official installed `curv.dat` reads
LH `1.118498/0.799434` and RH `0.515520/−0.252862`; the private pinned-source
build instead reads LH `1.116202/0.798967` and RH `0.515528/−0.252865`,
matching the independent six-decimal values. The two builds have byte-identical
frozen surface inputs; their different outputs are kept as separate reference
boundaries. FreeSurfer's pinned source calls `MRIScomputeMetricProperties`
after restoring original positions and before this first curvature fit; the
specific canonical/marked normal operation causing the installed discrepancy
is still under investigation. The [LH installed trace](lh.installed_curvature_first_divergence.json)
and [RH installed trace](rh.installed_curvature_first_divergence.json) retain
all selected-vertex float32 bits and hash gates. The first-pass histogram and
full topology stage remain **unaccepted**.


### Installed-binary canonical sphere stage boundary (2026-09-26)

A read-only, three-breakpoint GDB check of the SHA-verified installed binary
(`0ee25cee2760cde9858d2043f1a9d42be9c10cd613dc43d9620203feeadab02c`)
now separates the normal discrepancy above from its upstream coordinate cause.
The breakpoints are immediately after `MRISprojectOntoSphere` (`0x416a66`),
after five `MRISsmoothOnSphere` passes (`0x416a79`), and after
`MRIScenterSphere` (`0x472f27`). Each snapshot contains the first mismatching
curvature vertex and its six **ordered** one-ring vertices, with native xyz
and normal float32 bit patterns. The [focused validator](validate_topology_projection_phases.py)
independently rebuilds all three xyz stages from the frozen `qsphere.nofix`
and `orig.nofix` faces. Its [LH report](lh.installed_projection_phases.json)
and [RH report](rh.installed_projection_phases.json) retain both sides' bits.
The native GDB snapshots are validation references only, never algorithm inputs.

| Installed vs independent Python, 7 vertices × 3 xyz bits | LH | RH |
| --- | ---: | ---: |
| projected | 21/21 | 21/21 |
| five-pass smoothed | 14/21 | 8/21 |
| after centerSphere | 14/21 | 8/21 |
| centered normal bits | 0/21 | 1/21 |

The earliest observed stage split is therefore **inside the five smoothing
passes**, before `MRIScenterSphere` or the later curvature normal/Gram/SVD
fit. At the five-pass boundary, LH v2887 z is Python `c208b4a3` versus
installed `c208b4a5` (two float32 ULP), and RH v2197 x is Python `c15d5dec`
versus installed `c15d5ded` (one ULP). The native selected xyz bits are
unchanged between the smooth and centerSphere breakpoints; the latter does
recompute the normal. Projection source-order alternatives (ratio and
subtractive form) both give 21/21 selected xyz bits on each hemisphere after
the required initial center. Changing that projection formula cannot explain
the remaining first difference.

The report input hashes are LH `orig.nofix`
`ef00440f651026e4d099bd282f3e4e2cfbfffdc1e0877684fe22aa2948542b79`,
`qsphere.nofix` `8afffc181e2e05b8349099845e7ffe3b7a6a36e241d5689c4b6503c020485647`,
and RH `orig.nofix` `45d53fa51554b6f58f04ac37ee8eff1814fd31c001178e7648af2eb97b9f01cf`,
`qsphere.nofix` `b93d519dd0da418ca2fbfed01617af4939b3d7528e91a7d5f73eaae279a9977a`.
These are the same frozen installed-reference inputs used for the bilateral
histogram gate. The earlier statement that the first split was in the vertex
normal remains true for the *curvature fit* boundary; the three-stage trace
now identifies smoothing as the earliest upstream split. Five-pass sphere
smoothing and the full official histogram were **unaccepted at this pre-fix boundary**.


### Exact global first-pass curvature histogram after double sqrt correction

The source of the five-pass sphere divergence above was the Numba lowering of
`np.sqrt(float(squared))` inside the two-pass
`mrisSphericalProjectXYZ` translation. The squared sum was already a correct
float32 value; `float(squared)` still reached a float32 sqrt in the compiled
Numba function. At LH v2887 on smooth pass 1, the installed double sqrt was
`99.999980468748092`, while the previous Numba computation was float32
`99.9999771118164`; at RH v2198 the corresponding values were
`99.99509753607924` and `99.9950942993164`. Explicit
`np.sqrt(np.float64(squared))` at both projection points gives the installed
first and second projection bits. The input neighbor order, float32 sum, and
float32 mean were already exact for all seven selected vertices on each side.
The [first-pass probe](probe_topology_smooth_first_pass.py) and fixed
[LH](lh.installed_smooth_first_pass.fixed.json) /
[RH](rh.installed_smooth_first_pass.fixed.json) reports show 21/21 mean and
21/21 first-pass projected xyz bits for each hemisphere.

After that **two-line numerical correction**, the installed three-stage
7-vertex checks are 21/21 xyz float32 bits at projection, five-pass
smoothing, and centerSphere on both hemispheres. The recomputed centered
normals are also 21/21 bits for these vertices. The fixed
[LH](lh.installed_projection_phases.fixed.json) and
[RH](rh.installed_projection_phases.fixed.json) reports retain native and
independent bits. The earlier failed stage reports remain above as the
reproducing pre-fix diagnostic, not current acceptance results.

The full installed VERTEX RAM snapshots were taken immediately after the
five-pass smoothing and after centerSphere at the same pinned binary
breakpoints. The [full validator](validate_topology_canonical_full.py)
compares independent coordinates with those snapshots, for every vertex and
axis; it does not read native snapshots to compute the coordinates. The
[LH full report](lh.installed_canonical_full.json) records exact equality
at both boundaries for 102,764/102,764 vertices and 308,292/308,292
float32 components. The [RH full report](rh.installed_canonical_full.json)
records 101,454/101,454 vertices and 304,362/304,362 components at both
boundaries. Maximum absolute coordinate difference is 0 mm. The installed
RAM snapshot SHA-256 values are; the four binary snapshots and GDB capture
logs are retained on control-host in `topology_fix/native_canonical_reference_20260926/`:

| Installed snapshot | LH | RH |
| --- | --- | --- |
| after smooth | `4745d2644f763d33828734d4b696a7b816e674f639e4b869627ccae6fdae7376` | `93945d5f4516c8cc1302e6b204ccdc067b1b3481d07b7d3ab172c4020b325e38` |
| after centerSphere | `3932d126fdc686e1d07a2c1b7ebac30f0b926b9c93e7c4b450bc729be420fa82` | `8c06ee2f950788f75575fc2ab143c4529c34879ecc685fb67772dc4d16900f7b` |

The [global first-pass histogram validator](validate_topology_curvature_histogram.py)
was then rerun against the **installed**, SHA-frozen `k1.plt`, `k2.plt`,
`mri_k1_k2.mgh`, and first `curv.dat`. It derives every curvature from the
frozen `orig.nofix`/`qsphere.nofix`; those native outputs enter comparison
only. The [LH](lh.official_sqrt64_curvature_histogram.json) and
[RH](rh.official_sqrt64_curvature_histogram.json) reports both set
`frozen_reference_sha256_match=true` and `histogram_matches_native=true`.

| Independent first-pass vs installed reference | LH | RH |
| --- | ---: | ---: |
| k1 raw-count bins / printed PDF bins | 100/100 / 100/100 | 100/100 / 100/100 |
| k2 raw-count bins / printed PDF bins | 100/100 / 100/100 | 100/100 / 100/100 |
| k1/k2 mean float32 bits | equal / equal | equal / equal |
| ordered vertices with equal printed k1 and k2 | 102,764/102,764 | 101,454/101,454 |
| joint raw count / float32 PDF cells | 10,000/10,000 / 10,000/10,000 | 10,000/10,000 / 10,000/10,000 |

The focused topology preflight suite passes 5/5 tests, including two native
float-bit projection cases. This accepts the **global first-pass canonical
sphere and curvature histogram** for this frozen LH/RH input and installed
FreeSurfer 8.2 reference. It does not establish byte-identical per-vertex
curvature before the six-decimal native text formatting, nor later candidate
patch k1/k2 and `qcurv_ll`. The defect GA, complete `mris_fix_topology`, and
end-to-end `recon-all` remain unaccepted.
