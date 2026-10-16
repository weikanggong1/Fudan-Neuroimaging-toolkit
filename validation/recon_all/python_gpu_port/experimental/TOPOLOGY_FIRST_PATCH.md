# `mris_fix_topology -ga`: first defect patch gate

This isolated probe tests the pinned FreeSurfer 8.2 source commit
`d932c45b7941662ea380a05efef580568b98d41a` on the frozen `fs_sub01`
LH/RH inputs. It does **not** replace the topology stage in the recon-all runner.

## Reproduced slice

`topology_patch_edges_probe.py` reconstructs the initial corrected surface,
including intact vertex links whose incident faces were removed. Its original
mode uses the captured native post-`qsort` `EDGE` array to accept
nonintersecting edges, form and orient patch faces, and compute the ordered
vertex coordinates. Its `--independent` mode derives the defect maps,
candidate edges, MRI scores, and sort order from `qsphere.nofix`,
`orig.nofix`, and `brain.mgz`; it does not read the captured EDGE table.
The first-candidate edge acceptance kernel is now in the Python package at
`src/fnit/recon_all/topology_first_candidate.py`.

`topology_ga_segments_probe.py` also follows the pinned
`segmentIntersectingEdges` order for old mesh edges. Its predicted native
annotation matches all first-defect interior and border vertices: LH 245/245,
RH 441/441. The predicted old-edge counts are 164 and 403, with 3 and 1
unassigned edges after native-style grouping. Native annotations compress edge
segment identities to vertex colors, so this check does not prove that every
internal edge-to-segment assignment is identical.

| Fixed first defect | LH | RH |
| --- | ---: | ---: |
| Native first-candidate vertices | 102,764 | 101,454 |
| Bitwise equal ordered vertex coordinates | 102,764 | 101,454 |
| Unchanged base faces in the native prefix | 202,307 | 200,071 |
| New accepted edges, exact set | 223/223 | 487/487 |
| New faces, exact set and ordered rows | 170/170 | 364/364 |
| First-candidate ordered surface gate | Pass | Pass |

The reference is `rh.defect_0_select0` from each hemisphere's native
`mris_fix_topology -correct_defect 0 -ga -seed 1234 -threads 1 -save ...`
run. FreeSurfer uses the `rh.` snapshot prefix for both hemispheres. The
`-correct_defect` run exits with code 252 after the requested defect; this is
the native diagnostic termination, not an ordinary successful stage exit.
Native wall times including snapshot writes were 19.37 s for LH and 49.53 s
for RH; these are reference-capture times, not a fair speed benchmark.

## Independent first candidate

The capture-free `--independent` mode also produced the same first-candidate
ordered vertices and faces on both hemispheres. The LH/RH 223/487 accepted
new edges and 170/364 added ordered faces matched the native `select0`
snapshots exactly. It consumes Python-generated defect labels/status and
MRI-scored candidate edges; the snapshots serve only as final references.
The two RH new edges whose Python scores differ by one float32 ULP swap
ranks 16,570/16,571; both are absent from the native first candidate, and
the independent candidate remains exact. This does not establish equality
for later genetic candidates or the final `best` patch.

The independent fixed-subject results are
[`lh.independent_first_patch_control-host.json`](lh.independent_first_patch_control-host.json)
and [`rh.independent_first_patch_control-host.json`](rh.independent_first_patch_control-host.json).
The upstream capture-free old-edge segmentation is reported in
[`TOPOLOGY_INDEPENDENT_SEGMENTS.md`](TOPOLOGY_INDEPENDENT_SEGMENTS.md).
The first candidate's Euler/face validity gates and initial GA fitness
selection are reported in
[`TOPOLOGY_FITNESS_SEARCH.md`](TOPOLOGY_FITNESS_SEARCH.md).

## Required final-output gate still fails

| Native first-defect GA output | LH `best_13` | RH `best_16` |
| --- | ---: | ---: |
| Final raw snapshot vertices | 102,764 | 101,454 |
| Final raw snapshot faces | 202,477 | 200,279 |
| Added face overlap with the first candidate | 46/170 | 30/364 |
| First different ordered face index, zero-based | 202,310 | 200,071 |
| First candidate matches final best | No | No |

The native final `orig.premesh` after **all** defects has 101,689 vertices and
203,374 faces for LH, and 100,555 vertices and 201,106 faces for RH. The first
defect's raw best snapshot is therefore not the final stage output either.
The remaining steps include `segmentIntersectingEdges`, `generateOrdering`,
MRI/geometry patch fitness, GA mutation/crossover and selection, vertex
elimination, post-patch matching, subsequent defects, and final surface
compaction. The first-candidate gate cannot establish their parity. No
topology fixer or end-to-end recon-all parity is claimed.

## Reproducible evidence

Machine-readable results are `lh.topology_patch_first_candidate_control-host.json`,
`rh.topology_patch_first_candidate_control-host.json`, and the corresponding
`lh.topology_ga_segments_control-host.json` and `rh.topology_ga_segments_control-host.json`.
The frozen native files
are in `$P/topology_fix` on control-host, where
`P=/data/workspaces/freesurfer_synth/work/reconall_python_gpu_20260925`:

| Reference file relative to `$P/topology_fix` | SHA256 |
| --- | --- |
| `capture_lh/lh.edge.after.bin` | `87c5b1810efb592b586216fe49fcac318d84d65ce9d75d1746056879ae28227f` |
| `capture_rh/rh.edge.after.bin` | `eb343f6b15c69bca928d9ed5b04e82a86a5e93e8bd1a7f2961ef5b2b72c9b95f` |
| `ga_initial_lh/save/rh.defect_0_select0` | `6830e155603859dcb046570d6825e17c107581d953f8d6467ff2bc8834e9892e` |
| `ga_initial_lh/save/rh.defect_0_best_13` | `b66803847ca6c274c8db2fd520bd0606770253049914a5d1e79b26ec47e4e6f4` |
| `ga_initial_rh/save_full/rh.defect_0_select0` | `6829ecdf1c95536d5584d45276f8642054541aa5cc1aabd3f8129c4154f66a5e` |
| `ga_initial_rh/save_full/rh.defect_0_best_16` | `298eb0cc88b1c6784a7df72bea76a16a47a7191adc79b7180b1070be0483b252` |

The RH first candidate was also saved by an earlier bounded diagnostic run.
Its raw file hash differs only because the FreeSurfer surface header records
the write time; its ordered vertex and face arrays are bitwise identical.

To rerun the comparison in the synced checkout on control-host:

```bash
cd "$P/repo_ci/ci_checkout"
R=/data/workspaces/freesurfer_synth/work/reconall_main_20260924
PYTHONPATH=src:. "$R/venv/bin/python" \
  validation/recon_all/python_gpu_port/experimental/topology_patch_edges_probe.py \
  --diagnostics "$P/topology_fix/native_diag" \
  --captures "$P/topology_fix" \
  --patch "$P/topology_fix/ga_initial_lh/save/rh.defect_0_select0" \
  --final-patch "$P/topology_fix/ga_initial_lh/save/rh.defect_0_best_13" \
  --hemi lh
```

Use `ga_initial_rh/save_full` and `best_16` with `--hemi rh` for RH. The probe
runs on CPU with NumPy/Numba and is solely a parity diagnostic.
