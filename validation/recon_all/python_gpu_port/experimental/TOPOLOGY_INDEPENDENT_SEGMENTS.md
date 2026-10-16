# `mris_fix_topology`: first-defect old-edge segmentation

This bounded Python/Numba port follows the pinned FreeSurfer 8.2 source
`d932c45b7941662ea380a05efef580568b98d41a`. It verifies the first
`-ga -seed 1234 -threads 1` defect in each hemisphere of the frozen `fs_sub01`
subject. It does not run a FreeSurfer executable or use a native captured EDGE
array during prediction. The standalone `mris_fix_topology` stage, later
defects, genetic search, and final `orig.premesh` output remain unported.

## Official stage and source path

The completed native subject used the same command for LH and RH, changing
only the hemisphere:

```text
mris_fix_topology -threads 1 -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh -ga -seed 1234 -threads 1 fs_sub01 lh
```

`mris_fix_topology/mris_fix_topology.cpp` reads `surf/{hemi}.qsphere.nofix`,
`surf/{hemi}.orig.nofix`, `surf/{hemi}.inflated.nofix`, `mri/brain.mgz`, and
`mri/wm.mgz`, then calls `MRIScorrectTopology` in
`utils/mrisurf_defect.cpp`. That function detects defects, constructs and
MRI-scores candidate edges, sorts them, calls `segmentIntersectingEdges`
for the original edges, generates patch orderings, searches genetic patch
candidates, repeats for all defects, and returns the corrected surface. The
driver restores corrected original coordinates, copies WM volume geometry,
and writes `surf/{hemi}.orig.premesh`.

| Fixed native surface | LH | RH |
| --- | ---: | ---: |
| Input vertices / faces / edges | 102,764 / 205,560 / 308,340 | 101,454 / 202,936 / 304,404 |
| Input Euler number | −16 | −14 |
| Detected defects | 14 | 10 |
| Final vertices / faces / edges | 101,689 / 203,374 / 305,061 | 100,555 / 201,106 / 301,659 |
| Final Euler number | 2 | 2 |
| Native log stage duration | 1.5 min | 1.9 min |

The log durations cover complete native hemispheric runs and are not a speed
comparison with this partial Python primitive. The full `orig.premesh` topology
and surface data are not produced by this port.

## Independent first-defect validation

[`topology_edge_segments.py`](../../../../src/fnit/recon_all/topology_edge_segments.py)
implements the source's inside-first ordering, pairwise edge intersection,
and old-edge cluster assignment. The validator starts from the saved
`qsphere.nofix`, `orig.nofix`, and `brain.mgz` inputs. Existing Python preflight
computes canonical sphere, defect labels/status, vertex translation, and the
unsorted candidate edge table. The Python MRI score probe computes all edge
scores, and a host C library `qsort` with the pinned comparator reproduces
source sorting. This is a validation-only sort helper, not a FreeSurfer call.
Native diagnostic labels, EDGE captures, and saved annotation are opened only
after prediction for comparison.

| First defect check | LH | RH |
| --- | ---: | ---: |
| Python defect label / status differences | 0 / 0 | 0 / 0 |
| Candidate table differences | 0 / 6,456 | 0 / 26,560 |
| Bitwise MRI score matches | 6,388 / 6,456 | 26,334 / 26,560 |
| Maximum absolute MRI score error | 0.000030518 | 0.000030518 |
| All candidate sort position differences | 0 | 2 |
| Original edge sort position differences | 0 / 164 | 0 / 403 |
| Saved segment annotation differences, all vertices | 0 / 102,764 | 0 / 101,454 |
| First-defect interior and border annotation matches | 245 / 245 | 441 / 441 |

The two RH all-candidate ordering differences are ranks 16,570 and 16,571:
a one-unit float32 score difference swaps two **new** candidate edges. Neither
is an original edge used by this segmentation. This ordering difference must
be resolved or evaluated during the genetic search before claiming complete
topology parity. Saved annotation colors compress segment identities, so exact
annotation does not by itself establish that every internal edge assignment
matches native.

The machine-readable [report](independent_segments_control-host_report.json) and
[validator](validate_topology_independent_segments.py) reproduce these checks.
The earlier [preflight](../topology_preflight_control-host_report.json),
[edge-score probe](TOPOLOGY_EDGE_SCORE.md), and
[first-candidate patch probe](TOPOLOGY_FIRST_PATCH.md) describe adjacent
bounded steps. The same Python-generated sorted candidate table has now also
passed the independent initial patch check on both hemispheres; that test is
separate from the old-edge grouping result here. No complete recon-all
reconstruction was rerun for these checks.
