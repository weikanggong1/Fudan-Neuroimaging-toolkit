# `mris_fix_topology`: validated Python pre-search stages

The pinned FreeSurfer 8.2.0 source is commit
`d932c45b7941662ea380a05efef580568b98d41a`. Recon-all calls
`mris_fix_topology -threads 1 -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh -ga -seed 1234 -threads 1 fs_sub01 {lh,rh}`.

[`topology_preflight_python.py`](../../../src/fnit/recon_all/topology_preflight_python.py)
implements the fixed deterministic path through input Euler counting,
spherical projection/smoothing/centering, intersecting-edge defect detection,
connected components, borders, retention status, one-ring hulls, and the
ordered vertex/face translation into the genetic-search base mesh. The first
filter of candidate edges against existing base edges, including the ordered
candidate endpoint pairs and `used` flags, is also translated. It
is not called by the reconstruction entry point.

The official intermediate sphere was captured in a separate subject copy with
`-diagonly`, `DIAG=0x8`, and `DIAG_VERBOSE=1`. The four native defect maps were
captured in the same run. A separate `-correct_defect 0` run wrote the native
`vtrans` and `ftrans` logs before exiting after the first corrected defect.
The three official archived defect maps also match the diagnostic run byte for
byte. No complete subject reconstruction was repeated.

| Frozen `fs_sub01` check | Left | Right |
| --- | ---: | ---: |
| Input vertices / faces / edges | 102,764 / 205,560 / 308,340 | 101,454 / 202,936 / 304,404 |
| Input Euler number | -16 | -14 |
| Maximum Python-to-native sphere vertex distance | 0.0000329 mm | 0.0000315 mm |
| Vertices above 0.0001 mm | 0 | 0 |
| Defects | 14 | 10 |
| Different vertices in `defect_labels` | 0 / 102,764 | 0 / 101,454 |
| Different vertices in `defect_borders` | 0 / 102,764 | 0 / 101,454 |
| Different vertices in `defect_chull` | 0 / 102,764 | 0 / 101,454 |
| Different vertices in `defect_status` | 0 / 102,764 | 0 / 101,454 |
| Different entries in `vtrans` | 0 / 102,764 | 0 / 101,454 |
| Different entries in `ftrans` | 0 / 205,560 | 0 / 202,936 |
| First defect candidate edges / edges discarded | 6,903 / 447, exact native counts | 28,920 / 2,360, exact native counts |
| First defect ordered candidate endpoint differences | 0 / 6,456 | 0 / 26,560 |
| First defect ordered `used` flag differences | 0 / 6,456 | 0 / 26,560 |

The Python sphere-fitting loops take 14/11 iterations versus native 8/12;
their stored coordinates still satisfy the vertexwise tolerance above. The
Python geometric detector sees two additional partner faces on the left,
but those faces add no vertices: both defect components and all four diagnostic
maps are exact. This difference does not establish full algorithm parity on
other inputs. The [JSON report](topology_preflight_control-host_report.json) records
input/reference SHA-256 values and exact counts. The
[validator](validate_topology_preflight.py) recomputes the comparisons from
the frozen files; [focused tests](../../../tests/recon_all/test_topology_preflight_python.py)
cover mesh invariants, spherical edge crossing, and ordered base translation.
For the next boundary, a diagnostic-only `qsort` interposer captured the native
`EDGE` table immediately before and after its score sort, then exited before
genetic search. These native calls took 3.92 s (LH) and 5.17 s (RH). The
[edge-table validator](validate_topology_edge_table.py) confirms every Python
endpoint pair and `used` flag in original candidate order against the native
table; the [JSON report](topology_edge_table_control-host_report.json) records the
capture hashes. The diagnostic interposer is
[`topology_capture_edges.c`](topology_capture_edges.c). The alternate saved
native sphere image is not the exact geometry of this `-ga` run: its small
coordinate differences change five LH crossing decisions. The validation
recomputes the active Python spherical path, which yields the exact native
ordered table.

The next unmatched operation in `mrisTessellateDefect_wkr` assigns MRI-based
scores to each surviving candidate edge. This relies on smoothed original
surface normals, per-vertex gray/white samples with median filtering, and
trilinear MRI samples along each edge. The native scores and sorted order are
captured for comparison, but no Python scores have been claimed. Subsequent
seeded genetic search writes new faces and vertex positions. The first left
defect has 118 candidate vertices and 6,456 edges after the matched filter;
the right has 241 vertices and 26,560 edges. Until its chosen edges, ordered
`orig.premesh` faces, and coordinates match the official result,
`mris_fix_topology` remains an unimplemented end-to-end stage. Native final
surfaces here have Euler number 2 and 101,689/100,555 ordered vertices,
respectively.
