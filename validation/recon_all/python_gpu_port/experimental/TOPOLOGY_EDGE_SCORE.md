# `mris_fix_topology` first-defect edge score probe

This is an isolated diagnostic for the pinned FreeSurfer 8.2 source at
`d932c45b7941662ea380a05efef580568b98d41a`. It is **not** a replacement
for `mris_fix_topology`, and it is not wired into recon-all. The reference is
the frozen `fs_sub01` input to the exact `-ga -seed 1234 -threads 1` command.
No full reconstruction was rerun.

`topology_edge_score_probe.py` reproduces the source-order MRI score assigned
to every candidate `EDGE` before `qsort` for defect 0. It reads the native
pre/post-sort captures from `topology_capture_edges.c`, the original surface,
defect labels, and `brain.mgz`. It applies the source's two original-surface
smoothing passes, face-normal unitization for gray/white reference samples,
two median passes, original-face normals for each edge, 11 along-edge MRI
sample locations, float32 RAS-to-voxel conversion, and the `+100` penalty on
new edges. The diagnostic replays the pinned comparator using the host C
library `qsort`; no FreeSurfer binary is called by the probe.

| Hemisphere | Edges | Bitwise equal scores | Maximum absolute score error | Native qsort replay | Python-score qsort |
| --- | ---: | ---: | ---: | ---: | ---: |
| LH | 6,456 | 6,388 | 0.000030518 | 6,456/6,456 positions | 6,456/6,456 positions |
| RH | 26,560 | 26,334 | 0.000030518 | 26,560/26,560 positions | 26,558/26,560 positions |

The first non-bitwise score difference is LH pre-sort edge 7:
151.7467803955 native versus 151.7467956543 Python. It is RH pre-sort edge
65: 168.3198852539 native versus 168.3199005127 Python. All scores on both
sides are within 0.001, but the acceptance gate for the complete algorithm is
stricter because the score controls search order. The remaining RH ordering
swap is ranks 16,570 and 16,571: native ties edges `(100303, 2636)` and
`(100349, 2197)` at 138.5243530273, whereas the latter Python score is
138.5243682861. The C library qsort replay of native scores is exact, so
this swap comes from the score's last float32 unit, not the comparator.

The full stage remains unimplemented. After sorting, pinned source calls
`mrisComputeOptimalRetessellation` for the genetic patch search, repeats
correction across the other defects, and writes `orig.premesh`. Neither that
search nor the output surface has a Python parity result. Do not use this
probe to claim end-to-end topology or recon-all agreement.

The frozen reports are [`lh.edge_score_probe_control-host.json`](lh.edge_score_probe_control-host.json)
and [`rh.edge_score_probe_control-host.json`](rh.edge_score_probe_control-host.json).
The native capture command and SHA-256 values are recorded in
[`topology_edge_table_control-host_report.json`](../topology_edge_table_control-host_report.json).
