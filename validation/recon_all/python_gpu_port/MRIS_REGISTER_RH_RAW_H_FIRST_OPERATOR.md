# RH `debug0056`: first independent smoothwm curvature difference

This bounded check used the installed FreeSurfer 8.2.0-1 `mris_register`
(`75137b92fcbed63b441e6214b5b923d7664f05d00c454b62da724dec26636f80`)
and the frozen `rh.smoothwm`
(`b67ceb95069d865bb11f6c2b20bac99eab9f19982b73d3c9c43735b7bedbe7ca`).
The native run used `-N 1` solely to reach the raw H checkpoint before the
registration schedule. The captured native raw H hash is
`820582a7731addfb50c7453bbce534ed4fb314a590b4749191fee06af04031a1`,
identical to the previous raw H reference used for default RH `debug0056`.
It does not validate a `-N 1` final surface as the default result.

The [native capture](mris_register_rh_smoothwm_native_capture_control-host.json) and
[full-array comparison](mris_register_rh_smoothwm_first_operator_control-host.json)
are produced by the [read-only GDB capture](capture_mris_register_rh_smoothwm_normal.gdb)
and [Python probe](probe_mris_register_rh_smoothwm_first_operator.py). The
remote arrays remain in
`/data/workspaces/freesurfer_synth/work/reconall_python_gpu_20260925/mris_register_rh_raw_vertex_probe_20260926`.
The capture and comparison source hashes are respectively
`14f93050b112a67329eee9bcdd0797ee4e7ced7147b3a09fa928bfde8fef7f27`
and `6acb5bb3bd60d9f48731cc04cd28a3d375e718e5fee01dc1e03f119265923a2b`.
The two JSON hashes are respectively
`13d9abff5cbf4f41e95c4355fddb638074b910a65b3d500ae2dd869448023d3f`
and `1d7da3428a3ff9fc2a4ed4420259c1b0eb7e99d4bd5a25f3d8c643733b9de0c1`.

| Ordered checkpoint | Exact elements | Total | First difference | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: |
| smoothwm XYZ | 316,623 | 316,623 | none | 0 mm |
| vertex normals XYZ | 316,623 | 316,623 | none | 0 |
| raw H | 14,713 | 105,541 | vertex 0 | 9.1552734375e-5 |

At vertex 0, native H is `-0.6379879713058472` and Python H is
`-0.6379880905151367` (difference `1.1920928955078125e-7`). At vertex
57,378, native H is `-3.9725027084350586` and Python H is
`-3.9724111557006836` (difference `9.1552734375e-5`). The unchanged Python
source hashes are `4b9f8faff6da19abb6fb6e1c687ad7d631d38630b47cf34018b9dc583967b9e9`
for `mris_register_nonlinear.py` and
`9944484056d08df18c95fef8dc70262c18bd44e2695fdea950ab93d812e16b1c`
for `mris_register_smoothwm.py`; the probe recomputed its existing predicted
raw H hash `d41e4f96299e838926651765023d9ece80f3451eed39629d08d6a4bd9e72666f`.

The earliest established difference is **after the exact input positions and
normals, within the raw H curvature fit**. The installed checkpoint did not
expose its pre-fit tangent axes, ordered three-hop neighborhood, design rows,
Gram matrix, right-hand side, SVD inverse, Hessian, or 2×2 eigenvalues. The
local FreeSurfer source shows a float32 matrix multiplication, an
`OpenSvdcmp`-based explicit inverse and a 2×2 eigensystem; the current Python
code uses `torch.linalg.svd` and directly adds fitted coefficients. Those are
plausible causes, **not demonstrated first differing operators**. The local
source checkout is diagnostic; installed-binary values remain the acceptance
reference.

The next falsifiable, single-vertex check is to capture native values at
vertices 0 and 57,378 in this order: (1) pre-fit tangent axes and ordered
three-hop neighbor IDs, (2) the first `m_U` row and `v_z`, (3) `m_Ut*m_U` and
`m_Ut*v_z`, (4) inverse and fitted coefficients, then (5) Hessian/eigenvalues.
Compare each stage bitwise against the same independently computed Python
stage and stop at the first mismatch. Change only that operation and rerun
the 105,541-value raw H comparison before retesting RH `debug0056`. No
production correction or final `sphere.reg` claim follows from this check.
