# RH `debug0056`: independent raw H and first smoothwm update

The installed FreeSurfer 8.2.0-1 `mris_register` binary SHA-256 is
`75137b92fcbed63b441e6214b5b923d7664f05d00c454b62da724dec26636f80`.
The frozen `rh.smoothwm` SHA-256 is
`b67ceb95069d865bb11f6c2b20bac99eab9f19982b73d3c9c43735b7bedbe7ca`.
The installed raw H array SHA-256 is
`820582a7731addfb50c7453bbce534ed4fb314a590b4749191fee06af04031a1`.
The `-N 1` GDB run was stopped inside the raw H fit, before any smoothwm
registration update. Its 65.78 s wall time includes setup and the diagnostic
sulc prefix and is not a per-stage native benchmark.

The [installed RAM capture](mris_register_rh_smoothwm_fit_native_control-host.json)
at vertices 0 and 57,378 and the
[paired comparison](mris_register_rh_smoothwm_fit_comparison_control-host.json)
show where the old implementation first differed:

| Ordered stage | Vertex 0 | Vertex 57,378 |
| --- | ---: | ---: |
| XYZ, normal and both tangent axes | 12/12 exact | 12/12 exact |
| Three-hop ordered neighbors | 36/36 exact | 37/37 exact |
| Quadratic design and height | 144/144 exact | 148/148 exact |
| Gram and right-hand side | 12/12 exact | 12/12 exact |
| Old `torch.linalg.svd` explicit inverse | 1/9 exact; max error 3.54e-8 | 0/9 exact; max error 8.01e-5 |

The first differing operator was therefore the 3×3 SVD inverse, not the
mesh, neighborhood, normal, tangent basis or Gram/RHS accumulation. Reusing
the repository's source-order VNL inverse (`topology_vnl_svd.svd_inverse_3`)
matched the installed inverse **9/9** and fitted coefficients **3/3** at
both vertices; see the [one-operation report](mris_register_rh_smoothwm_vnl_inverse_control-host.json).
After that one change, directly summing coefficients matched 76,241/105,541
raw H values, maximum error 4.77e-7; see the
[trace-only report](mris_register_rh_smoothwm_vnl_trace_control-host.json).
The installed code instead builds a float32 2×2 Hessian, solves its
eigensystem, then stores `(k1+k2)/2`. Reproducing that order matched
**105,541/105,541 raw H values bitwise**, with zero maximum error and the
same raw-array SHA-256; see the
[all-vertex eigen report](mris_register_rh_smoothwm_vnl_eigen_control-host.json).

The corrected production
[`mris_register_smoothwm.py`](../../../src/fnit/recon_all/mris_register_smoothwm.py)
has SHA-256 `b4ef242acdcfd2f237f0496208486ab50f5ce9fa54e4e78d42a5e80e7d5f907d`.
Its unchanged dependencies are `mris_register_nonlinear.py`
(`4b9f8faff6da19abb6fb6e1c687ad7d631d38630b47cf34018b9dc583967b9e9`)
and `topology_vnl_svd.py`
(`54474eab97ec17c450a9d51f85c46936e905eeb873e0e1631afad520cef60ff5`).
The production [raw H retest](mris_register_rh_smoothwm_production_raw_control-host.json)
independently matched **105,541/105,541**, with the same native array hash
and 10.66 s observed CPU time including the Numba JIT startup. The current
fit transfers each Gram/RHS batch to CPU and runs the source-order inverse
and eigen recombination there, even when the input tensor is on GPU. Numba
is already a declared dependency in the old worktree and in the clean
worktree's `recon-all-python-stages` extra.

The first default smoothwm update was then replayed from the independently
matched RH `debug0055` sulc seed. The [fixed epoch report](mris_register_rh_default_epoch0056_independent_fixed_control-host.json)
records identical source/seed/atlas/raw-H/native-output hashes to the earlier
native-raw conditional trial. The selected `dt` is
`2.542891502380371`, and the saved RH `debug0056` surface matches **all
105,541 ordered vertices exactly**, maximum error 0 mm. This is an
independent-input first-update result; no native raw H was injected.

Reproduction uses the [native fit capture](capture_mris_register_rh_smoothwm_fit.gdb),
[stage comparison](probe_mris_register_rh_smoothwm_fit.py),
[source-order inverse comparison](probe_mris_register_rh_smoothwm_vnl_inverse.py),
[all-vertex raw H diagnostic](probe_mris_register_rh_smoothwm_full_vnl.py),
existing [production raw H comparator](probe_mris_register_smoothwm_curvature.py)
and existing [single-epoch comparator](probe_mris_register_sno2_epoch.py).
The full-array production and epoch reports, rather than the two selected
vertices alone, are the acceptance evidence.

The corrected independent LH raw H and all 45 normal smoothwm updates have
passed; see the [LH continuation report](MRIS_REGISTER_LH_SMOOTHWM_BOUNDARY.md).
The later RH `debug0058` line-search divergence was traced to the spring
SSE area-ratio precision. The corrected probe matches `debug0056`–`debug0097`
at all 105,541 ordered vertices per update; see the
[native RH58 fit](mris_register_rh_default_epoch0058_native_line_fit_accelerator-a-1.json),
[RH83 fit](mris_register_rh_default_epoch0083_native_line_fit_accelerator-a-1.json)
and [corrected continuation](mris_register_rh_default_epoch0083_0097_lcorr_f32_accelerator-a-1.json).
The bounded RH path reaches the official final sphere geometry. End-to-end
reconstruction, vertex/ROI metrics and a full GPU timing comparison remain open.

## Archived evidence hashes

| Artifact | SHA-256 |
| --- | --- |
| Native two-vertex fit JSON | `266c696621bb631145b4074b9716fdfcf082dcefbb1b520789c34f4c7b172beb` |
| Paired first-difference JSON | `c325130e6048289071016db9a9bec26a13e0e1e372697dc42f17279206075289` |
| VNL inverse two-vertex JSON | `7a4eddfb30bc775c31d7005a0c1aca03f95b1b65f1d1b151f775b3096f7a2e0c` |
| All-vertex VNL eigen JSON | `1c26a9ea1012044d153b1fd21fd5f430ff729e71a5a70e03a125e324b5b9047a` |
| Production raw H JSON | `bc2040a8f4199e7553c9e32bd003805ad6d765761a62839aba2a6485d9a67a70` |
| Independent RH `debug0056` JSON | `e3afbf45e3010f20fac5aec37c0503914f7c5596170ee92fefd0a8a4829a1e8e` |
