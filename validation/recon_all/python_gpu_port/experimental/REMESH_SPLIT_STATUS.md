# `mris_remesh --remesh --iters 3`: split phase only

This historical probe covers only the edge-splitting phase of iteration 0. The subsequent three-iteration Python stage and its fixed-T1 output validation are documented in [`REMESH_VALIDATION.md`](../REMESH_VALIDATION.md). The reference is FreeSurfer 8.2.0 source commit `d932c45b7941662ea380a05efef580568b98d41a`, `mris_remesh/{mris_remesh.cpp,remesher.cpp,remesher.h}`.

The pinned source was rebuilt in isolated control-host scratch with passive double-coordinate/int-face dumps. Its **one-iteration final** ordered vertices and faces were exactly equal to a fresh run of the installed FreeSurfer executable on both hemispheres. This establishes the source diagnostic as a valid reference for these inputs; the diagnostic binary is not used by Python inference.

| Fixed input | Native split passes | Vertices after split | Faces after split | Python vs source dump | Python CPU wall |
| --- | --- | ---: | ---: | --- | ---: |
| LH `orig.premesh` | 118,835, 2,459, 476, 24, 0 | 223,483 | 446,962 | Every float64 coordinate and ordered int32 face exact | 6.90 s |
| RH `orig.premesh` | 116,335, 2,469, 933, 256, 0 | 220,548 | 441,092 | Every float64 coordinate and ordered int32 face exact | 6.72 s |

The Python probe follows the source's first-seen edge order, max-heap ordering by edge length and index, midpoint insertion, and triangle/edge updates. Its pass counts, ordered vertices and ordered faces match the native diagnostic exactly. CPU peak RSS was 563,312/572,356 kB for LH/RH. These Python times cover the split probe and comparison, while the native one-iteration commands include collapse, smoothing, I/O and intersection removal; they are **not** a speed ratio.

Frozen input SHA-256: LH `06eb4b9603e21e07a59ec02e21c97d33d4e8217e6c64eb0986a07bc42abf09aa`, RH `c5808d16eb02753a92d19cbd45c127b9c8e17921803be4c1f7ef5e6a29622d79`. Native split-dump SHA-256: LH `182e20a9a19b6d42488e42871b63bf1fc08ea1ac9843bf5eef75bd1227b3bf7a`, RH `f5cb6de395fc786b22bc2a626efad6e7c205428c558457decef471379aee13bf`. The raw diagnostics and logs are in `work/reconall_python_gpu_20260925/remesh_probe` on control-host. Run [`remesh_split_probe.py`](remesh_split_probe.py) with the corresponding `orig.premesh` and `*-after_split_0.bin` paths to repeat the full array comparison.
