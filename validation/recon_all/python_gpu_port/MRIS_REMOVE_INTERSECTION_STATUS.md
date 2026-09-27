# `mris_remove_intersection` Python stage status

The fixed FreeSurfer 8.2 `recon-all.cmd` invokes `mris_remove_intersection ../surf/{lh,rh}.orig ../surf/{lh,rh}.orig` immediately after three remesh iterations. Its native logs report **0 marked intersection vertices for both hemispheres** and the pinned source's `MRISremoveIntersections` returns without moving vertices when its intersection-face count is zero. The independent [remesh validation](REMESH_VALIDATION.md) matched its pre-call ordered geometry exactly on the same `fs_sub01` T1.

[`mris_remove_intersection_python.py`](../../../src/fnit/recon_all/mris_remove_intersection_python.py) implements a native-free intersection check for this case. A bounding-sphere/box search enumerates triangle candidates, excludes faces sharing a vertex, and tests triangle intersection. With zero intersections the entry point retains the entire input surface file byte for byte, including FreeSurfer volume geometry. With positive intersections it raises before writing; the native iterative soap-bubble repair has **not** been ported. The implementation runs in NumPy/SciPy on CPU. It is called by the current runner immediately after Python remesh; its positive-intersection repair branch still raises.

## Fixed-subject comparison

An installed FreeSurfer 8.2.0 `mris_remove_intersection` was rerun on separate outputs from the archived `fs_sub01` `orig` files; Python read those same files. No official subject file was changed. The post-remesh/native-removal geometry is already identical for this fixed T1, as the native pipeline log and independent remesh check establish.

| Hemisphere | Vertices / faces | Native marked vertices | Python marked vertices | Ordered coordinates / faces | Volume geometry | Python bytes vs input |
| --- | ---: | ---: | ---: | --- | --- | --- |
| LH | 106,622 / 213,240 | 0 | 0 | exact / exact | exact | exact |
| RH | 105,541 / 211,078 | 0 | 0 | exact / exact | exact | exact |

The fresh native output differs in file hash because its writer adds a command-line footer. The structured reports are [LH](experimental/lh.remove_intersection_validation.json) and [RH](experimental/rh.remove_intersection_validation.json); the reusable comparator is [`validate_remove_intersection_python.py`](experimental/validate_remove_intersection_python.py). A synthetic pair of overlapping tetrahedra marked the **same 7 of 8 vertices** as native `-map`; a separated pair marked zero in both programs. See the [control report](experimental/synthetic.remove_intersection_validation.json) and [tests](../../../tests/recon_all/test_mris_remove_intersection_python.py) (5 passed on headcw). These controls do not validate the unimplemented repair branch or prove detector equivalence on other real subjects.

The archived original native `@#@FSTIME` wall records are 4.20 s LH and 4.46 s RH. Separate Python CLI calls took 1.19 s LH and 1.24 s RH, including startup and surface copy. Other CPU stage checks ran concurrently, so these are **not paired speed trials** and do not establish a speedup.

This stage replaces the fixed `fs_sub01` no-intersection call in the current runner. A fresh candidate LH `orig` from the same T1 also had 0 detected intersections and byte-identical input/output in a 2.48 s gpucw1 CPU check; the new connected whole subject is still being evaluated. Inputs with real self-intersections remain an explicit native-free recon-all blocker; no vertex thickness, area, volume, or curvature output is newly validated by this stage alone.
