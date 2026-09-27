# Conda topology GA with Python sphere preflight

The native-topology option now selects the pinned Conda-built
`mris_fix_topology_fnit`. Its Python preflight reproduces the sphere used by
FreeSurfer 8.2 at the start of defect search. The source build changes the
centering handoff plus float-versus-double `tanh` and `sqrt` calls in defect smoothing. The original
Conda-built `mris_fix_topology` is retained for diagnostic comparison only.
This stage runs on CPU. No FreeSurfer executable is required at runtime.

## Inputs, outputs, and use

`write_centered_topology_sphere(input_qsphere, output_centered)` reads one
ordered FreeSurfer triangle `qsphere.nofix` and writes a new triangle surface
with the same faces and footer. It returns a dictionary with `input`,
`output`, `vertices`, `faces`, `iterations`, and `seconds`. It uses
nibabel, NumPy and Numba. It does not repair topology or read official
intermediate data.

`run_topology_ga_conda(subject, hemisphere, binary, assets)` requires the
following files beneath a FreeSurfer-style subject directory:

| Input | Role |
| --- | --- |
| `surf/{hemi}.orig.nofix` | Original-space ordered mesh |
| `surf/{hemi}.inflated.nofix` | Inflated coordinates |
| `surf/{hemi}.qsphere.nofix` | Quick spherical mesh |
| `mri/brain.mgz`, `mri/wm.mgz` | Matched intensity image and WM mask |
| `binary` | Patched `mris_fix_topology_fnit` from this repository's Conda build |
| `assets` | Data-only FreeSurfer 8.2 lookup assets |

The function writes `surf/{hemi}.topology-centered.sphere`,
`surf/{hemi}.orig.premesh` and
`scripts/{hemi}.topology-ga-fnit.log`. It returns a dictionary containing
`hemisphere`, `preflight` (the first function's report), `output`, `log`,
`command`, and `native_seconds`. The Conda binary **requires**
`FNIT_CENTERED_COORDS`; an unset path aborts rather than silently reverting
to its different centering result. The Python function sets it automatically.
It verifies the binary's substitution marker and output.

From the repository root after activating the Conda build environment:

```bash
bash tools/build_recon_all_fs_cpp_conda.sh /path/to/clean/freesurfer-8.2-source /path/to/conda-build
python -m fnit.recon_all.topology_conda_ga /path/to/subjects/sub01 lh \
  /path/to/conda-build/bin/mris_fix_topology_fnit /path/to/assets \
  --report /path/to/lh-topology.json
```

Repeat for `rh`. The optional `native_topology=True` recon-all runner
selects this patched binary and calls the same stage API, then
`remesh_surface(..., iterations=3)` to produce `surf/{hemi}.orig`.
The default runner still has other approximate surface and segmentation steps;
selecting this stage does not establish equivalent end-to-end reconstruction.

The official FreeSurfer 8.2 command from `subject/scripts` is:

```bash
mris_fix_topology -threads 1 -mgz -sphere qsphere.nofix \
  -inflated inflated.nofix -orig orig.nofix -out orig.premesh \
  -ga -seed 1234 -threads 1 sub01 lh
mris_remesh --remesh --iters 3 ../surf/lh.orig.premesh ../surf/lh.orig
```

## Real-T1 same-input result

The frozen input is a real T1, with original MRI SHA-256
`c99c246200cc35479b6b8cd691457985b66f2062d4a37a0c3592ff1b678b985c`.
FNIT's corrected `filled` and `norm` generated all eight LH/RH
`orig/inflated/smoothwm/qsphere.nofix` surfaces exactly against the
completed FreeSurfer 8.2 subject. The narrow topology comparison then read
those independent surfaces plus the frozen official `brain.mgz` and
`wm.mgz` through the scratch subject. Thus it isolates this stage; it is
not a current whole-subject E2E result. No patient image is committed.

| Accurate same inputs | LH `orig.premesh` vertices/faces | RH vertices/faces | Relation to official |
| --- | ---: | ---: | --- |
| Official FreeSurfer 8.2 | 101,689 / 203,374 | 100,555 / 201,106 | Reference |
| Unmodified Conda C++ | 101,737 / 203,470 | 100,655 / 201,306 | Both differ |
| Python exact sphere + original Conda C++ | 101,689 / 203,374 | 100,575 / 201,146 | LH vertices/faces all ordered and float32 exact; RH differs |
| Python exact sphere + double-`tanh` Conda C++ | 101,689 / 203,374 | 100,548 / 201,092 | LH exact; RH still differs |
| Python exact sphere + double-`tanh`/`sqrt` Conda C++ | 101,689 / 203,374 | 100,555 / 201,106 | Both hemispheres have exact ordered vertices, faces, and float32 coordinates |

Python preflight sphere coordinates and ordered faces are **exact** against
official diagnostic centered spheres for LH 102,764/205,560 and RH
101,454/202,936. With both C++ math fixes, LH `orig.premesh` has all
101,689 vertices, 305,067 float32 coordinate components and 203,374 ordered
faces exact; RH has all 100,555 vertices, 301,665 coordinate components and
201,106 ordered faces exact. File SHA-256 differs because the surface comment
records creation provenance. The controlled baseline with only double
`tanh` had its first RH difference after smoothing defect 0's first
crossover: three vertices differed by at most 1.19e-7 mm, then MRI matching
amplified the error and changed GA decisions. Promoting the type-2 smoother's
`sqrt` argument to double restored this crossover and the complete RH mesh.

The observed final Conda C++ stage took 56.92/95.48 s (LH/RH); Python
preflight took 6.34/3.55 s in a separate call. Earlier official same-input
runs took 60.98/82.10 s. Shared node load varied, so these observations are
not a controlled speed comparison. The frozen paired inputs include official
MRI/WM and independent FNIT nofix surfaces; current connected E2E white/pial
and atlas metrics are not thereby accepted.

The connected Python remesher was also run on both newly generated
`orig.premesh` surfaces, with no official surface supplied to the call.
Against the archived official `orig`, LH had 106,622/106,622 vertices,
319,866/319,866 float32 coordinates and 213,240/213,240 ordered faces exact;
RH had 105,541/105,541 vertices, 316,623/316,623 coordinates and
211,078/211,078 faces exact. It took 100.05/119.32 s on the shared
headcw node. The [earlier independent remesher benchmark](../../validation/recon_all/python_gpu_port/REMESH_VALIDATION.md)
compared Python and official commands on a different frozen premesh; its
timings are not paired with this generated premesh.

The volume geometry metadata fields match numerically for both hemispheres.
Its `filename` field correctly names each subject's own `mri/wm.mgz`;
that path naturally differs between the official and scratch subjects. The
candidate `orig` lacks 425 bytes of official FreeSurfer build/run provenance
tags, so full footer bytes and file SHA-256 differ. The [bilateral orig
geometry](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/accurate_filled_initial/curvature_trial/lh_fnit_sqrt_orig_vs_official.json)
and [footer/metadata report](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/accurate_filled_initial/curvature_trial/orig_metadata_report.json)
record the distinction; the RH geometry JSON is beside the LH file.

The [bilateral JSON evidence](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/accurate_filled_initial/curvature_trial/)
records exact-coordinate/face comparisons, timings, and the diagnostic
checkpoint boundary before the final sqrt fix. The Conda build pins source file hashes before patching; its
`ldd` and SHA-256 outputs are recorded in the build directory. No GPU
allocation occurs in this stage.
