# Conda topology GA with Python sphere preflight

The native-topology option now selects the pinned Conda-built
`mris_fix_topology_fnit`. Its Python preflight reproduces the sphere used by
FreeSurfer 8.2 at the start of defect search. The source build changes the
centering handoff and one float-versus-double `tanh` call. The original
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
| Python exact sphere + double-`tanh` Conda C++ | 101,689 / 203,374 | 100,548 / 201,092 | LH again exact; RH still differs |

Python preflight sphere coordinates and ordered faces are **exact** against
official diagnostic centered spheres for LH 102,764/205,560 and RH
101,454/202,936. The LH patched result has all 101,689 vertices,
305,067 coordinate components and 203,374 ordered faces exact. The RH patched
mesh has seven fewer vertices and 14 fewer faces than official, so
vertexwise downstream measures cannot be compared by index.

One observed paired API run spent 6.84/5.80 s in Python preflight and
71.57/76.29 s in Conda C++ (LH/RH). The earlier official same-input runs took
60.98/82.10 s. Shared node load varied; these times do not support an
equivalent-reconstruction speed claim.

The first remaining RH difference is in defect 0's **first crossover and
mutation**: initial candidate patch fitness values are all ten exact after
the double-`tanh` fix, but crossover fitness prints -108.03 for official
and -107.96 for Conda. Its raw saved mesh is exact; its smoothed mesh differs
at three vertices by at most 1.19e-7 mm, then the MRI-matched mesh differs
at 52 vertices (maximum 0.2013 mm). This amplification changes the GA path.
The next source numeric difference remains unresolved. This is a strict
parity failure and later surface/ROI metrics are not yet accepted.

The [bilateral JSON evidence](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/accurate_filled_initial/curvature_trial/)
include each exact-coordinate/face comparison, API timing, and checkpoint
boundary. The Conda build pins source file hashes before patching; its
`ldd` and SHA-256 outputs are recorded in the build directory. No GPU
allocation occurs in this stage.
