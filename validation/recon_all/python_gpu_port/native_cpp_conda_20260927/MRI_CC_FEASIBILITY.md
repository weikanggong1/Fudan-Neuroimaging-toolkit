# `mri_cc`: Python integration and C++ replacement decision

## Decision and function

The existing Python port `run_mri_cc(aseg_file, norm_file, output_file, lta_file)` reads a SynthSeg-derived `aseg.auto_noCCseg.mgz` and intensity-normalized `norm.mgz`, labels the corpus callosum as 251–255, and writes `aseg.auto.mgz` plus `transforms/cc_up.lta`. It uses NumPy, SciPy, and nibabel on CPU and needs neither an installed FreeSurfer runtime nor an extra model file. The recon-all runner now calls it after `mri_ca_normalize`, then propagates the edited volume to its current `aseg.presurf.mgz` and `aseg.mgz` approximation before WM, fill, surfaces, and statistics. This does **not** add the later official aseg edits.

The official equivalent command, run from a subject's `mri` directory, is:

```bash
SUBJECTS_DIR=/path/to/subjects FREESURFER_HOME=/path/to/freesurfer \
FS_LICENSE=/private/path/license.txt \
/path/to/freesurfer/bin/mri_cc -aseg aseg.auto_noCCseg.mgz \
  -o aseg.auto.mgz -lta transforms/cc_up.lta subject01
```

The Python call is:

```python
from fnit.recon_all.mri_cc_python import run_mri_cc
run_mri_cc("aseg.auto_noCCseg.mgz", "norm.mgz", "aseg.auto.mgz",
           "transforms/cc_up.lta")
```

The full runner exposes the same stage through `run_recon_all_python(...)` or `fnit-recon-all` without an extra switch. The paired official command below was **only** used as a reference; the integrated runner calls Python.

## Frozen candidate-input pair, 2026-09-27

Both programs ran on `headcw` from independent copies of the current Conda E2E candidate inputs. `synthseg.rca.mgz` is exactly equal to the archived official `aseg.auto_noCCseg.mgz` at all 16,777,216 voxels. Frozen file SHA-256: input segmentation `b87845b23c32c1f75d6082674f869bbf53f899dd15b9b0c5ef7c6b6dc17a8279`, candidate `norm.mgz` `d29b3b59787987b4de51db8f19129554be399ef5e2b7527c3c1b350c53b13f13`. Official FreeSurfer was 8.2.0, source commit `d932c45b7941662ea380a05efef580568b98d41a`. Both commands exited 0.

| Check | Result |
|---|---:|
| Official edits from input | 2,263 voxels |
| Python edits from input | 2,263 voxels |
| Official versus Python `aseg.auto.mgz` | **0 / 16,777,216 unequal voxels** |
| Official versus Python decoded MGH header / voxel payload | **0 / 0 differing bytes** |
| Official versus Python `cc_up.lta` maximum matrix difference | 7.63 × 10⁻⁶ |
| Official versus Python CC labels 251–255 | 714, 270, 292, 327, 660 each |
| Official and Python outputs versus archived official `aseg.auto.mgz` | 0 unequal voxels each |
| Official command wall / maximum RSS | 99.16 s / 354,196 KiB |
| Python function wall / whole validator process wall / maximum RSS | 17.34 s / 18.96 s / 772,440 KiB |

The single-run official-command/Python-function wall ratio is **5.72×** on headcw. These are individual timings with different command versus in-process boundaries, so they are not a robust multi-run speed benchmark. The compressed `.mgz` SHA-256 values differ even though the MGH header and voxel payload match; the strict comparator evaluates decoded data and required metadata rather than compressed-file SHA. The [same-input machine report](mri_cc_candidate_input_pair.json), [official timing](mri_cc_official_headcw.time), and [Python timing](mri_cc_python_headcw.time) contain the underlying values.

## Runner handoff check

The updated `_segment_callosum` helper ran on a **new isolated subject directory** containing copies of the same frozen candidate inputs; it did not edit the prior E2E subject or the official reference. Its `aseg.auto.mgz` has the same dtype (`>f4`), exact MGH header, and identical affine as archived official output. The earlier candidate `aseg.auto.mgz` differed by 2,263 voxels; the integrated output differs by **0**. Archived official `aseg.auto.mgz` and `aseg.presurf.mgz` are themselves voxel-identical, so the integrated `aseg.presurf.mgz` also differs by **0**. The copied candidate `aseg.mgz` still differs from official final `aseg.mgz` by **104,986** voxels because later official postprocessing remains absent. The helper took 18.80 s inside the independent headcw script; the full script, including imports and comparisons, took 22.20 s. See [integration result](mri_cc_integrated_stage.json).

The focused handoff test checks use of both input files, creation of the CC LTA, and propagation of edited labels into presurf/aseg without overwriting the SynthSeg source. It passed **1/1** against the synced FNIT snapshot on `gpucw1` (3.14 s pytest run). The post-integration **full recon-all E2E has not been rerun in this report**, so downstream improvements are not yet claimed.

## Conda C++ feasibility boundary

Upstream `mri_cc/CMakeLists.txt` has a standalone `mri_cc` target linked against FreeSurfer's `utils` library. A separate Conda CMake build tree configured successfully in 9.46 s using the existing GCC/G++/Fortran 11.4, ITK 5.4.7, and glibc 2.17 sysroot environment. Compilation was stopped before producing a binary to avoid load during the paired sphere benchmark. The Python port then proved exact on the current input and was substantially faster in the headcw pair, so this stage does not justify another C++ runtime dependency. **No source-built C++ `mri_cc` timing or accuracy claim is made.** The isolated probe directory is under `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_cpp_conda_20260927/mri_cc_feasibility_20260927`; no probe binary or script is added to the public build.
