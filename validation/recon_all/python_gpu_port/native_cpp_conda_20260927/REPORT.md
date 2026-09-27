# Conda-built C++ recon-all: pre-CC baseline and same-input validation

**Status:** Six FreeSurfer 8.2 C++ programs were compiled with a Conda toolchain, connected to FNIT's Python/CUDA recon-all profile, and run from one T1 into an empty subject directory. This **pre-CC v1 baseline** completed but was not numerically equivalent to official `recon-all` (6/138 fixed outputs passed). A subsequent v2 change connected the existing Python `mri_cc` stage and reduced the frozen candidate `aseg.auto` error to zero; its new full E2E run is reported separately after completion. This report separates same-input program checks from the pre-CC full T1-to-output comparison.

## Build and execution

- Upstream source: FreeSurfer commit `d932c45b7941662ea380a05efef580568b98d41a`; the tested source archive SHA-256 is `c31a23b8d8c73a513715eb011b7b39b244586de516e8af5d50aebb4c34900476` and its 2,592-file content-manifest SHA-256 is `df2ace4b904dc722090782895c251ceac65b8c52f192c2abcf7ce3daabc83585`. The build script changes only CMake's Python executable selection in a copied source tree; image and surface algorithms are unchanged. The six programs are `mri_em_register`, `mris_fix_topology`, `mris_inflate`, `mris_sphere`, `mris_place_surface`, and `mris_register`.
- Actual environment: Python 3.11, PyTorch 2.5.1/CUDA 11.8, Conda GCC/G++/Fortran 11.4, ITK 5.4.7 and glibc 2.17 sysroot on gpucw1. All six binaries launch, and their `ldd` records contain no installed FreeSurfer runtime. The full provenance is in [`bin.sha256`](bin.sha256), [`conda-explicit.txt`](conda-explicit.txt), [`build-provenance.txt`](build-provenance.txt), and [`BUILD_ENV_NOTES.md`](BUILD_ENV_NOTES.md). The one-command environment YAML passed a Conda solver dry-run; an additional environment was not installed from that YAML. The actual environment was installed through equivalent phased Conda commands.
- Run: `fnit-recon-all` with `--native-bin-dir`, `--native-topology`, `--native-sphere`, `--native-surface-metrics`, `--native-registration`, `--device cuda:1`, and 4 CPU threads. The invocation cleared `FREESURFER_HOME`, `FSLDIR`, and `SUBJECTS_DIR`; it used only external verified weights/templates and an external private `FS_LICENSE`. No installed FreeSurfer executable or 0.70 GB runtime package was needed. The six source-built programs still execute on **CPU**.
- Input SHA-256: `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`. Exit status 0; 29 top-level stages; process elapsed 2714.66 s, pipeline stage report 2706.64 s. See [`run.json`](run.json), [`e2e.status`](e2e.status), and [`e2e.time`](e2e.time).

## Full-subject accuracy and time

The fixed 138-output comparator found **6 passed, 81 present but failed, and 51 missing** candidate outputs. The six exact files are `mri/orig.mgz`, `mri/orig/001.mgz`, `mri/rawavg.mgz`, `mri/nu.mgz`, `mri/T1.mgz`, and `mri/synthseg.rca.mgz`. The same voxel grid and affine are used for MRI comparisons. `brainmask.mgz` differs at 38 voxels; `aseg.auto.mgz` and `aseg.presurf.mgz` each at 2,263; final `aseg.mgz` at 107,249; `wm.mgz` at 435,774; `filled.mgz` at 90,420. These counts are file-value differences, not all spatial boundary errors.

LH `orig` has 117,076 vertices in the candidate versus 106,622 official; RH has 120,273 versus 105,541. Therefore ordered vertex comparisons of white/pial/sphere, thickness, area, volume and curvature are **not defined** between complete subjects. No cortical stats file passes: for example, mean LH thickness is 2.90596 mm candidate versus 2.07854 mm official. MRI label and atlas differences propagate into region statistics. Every file result and the diagnostic spatial comparison are in [`strict_138.json`](strict_138.json), [`comparison.json`](comparison.json), and [`STRICT_OUTPUT_ANALYSIS.md`](STRICT_OUTPUT_ANALYSIS.md).

The candidate reported 2706.6 s; the archived official same-T1 run reported 6789.6 s. These were not paired under controlled load and the outputs differ. **The ratio 0.399 is an observed wall-time ratio, not an equivalent-reconstruction speedup.** The candidate's largest top-level stages were RH surface 640.1 s, LH surface 509.8 s, GCA registration 358.2 s, LH sphere registration 269.1 s, and RH sphere registration 231.2 s. All stage and native substep times, official command log times and selected output results are in [`BENCHMARK.md`](BENCHMARK.md) and [`benchmark_summary.json`](benchmark_summary.json). Official command wrappers overlap; their listed times must not be summed into an end-to-end total.

## Isolated program checks

The paired runs below fed the same frozen candidate inputs to the Conda-built executable and the installed FreeSurfer 8.2 executable. They establish program-level agreement, **not** equivalence of the inputs generated from T1.

| Program | Same-input result | Official / Conda wall time |
|---|---|---:|
| `mri_em_register` | LTA 4×4 matrix 16/16 values exact; text creation timestamp differs | 369.43 / 299.21 s |
| `mris_fix_topology` | Both hemispheres' ordered vertices and faces exact; file metadata differs | LH 43.2 / 53.3 s; RH 43.9 / 45.7 s |
| `mris_place_surface` metric modes | Bilateral thickness, area, pial area, white curvature, pial curvature: 1,188,540/1,188,540 values exact and 10/10 file hashes exact | 84.77 / 80.20 s for 10 commands |
| `mris_inflate` | LH ordered vertices, faces and sulc values exact | 12.21 / 12.19 s |
| `mris_sphere` | **Failed:** same-input LH vertex distance mean 2.435 mm, max 8.204 mm; official same-input repeat was exact | 376.62 / 338.37 s |
| `mris_register` | **Failed:** same-input LH `sphere.reg` vertex distance mean 0.1616 mm, max 2.8237 mm | 273.28 / 274.34 s |

Evidence: [`gca_source_vs_official_same_input.json`](gca_source_vs_official_same_input.json), [`topology_source_vs_official_same_input.json`](topology_source_vs_official_same_input.json), [`vertex_metric_source_vs_official_same_input.json`](vertex_metric_source_vs_official_same_input.json), [`SPHERE_REGISTRATION_SAME_INPUT.md`](SPHERE_REGISTRATION_SAME_INPUT.md), and [`CONDA_CPP_STAGES.md`](../../../../docs/recon_all/CONDA_CPP_STAGES.md). The paired timings are single observations on a shared node. Identical source commit and matching input bytes did not make the sphere and registration outputs identical. One compiler-dependent `abs(float)` span truncation was identified, but an isolated patch worsened the final sphere geometry; see [`SPHERE_ABS_INT_TRIAL.md`](SPHERE_ABS_INT_TRIAL.md). The remaining numerical causes are unresolved.

The Python `mri_ca_normalize` stage was also compared on frozen candidate inputs: `norm.mgz` and `ctrl_pts.mgz` match official data and headers exactly, with 35.47 s Python versus 91.44 s official in that pair. It remains in Python; the complete-subject `norm.mgz` differs because upstream inputs differ. See [`CA_NORMALIZE_CANDIDATE_SAME_INPUT.md`](CA_NORMALIZE_CANDIDATE_SAME_INPUT.md).

## Remaining numerical blockers

1. In this v1 baseline, corpus callosum assignment and final segmentation/ribbon postprocessing were incomplete after exact SynthSeg labels. The v2 runner now connects the validated Python corpus-callosum port: on the frozen candidate input it matched official `aseg.auto` and `aseg.presurf` at every voxel, but final `aseg` still differed at 104,986 voxels. See [the same-input CC report](MRI_CC_FEASIBILITY.md). `wm`, `filled`, and the current geometry also require a fresh v2 E2E comparison.
2. Native topology matches the official executable on *candidate* inputs, but the candidate mesh has different topology and vertex count from the official reconstruction. Directly swapping a C++ surface routine cannot fix that input mismatch.
3. The integrated `mris_place_surface` calls compute five vertex maps on the current white/pial meshes. Its white and pial geometry placement modes still lack `brain.finalsurfs`, autodetected gray/white stats, remeshed `orig`, auxiliary segmentations, and the official stage order. See [`WHITE_PIAL_DEPENDENCY_GAP.md`](WHITE_PIAL_DEPENDENCY_GAP.md).
4. Downstream atlas labels, ribbon-based volume assignment, per-vertex volume/area.mid and several regional statistics are absent or approximate. They must be checked after the upstream segmentation and mesh are fixed.

The GPU memory and FP32 output audit is in [`GPU_MEMORY_PROFILE.md`](GPU_MEMORY_PROFILE.md). No FP16 or BF16 mode was introduced. For production studies requiring FreeSurfer-equivalent cortical metrics, the current result fails the numerical acceptance gate.
