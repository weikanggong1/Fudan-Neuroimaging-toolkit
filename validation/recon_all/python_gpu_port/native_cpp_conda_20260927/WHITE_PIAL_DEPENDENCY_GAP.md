# White/pial geometry placement: integration dependency audit

This is a read-only audit of the fixed `sub-01_T1w.nii.gz` case. The current
runner calls the Conda-built `mris_place_surface` for bilateral thickness,
area, pial area, white curvature and pial curvature **on its approximate
surfaces**. It does not call the C++ white/pial geometry optimizer. The ten
metric calls passed same-input comparisons against the installed FreeSurfer
binary at every vertex and by complete file SHA-256; this certifies those
metric commands on frozen surfaces, not the reconstructed surfaces themselves.

## Inputs and command order

The source of the official commands is the unchanged FreeSurfer 8.2 log at
`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official/scripts/recon-all.log`.
Lines 3369/3771 place `lh/rh.white.preaparc` from `orig` with `--white
--nsmooth 5`; lines 5530/5849 place final `white` from `white.preaparc` with
`--white --nsmooth 0 --rip-label cortex.label --aparc aparc.annot`; lines
6168/6483 place `pial.T1` from `white` with `--pial --rip-label
cortex+hipamyg.label --pin-medial-wall cortex.label --repulse-surf white
--white-surf white`. All six commands use `brain.finalsurfs.mgz`, `wm.mgz`,
`aseg.presurf.mgz` and the matching `autodet.gw.stats.{hemi}.dat`.
The logged left calls took 242.65, 225.85 and 231.41 seconds; these are
historical shared-node wall times, not a current paired benchmark.

| Required upstream item | Current candidate | Reuse and gap |
| --- | --- | --- |
| `brain.finalsurfs.mgz` | Absent | `mri_mask_gpu.mask_volume` already implements the three fixed mask calls, but this candidate lacks `mca-dura.mgz` and `vsinus.mgz`; final EntoWM and ACJ edits also need `entowm.mgz`. The historical fixed-input mask/edit tests do not establish a connected output here. |
| `autodet.gw.stats.{hemi}.dat` | Absent | `autodet_gwstats_python.write_autodet_stats` matched the official frozen case in all 40 fields and complete bytes. Its official inputs are finalsurfs, WM and `orig.premesh`, so it cannot currently be run on the same inputs. |
| `orig.premesh` then remeshed `orig` | `orig.premesh` absent; `orig` present | The current `mris_fix_topology` call writes directly to `orig`. Official lines 2884–3104 write `orig.premesh`, then lines 3203/3222 run `mris_remesh --remesh --iters 3` to create `orig`. The existing `mris_remesh_python` matched ordered frozen-input geometry, but has not been connected to this candidate. |
| `cortex.label` and `cortex+hipamyg.label` | First present, second absent | `label_cortex_python` can make the latter from `white.preaparc` and `aseg.presurf.mgz`; `label_cortex_fix_ga_python` can make the former with the EntoWM gyrus-ambiens correction. Both were validated on frozen official inputs. The runner currently makes `cortex.label` from its later `white` surface, so orchestration must move before final white/pial placement. |
| `aparc.annot` and registered sphere | Present only after the candidate's white/pial generation | Official final white/pial placement reads the annotation. The runner must generate sphere, register it and annotate after `white.preaparc` but before final `white`. Its current late annotation order cannot supply the optimizer's official inputs. |
| MCA/dura, venous sinus, EntoWM models and priors | Not in this run's six weights and 15 data assets | `aux_seg.py` and `sclimbic.py` provide Python inference wrappers, but the external model/prior inputs and the separate SynthMorph prior voxel LTA are not in this candidate. Their connected outputs require validation before use as official-equivalent inputs. |

The candidate inspected here is
`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_cpp_conda_20260927/sphere_registration_preflight/subjects/sub01`.
The asset/weight inventory is from the adjacent `assets/` and `weights/`
directories; the source-built binary is in the adjacent `fs_cpp/bin/`.

## Accuracy gate and next implementation slice

The candidate's source-built topology run has 117,777 left and 119,990 right
vertices; the unchanged official subject has 106,622 left and 105,541 right.
The candidate counts are recorded in
[`topology_source_vs_official_same_input.json`](topology_source_vs_official_same_input.json),
which compares the source-built and installed binaries **on the same candidate
input**, rather than comparing candidate topology with official recon-all.
The official counts are present in the frozen
[`strict_138.json`](../native_free_sub01_20260927/strict_138.json)
reference geometry. `mris_place_surface` changes positions while preserving
mesh vertex/face count, so substituting it alone cannot make ordered official
white/pial vertices, per-vertex measures, or ROI statistics agree.

The smallest meaningful integration slice is: reproduce the official WM,
aseg, finalsurfs and auxiliary segmentations; write topofix output as
`orig.premesh`; remesh to `orig`; calculate bilateral autodet thresholds;
place `white.preaparc`; generate both cortex labels; generate/register sphere
and annotate; place `white` and `pial.T1`; then recompute pial cleanup,
vertex maps and ROI statistics. Compare each stage with the installed binary
on **identical upstream files** before testing end-to-end parity. No white or
pial placement experiment was run during this audit because the connected
inputs above are missing and the full candidate run was occupying gpucw1.
