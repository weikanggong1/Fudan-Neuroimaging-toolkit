# `mri_edit_wm_with_aseg`: fixed option edits after the core

## Scope and source

The frozen `recon-all` command from `fs_sub01/scripts/recon-all.cmd` is:

```bash
mri_edit_wm_with_aseg -keep-in \
  -fix-ento-wm entowm.mgz 3 255 255 \
  -fix-acj aseg.presurf.mgz 255 255 \
  -fill-seg-wm -fix-scm-ha 1 \
  wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz
```

FreeSurfer 8.2.0 commit `d932c45b7941662ea380a05efef580568b98d41a`: [`mri_edit_wm_with_aseg/mri_edit_wm_with_aseg.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_edit_wm_with_aseg/mri_edit_wm_with_aseg.cpp) SHA-256 `6cddf001f37981e7f96d5f4658bd16eb0b06d51cf8d235cac46ff8d8f9da6f21`, [`utils/mri2.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mri2.cpp) SHA-256 `f6f01c07a2e44c127eb1b22bca61d0e6e87a6065f538fbafc8de343bcec84ed4` for `FixSubCortMassHA` and `MRIfixEntoWM`, and [`utils/mriset.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mriset.cpp) SHA-256 `90bf1b20ba0e4aad4a3827cba1976d87a63ec0eef07b8693ea18c319b3f6e96a` for the 3×3×3 dilation.

The command always calls `remove_paths_to_cortex`, `edit_segmentation`, and `spackle_wm_superior_to_mtl` first. `-fill-seg-wm` acts inside `edit_segmentation`. After that come SCM, keep-in, entorhinal WM, and ACJ edits. The new `edit_wm_aseg_late_python.py` handles these options **from an already produced core output**. Its `fill_seg_wm_from_core` is an equivalent rearrangement on this one fixed subject: seed all aseg cerebral WM voxels without adjacent cortex, then propagate to neighboring aseg WM below the 5-valued WM threshold. The source actually performs this inside `edit_segmentation`; this rearrangement has not been established for other subjects. SCM creates an aseg 0/2/3/41/42 mask, dilates one voxel in 3D, clears amygdala and inferior lateral ventricles, and clears hippocampus outside the dilated mask. Keep-in copies original 1/255 edited voxels. Entorhinal level 3 and ACJ use the existing independently tested `wm_edits_python.amygdala_cortex_junction` helper.

## Inputs and isolated references

Subject MRI path: `/data/workspaces/freesurfer_synth/work/reconall_main_20260924/single_subjects/fs_sub01/mri`. Isolated references: `/data/workspaces/freesurfer_synth/work/reconall_python_gpu_20260925/wm_asegedit_ref`.

| File | SHA-256 |
| --- | --- |
| `wm.seg.mgz` | `3cf79694f8e13889395e0e2a6e38508627ed1766bedecd751e76cece7884cd9f` |
| `brain.mgz` | `1b7360d069b76c8a5a63f93f3c296dddeb4db725c4e2625e11ebfc156279f401` |
| `aseg.presurf.mgz` | `263653b66a9aacba4d0d704781c6e67e581c86a9f99374e37d7e3f28d1674ce2` |
| `entowm.mgz` | `8bd292fc03e6a4a32ba9bedaa47f7e84e83e9594975df02f16f7fbd8cd9ca264` |
| Original `wm.asegedit.mgz` | `c1b0b0bfb9da0b240c3c07c82378db0411617989e7277614c11602616f7056c4` |
| Isolated native no-option `core.mgz` | `dd7b5426fd06662e61130d5598d72a1ba9aaae0e6f4e4f50c0e2a8ac0bdb3aca` |

The no-option native `core.mgz` differs from `wm.seg.mgz` at **66,173 / 16,777,216** voxels. The unconditional core is now independently matched for this frozen subject by [the Python core port](EDIT_WM_ASEG_CORE.md). Running `-fix-scm-ha-only aseg.presurf.mgz core.mgz 1 core_scm.mgz` gives the standalone SCM reference. `core_opts.mgz` is a native run with all options except `-fill-seg-wm`; `full.mgz` is a fresh run with the exact command above. The fresh full result differs from the original recon-all `wm.asegedit.mgz` at **0 / 16,777,216** voxels. The full result differs from `core_opts.mgz` at **172,522** voxels, all changed to 250. The 3×3×3 WM propagation accounts for 13,702 changed voxels outside the direct no-cortex seeds; seed writes and subsequent option interactions account for the remainder.

## Fixed-subject gates

| Comparison | Mismatched voxels |
| --- | ---: |
| Python SCM on native `core.mgz` vs native `core_scm.mgz` | **0 / 16,777,216** |
| Python late edits without fill vs native `core_opts.mgz` | **0 / 16,777,216** |
| Python late edits with fill vs native `full.mgz` | **0 / 16,777,216** |
| Python late edits with fill vs original recon-all output | **0 / 16,777,216** |

The Python late output has Fortran-order voxel SHA-256 `6a1312de6ff32e810f983a131dded18cdea6a11f845adb5077efd51cac74da2e`, identical to both full native voxel arrays. Decompressed MGH 284-byte header, all voxel bytes, and 20-byte scan parameters are identical to fresh native. Footer and gzip bytes differ because the Python writer retains the no-option core's command history, whereas full native records its complete command. Three synthetic tests passed.

On control-host, fresh native no-option core took 18.29 s and the full native command 28.29 s. Python options from the already existing native core took 4.48 s with fill, including file I/O. These are different scopes and do not establish a complete replacement speedup. This stage runs on CPU using NumPy, SciPy and nibabel; its Python code never executes or reads a FreeSurfer runtime binary.

Reproduce this **substage** using the frozen native core reference:

```bash
BASE=/data/workspaces/freesurfer_synth/work
MRI=$BASE/reconall_main_20260924/single_subjects/fs_sub01/mri
REF=$BASE/reconall_python_gpu_20260925/wm_asegedit_ref
PYTHONPATH=src python -m fnit.recon_all.edit_wm_aseg_late_python \
  "$REF/core.mgz" "$MRI/aseg.presurf.mgz" "$MRI/entowm.mgz" \
  "$MRI/wm.seg.mgz" "$REF/full_late_python.mgz" --fill-seg-wm
PYTHONPATH=src python validation/recon_all/python_gpu_port/experimental/check_edit_wm_aseg_late.py \
  "$MRI" "$REF"
```

This late-edit module alone starts from an existing core output. The [combined fixed-subject Python port](EDIT_WM_ASEG_CORE.md) now matches the complete native call at every voxel without a native intermediate. It remains gated to the four frozen input hashes; other subjects and integration into the main recon-all entry point are not accepted.
