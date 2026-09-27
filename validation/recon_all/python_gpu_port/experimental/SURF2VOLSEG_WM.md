# `mri_surf2volseg --label-wm`: fs_sub01

## Source and fixed command

The recon-all WM parcellation command is `mri_surf2volseg --label-wm --i aparc+aseg.mgz --lh-annot lh.aparc.annot 3000 --rh-annot rh.aparc.annot 4000`, with both cortex masks and both white/pial surfaces supplied. Source is FreeSurfer 8.2.0 commit `d932c45b7941662ea380a05efef580568b98d41a`, [`mri_aparc2aseg/mri_surf2volseg.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_aparc2aseg/mri_surf2volseg.cpp), SHA-256 `33d7e17e7ede3ea4136ea4c26b7de4740b407cc43e724af5e02229baa989275f`.

The source only relabels cerebral WM (2/41) and WM hypointensity IDs. `--label-wm` restricts nearest white-surface vertices to `?h.cortex.label`, applies the inward normal check, assigns the annotation index plus 3000/4000 within 5 mm, and otherwise uses 5001/5002. The fixed `aparc+aseg.mgz` contains 180,858 left WM and 179,377 right WM voxels, and no hypo voxels. Every cortex-mask vertex has a positive aparc annotation on this subject. The Python stage therefore reuses the already validated `_nearest_with_dot` white-surface query from `surf2volseg_cortex_python.py`; it need not load pial surfaces for this branch.

`src/fnit/recon_all/surf2volseg_wm_python.py` provides `label_wm_voxels`, `label_wm_volume`, and a standalone step CLI. It uses Python, NumPy, SciPy, PyTorch, and nibabel, without calling a FreeSurfer binary or reading its runtime package. The source MGZ header/footer writer is the existing `mgh_compat.save_same_dtype_mgh`. This was an earlier standalone-stage status. The current runner now calls this module after generating aparc+aseg; see [the current real-input runner check](../ATLAS_VOLUME_RUNNER_20260928.md).

## Frozen inputs

All paths below are under `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_main_20260924/single_subjects/fs_sub01`; the isolated native and Python results are in `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_python_gpu_20260925/wmparc_ref`.

| Relative input path | SHA-256 |
| --- | --- |
| `mri/aparc+aseg.mgz` | `718995e25b825a50f0a5dae95c5f02b36a53369cd1086a4604f58a64f8405770` |
| `surf/lh.white` | `9c88a786c4a571c07d830b83fe647461b09fdacb63ff143670771340ef50b51c` |
| `surf/rh.white` | `5c7fda4367dd3624de0ef265303dab49579aa603aee75f1cce2f9c30bec965b9` |
| `label/lh.cortex.label` | `524245d2c800a7ff3fc8ba1d1fe04c02d09354bcc29e99cf3f64182c2f312506` |
| `label/rh.cortex.label` | `609cce84a6a0488c3333af213639ab55bc0312a95485baddf5f4d58e82fd6630` |
| `label/lh.aparc.annot` | `9b4b6dce47037a623ca4bbab6564cd3bd31369c22c2e7c55d305a85c46517536` |
| `label/rh.aparc.annot` | `aafe54c8b3ae84fb90ce38f766637ee2b80f0a2549cfde2d80698eacc89646de` |

The official native command also reads `surf/lh.pial` (`f7c61a103d48497d38fb46ff0495984624eba726f384473f5634a8d7102d66d5`) and `surf/rh.pial` (`0e0ac9a32a6e3d8087d63be2f3a68a237bef6331e1e867d33531611dd11e1eac`), although its `LabelWM` branch permits white surfaces only.

## Result and timing

A fresh headcw native rerun matched the existing official `mri/wmparc.mgz` at **0 / 16,777,216** voxels. The Python result also matched that original official output at **0 / 16,777,216** voxels. Its Fortran-order voxel SHA-256 is `5f00cb7ff68868ea47f21c7286dd5473dca1264c1f4e35ce4ded05977578d4a5`. More strongly, the **entire decompressed MGH payload**, including geometry and footer metadata, is byte-identical to the original official output. Gzip container bytes differ: Python file SHA-256 `3021e5b6dcdc2804f32ebf03706100eee7c84de1ad182e97f1e5db7d201e9354`, original official file SHA-256 `fc764c8396989984c550319240a96a06134e4a7708579a8dfae1321a9e77da2c`.

On the same headcw host and fixed input, the complete Python step took **5.70 s**, versus **9.69 s** for the fresh native rerun (`--threads 4`). Both include input and output. This is one paired run, not a multi-subject speed estimate. The Python work ran on CPU; its PyTorch normal computation also used CPU. The test `test_surf2volseg_wm_python.py` checks lateralization, the 5 mm cutoff, future WMSA unknown labels, and preservation of unrelated labels. The fixed subject does not test native nearest-annotation recovery for unannotated cortex vertices or the real-data hypo labels.

Reproduce the Python stage and compare against both native references:

```bash
BASE=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work
SUBJ=$BASE/reconall_main_20260924/single_subjects/fs_sub01
REF=$BASE/reconall_python_gpu_20260925/wmparc_ref
PYTHONPATH=src python -m fnit.recon_all.surf2volseg_wm_python \
  "$SUBJ/mri/aparc+aseg.mgz" "$SUBJ/surf" "$SUBJ/label" "$REF/wmparc_python.mgz"
python validation/recon_all/python_gpu_port/experimental/check_surf2volseg_wm.py \
  "$SUBJ/mri/aparc+aseg.mgz" "$REF/wmparc_python.mgz" \
  "$SUBJ/mri/wmparc.mgz" "$REF/wmparc_native.mgz"
```
