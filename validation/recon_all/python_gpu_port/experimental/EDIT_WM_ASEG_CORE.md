# Fixed `mri_edit_wm_with_aseg` Python port: fs_sub01

## Source and scope

This port follows FreeSurfer 8.2.0 [`mri_edit_wm_with_aseg.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_edit_wm_with_aseg/mri_edit_wm_with_aseg.cpp), pinned commit `d932c45b7941662ea380a05efef580568b98d41a`, source SHA-256 `6cddf001f37981e7f96d5f4658bd16eb0b06d51cf8d235cac46ff8d8f9da6f21`. The validation binary SHA-256 is `7a2e829de0ccfeba478a0acfa70e27340214803e7547197aa8af689d5ada7a68`. `edit_wm_aseg_core_python.py` translates the active `edit_segmentation` scans and final `spackle_wm_superior_to_mtl` into NumPy/Numba. The earlier `edit_wm_aseg_late_python.py` supplies the fixed `-keep-in -fix-ento-wm ... -fix-acj ... -fill-seg-wm -fix-scm-ha 1` options.

The complete callable and CLI are deliberately **gated to the four frozen fs_sub01 voxel hashes**. Other subjects raise `NotImplementedError`. Several source branches have labels absent from this subject, and the `-fill-seg-wm` rearrangement has only been proved for this subject. This is a complete replacement of **this one fixed command/input**, not yet a generalized recon-all stage. The production call reads no native FreeSurfer executable or native intermediate output. It runs on CPU; this short stage is not yet a GPU kernel.

## Path function and int32 byte access

The native no-option run reports `0 voxels added to wm to prevent paths from MTL structures to cortex`. The pinned source modifies WM in `remove_paths_to_cortex` only when it increments this count. The source clones the **int32** `aseg.presurf.mgz` into its ROI and seed volumes, then accesses those clones with [`MRIvox`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/include/mri.h#L1480), a `BUFTYPE` (uint8) macro. Byte index `x` therefore addresses int32 voxel `x//4`, not int32 voxel `x`.

An isolated native run with `DIAG=0x8` wrote `lh_roi.mgz`, `rh_roi.mgz`, and `f1.mgz`. Both ROI volumes match Python's initial five 3×3×3 dilations at **0 / 16,777,216** voxels; the native trimming writes bytes outside their active support. Their minimum x coordinates are **134** and **90**. Since byte indexing with width 256 can address only int32 voxel x ≤63, the native path scans cannot access active ROI/seed locations for this subject. The Python `remove_paths_to_cortex` verifies this condition from the input and returns the unchanged WM. It explicitly rejects any input where an MTL ROI reaches x ≤63. A first, conventional int32-voxel translation erroneously filled 1113 voxels and was discarded after comparing these native diagnostic outputs.

## Frozen reference and voxel gates

Input files are in `/data/workspaces/freesurfer_synth/work/reconall_main_20260924/single_subjects/fs_sub01/mri`; independent native references and GDB diagnostic raw arrays are in `/data/workspaces/freesurfer_synth/work/reconall_python_gpu_20260925/wm_asegedit_ref`. The validator checks input MGZ SHA-256 hashes: `wm.seg.mgz` `3cf79694f8e13889395e0e2a6e38508627ed1766bedecd751e76cece7884cd9f`, `brain.mgz` `1b7360d069b76c8a5a63f93f3c296dddeb4db725c4e2625e11ebfc156279f401`, `aseg.presurf.mgz` `263653b66a9aacba4d0d704781c6e67e581c86a9f99374e37d7e3f28d1674ce2`, and `entowm.mgz` `8bd292fc03e6a4a32ba9bedaa47f7e84e83e9594975df02f16f7fbd8cd9ca264`.

| Stage | Python vs native differing voxels | Fortran-order voxel SHA-256 |
| --- | ---: | --- |
| After MTL path check | 0 / 16,777,216 | `591b231eca77bad3c3d1a4cedddb910a0db5f5f572732e30b54b8f6ad6d9fae6` |
| First edit scans through WM propagation | 0 / 16,777,216 | `d00b89316a242fa736e4ff9a04529ecbffbea4f917c5cdab1cce59744246b69b` |
| Full `edit_segmentation`, before final spackle | 0 / 16,777,216 | `449dccb41c60771a71cf023ee0076829c274c62bd7e48490525aee25ce8728da` |
| Native no-option core | 0 / 16,777,216 | `b97c44648cb2c5f494495cfd9201a1ba149cd5f38cb7041c354d9328a46e810b` |
| All fixed options vs fresh native full and original recon-all output | 0 / 16,777,216 | `6a1312de6ff32e810f983a131dded18cdea6a11f845adb5077efd51cac74da2e` |

The first edit scans change **65,951** voxels from input. Later active MTL rules add 172; the final `spackle_wm_superior_to_mtl` adds 50. Native `core.log` reports **55,088** WM turn-ons and **24,075** turn-offs; source counters include repeated writes, while final no-option core differs from input at **66,173** positions. The fresh Python `.mgz` has the same decompressed 284-byte header, all 16,777,216 voxel bytes, and 20 scan-parameter bytes as fresh native full output. Footer command histories differ (2473 vs 3210 bytes) because the writer preserves input tags and does not forge a native command line.

## Timing and reproduction

On the same control-host CPU and NFS input, one sequential CLI pair took **12.21 s** for Python and **35.00 s** for the exact native full command. This is one run per command, not a distribution or a GPU benchmark. A warm Python validator timed path proof 0.99 s, first edit scans 1.06 s, early/late MTL scans 0.23/0.11 s, below-hippocampus fill 0.08 s, final spackle 0.18 s, and fixed options 3.02 s; the validator also loads and compares several native volumes, so its total wall time is a different scope. Four synthetic tests passed.

```bash
BASE=/data/workspaces/freesurfer_synth/work
MRI=$BASE/reconall_main_20260924/single_subjects/fs_sub01/mri
REF=$BASE/reconall_python_gpu_20260925/wm_asegedit_ref
PYTHONPATH=src python -m fnit.recon_all.edit_wm_aseg_core_python \
  "$MRI/wm.seg.mgz" "$MRI/brain.mgz" "$MRI/aseg.presurf.mgz" \
  "$MRI/entowm.mgz" "$REF/full_python_no_native.mgz"
PYTHONPATH=src python validation/recon_all/python_gpu_port/experimental/check_edit_wm_aseg_core.py \
  "$MRI" "$REF"
pytest -q tests/recon_all/test_edit_wm_aseg_core_python.py
```

Native diagnostic references were generated in isolation using `DIAG=0x8`, `FREESURFER_HOME` and `FS_LICENSE` environment variables, and the pinned native executable solely for validation. The Python CLI above has no FreeSurfer runtime dependency. Generalizing the path search requires emulating the native int32/uint8 byte aliasing for ROIs that reach x ≤63, plus held-out subjects for the branches absent in fs_sub01.
