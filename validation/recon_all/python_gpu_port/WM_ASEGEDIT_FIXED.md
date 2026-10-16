# Fixed combined `mri_edit_wm_with_aseg` call

The frozen FreeSurfer 8.2 `fs_sub01` run calls `mri_edit_wm_with_aseg
-keep-in -fix-ento-wm entowm.mgz 3 255 255 -fix-acj aseg.presurf.mgz
255 255 -fill-seg-wm -fix-scm-ha 1 wm.seg.mgz brain.mgz aseg.presurf.mgz
wm.asegedit.mgz`. `edit_wm_aseg_core_python.py` and
`edit_wm_aseg_late_python.py` replay this fixed call in NumPy, SciPy and
Numba on CPU. They invoke no FreeSurfer program.

On control-host, the Python output, a fresh native rerun, and the archived native
`wm.asegedit.mgz` had **zero mismatches across all 16,777,216 voxels**. The
284-byte MGH header also matched. The first decompressed byte difference
between Python and fresh native was at offset 16,777,679, after the voxel
payload ending at offset 16,777,500; the following command-history tags differ.
This is a voxel/header result, not complete file byte identity.

One matched-input, fresh-process CLI pair on control-host took 27.49 s native and
8.45 s Python. These are single observations with file I/O and Python startup
included, so they do not establish an end-to-end speed ratio. The Python
implementation is CPU only, and its core has an explicit input-hash gate:
other T1 subjects are not yet accepted. In particular, the path-removal and
`-fill-seg-wm` branches need general-subject validation before removing this
gate.

Input SHA-256 values (on-disk MGZ) were:

| Input | SHA-256 |
| --- | --- |
| `wm.seg.mgz` | `3cf79694f8e13889395e0e2a6e38508627ed1766bedecd751e76cece7884cd9f` |
| `brain.mgz` | `1b7360d069b76c8a5a63f93f3c296dddeb4db725c4e2625e11ebfc156279f401` |
| `aseg.presurf.mgz` | `263653b66a9aacba4d0d704781c6e67e581c86a9f99374e37d7e3f28d1674ce2` |
| `entowm.mgz` | `8bd292fc03e6a4a32ba9bedaa47f7e84e83e9594975df02f16f7fbd8cd8ca264` |

The candidate, fresh native, and archived output MGZ SHA-256 values were
`ea4bcc1d69fa606a22738ea65205192b72106f8a4c29baa11307db9f8d9aee8b`,
`21d6213d15ed0aaffa2d6ed1b77aaa70e4bc514682eac13e42e475d1f25b2a96`,
and `c1b0b0bfb9da0b240c3c07c82378db0411617989e7277614c11602616f7056c4`,
respectively. Different whole-file hashes are expected from the differing
history/footer data.
