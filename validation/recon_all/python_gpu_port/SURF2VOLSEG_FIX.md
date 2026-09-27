# `mri_surf2volseg --fix-presurf-with-ribbon` isolated replacement

This Python/NumPy/SciPy branch reads `aseg.presurf.hypos.mgz`, `ribbon.mgz`,
bilateral white/pial surfaces, and bilateral `cortex.label` masks; it writes
`aseg.mgz`. The voxel rules follow pinned FreeSurfer 8.2
`mri_aparc2aseg/mri_surf2volseg.cpp`, including a nearest-vertex cortex-mask
check for cerebral white matter outside the ribbon and unknown voxels inside
it. `surf2volseg_fix_python.py` is the fixed `--fix-presurf-with-ribbon` branch
only in this historical check; `--label-cortex` and `--label-wm` were later connected and checked on real frozen inputs in [the current runner report](ATLAS_VOLUME_RUNNER_20260928.md).

On frozen `fs_sub01` inputs, **16,777,216/16,777,216 output voxels** matched
both the archived and freshly run official `aseg.mgz`. The first categorical
rules left 751 voxel mismatches; all 751 were resolved by the nearest cortex
vertex check among 1,278 candidate voxels. The Python output MGH header and
voxel payload bytes match the fresh official output. The trailing MGH color
table/history differs, so the complete decompressed file is not byte
identical. A focused synthetic rule test passed.

One same-input headcw CLI pair took **3.28 s native and 4.69 s Python**. Both
times include process startup and I/O; this short stage is currently slower.

```bash
python -m fnit.recon_all.surf2volseg_fix_python \
  /path/to/aseg.presurf.hypos.mgz /path/to/ribbon.mgz \
  /path/to/surf /path/to/label /path/to/aseg.python.mgz
```

The Python nearest-vertex search is global; the native program uses a spatial
hash. Exact parity is established for the tested T1 and fixed inputs, while
other subjects need independent voxel checks.
