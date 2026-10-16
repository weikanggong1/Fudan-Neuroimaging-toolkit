# `mri_relabel_hypointensities` isolated replacement

The Python stage reads `aseg.presurf.mgz` and the bilateral `white` surfaces,
then writes `aseg.presurf.hypos.mgz`. It follows the pinned FreeSurfer 8.2
source: merge labels 78/79 into 77; classify cortex voxels inside each white
surface using the nearest vertex, surface normal and a 1 mm distance cutoff;
then make two six-neighbor gray-matter recovery passes with left-cortex
precedence. PyTorch computes surface normals, SciPy finds nearest vertices,
and NumPy/SciPy perform the voxel edits. This short stage runs on CPU.

On `fs_sub01`, the Python pass classified 634 left and 330 right cortex voxels
as hypointense, matching fresh native logs. Its two recovery passes changed
963 voxels; the final result differs from the input in one voxel. The output
matched both a fresh official command and the archived official result for
**all 16,777,216 voxels and the complete decompressed MGH content**. The
same-input control-host CLI run took **5.15 s Python versus 11.14 s native**. Both
times include process startup and file I/O; only one pair was collected, so
this is a measured example rather than a general speed claim. One synthetic
two-pass recovery test passed.

```bash
python -m fnit.recon_all.relabel_hypointensities_python \
  /path/to/aseg.presurf.mgz /path/to/surf \
  /path/to/aseg.presurf.hypos.mgz
```

The Python nearest-vertex search uses a global `cKDTree`, whereas the native
program uses a spatial hash and can fall back to a signed-distance volume
when no nearby vertex is found. The frozen subject matched exactly, but
additional T1 subjects must be checked before claiming general parity. This
stage still needs integration with Python implementations of its upstream
surface and segmentation stages.
