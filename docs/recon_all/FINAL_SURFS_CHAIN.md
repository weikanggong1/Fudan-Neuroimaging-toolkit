# `brain.finalsurfs.mgz`: connected Python mask and edit chain

`fnit.recon_all.finalsurfs_python.run_finalsurfs(subject_dir, device="cpu")`
reads six conformed MGZ files from `subject_dir/mri`: `brain.mgz`,
`brainmask.mgz`, `mca-dura.mgz`, `vsinus.mgz`, `entowm.mgz`, and
`aseg.presurf.mgz`. It writes `brain.finalsurfs.mgz` and its identical
pre-manual-edit checkpoint `brain.finalsurfs.manedit.mgz` in the same directory.
The Python return value is the final MGZ path. All inputs must have the same
voxel grid. The masks run through PyTorch on the selected CPU or CUDA device;
the two final edits use NumPy/SciPy on CPU. This function requires no
FreeSurfer executable.

Python API:

```python
from fnit.recon_all.finalsurfs_python import run_finalsurfs
final_mgz = run_finalsurfs("/path/to/subjects/sub01", device="cpu")
```

Command line:

```bash
python -m fnit.recon_all.finalsurfs_python /path/to/subjects/sub01 --device cpu
```

The FreeSurfer 8.2 equivalent, run from `subject_dir/mri`, is:

```bash
mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz
mri_mask -oval 1 -invert brain.finalsurfs.mgz mca-dura.mgz brain.finalsurfs.mgz
mri_mask -oval 1 -invert brain.finalsurfs.mgz vsinus.mgz brain.finalsurfs.mgz
mri_edit_wm_with_aseg -sa-fix-ento-wm entowm.mgz 2 255 255 brain.finalsurfs.mgz brain.finalsurfs.mgz
mri_edit_wm_with_aseg -sa-fix-acj aseg.presurf.mgz 255 255 brain.finalsurfs.mgz brain.finalsurfs.mgz
cp brain.finalsurfs.mgz brain.finalsurfs.manedit.mgz
```

## Real T1 comparison

The [saved report](../../validation/recon_all/python_gpu_port/finalsurfs_chain_20260927/report.json)
uses the deidentified `sub-01_T1w.nii.gz` scan, SHA-256
`f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`,
and its completed FreeSurfer 8.2 reconstruction on `gpucw1`. Both full
five-step replays read the **same six saved official input volumes**, with
their input hashes recorded in the report. Python produced **0 differing
voxels out of 16,777,216** for the final volume and checkpoint. It took
8.83 seconds including Python function I/O; the five official commands took
8.21 seconds including process startup and I/O in one unpaired run on the
same host. This does not establish a speed difference.

The current FNIT v5 T1-prefix output already matches the official `brain`,
`brainmask`, `entowm`, `aseg.presurf` and `nu` volumes voxelwise on this scan.
It has no `mca-dura` or `vsinus` volumes. A second replay used those **four
FNIT-generated inputs**, borrowed only the **two official auxiliary label
volumes**, and again produced 0/16,777,216 differing final voxels in 9.13
seconds. This is a conditional upstream test, not an end-to-end FNIT result.

The [byte analysis](../../validation/recon_all/python_gpu_port/finalsurfs_chain_20260927/byte_analysis.json)
found identical decompressed MGH headers and voxel payloads. The Python file
is one byte shorter in a trailing provenance tag: the native replay and the
saved official output have an extra null byte after `UNKNOWN`. This footer
does not enter voxel or surface computations.

The opt-in `--native-white-preaparc` runner now invokes this function after
generating the subject-specific MNI152 LTA and both auxiliary segmentations.
Its default path does not. The connected prefix and later cortical measures
still need separate end-to-end validation; the paired frozen-input result
above does not establish whole-subject morphometry parity.
