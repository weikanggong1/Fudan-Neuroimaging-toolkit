# NIfTI T1 import: isolated validation

The first single-T1 recon-all command is `mri_convert sub-01_T1w.nii.gz
mri/orig/001.mgz`. `nifti_import.import_t1` replaces this command using
nibabel. This stage only changes image containers; it does not resample or
process voxels, so moving the operation to GPU would add transfer overhead.

On `control-host`, the same source image was converted with FreeSurfer 8.2.0-1 and
the Python implementation. Against the completed subject's `rawavg.mgz`
(an exact copy of `mri/orig/001.mgz`):

| Check | Result |
| --- | --- |
| 256 × 156 × 256 float32 voxels | 10,223,616/10,223,616 exact; payload bytes identical |
| First 284 MGH header bytes | Identical |
| 20 scan parameter bytes | Identical |
| Voxel-to-RAS affine | Exact |
| Remaining FreeSurfer tags and command history | Not reproduced |

The reference file has 631 footer bytes in total; the Python output has the
20 scan-parameter bytes only. New native benchmark outputs have 686 footer
bytes because their command-line paths differ. The omitted data is provenance,
not image or scan geometry. It remains an output-contract difference if full
MGH decompressed byte identity is required.

Three paired fresh-process CLI trials on `control-host`, including startup, input
read and MGZ write without clearing the OS cache: FreeSurfer median
**1.925 s**, Python median **0.594 s**. Every
trial matched the header, voxels and scan parameters. See
[`nifti_import_control-host_report.json`](nifti_import_control-host_report.json) for
per-trial measurements and [`benchmark_nifti_import.py`](benchmark_nifti_import.py)
for the method. Two focused unit tests passed on the same host.

This covers the tested 3D float32 NIfTI-1 T1 path with sform, millimeter
units and unscaled voxels. Other `mri_convert` modes are outside this stage.
