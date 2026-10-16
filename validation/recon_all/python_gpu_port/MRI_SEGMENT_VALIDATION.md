# `mri_segment` fixed-T1 port, 2026-09-25

The complete fixed-profile Python implementation is in `src/fnit/recon_all/mri_segment.py`. `segment_white_matter(image)` returns the WM volume; `segment_white_matter_mgz(source_path, output_path)` reads and writes MGZ while preserving the input MGH header. It uses PyTorch for the trinary intensity labels, NumPy/SciPy for most later stages, and no FreeSurfer runtime.

Reference command: `mri_segment -wsizemm 13 -mprage antsdn.brain.mgz wm.seg.mgz` from FreeSurfer 8.2.0-1. The native `-diag-write -diag-verbose` run used the frozen `fs_sub01` input and wrote diagnostic images under `work/reconall_python_gpu_20260925/mri_segment_diag` on control-host. Its final `wm.seg.mgz` has 0 differing voxels versus the saved official reconstruction (both 256³ uint8); its 284-byte MGH header and affine also match. The diagnostic run reported 2.5 minutes, including diagnostic image writes.

| Stage | Native diagnostic image | Checked voxels | Mismatches | Python CPU seconds on control-host |
| --- | --- | ---: | ---: | ---: |
| First trinary intensity labels | `wmseg.int.1.mgz` | 16,777,216 | 0 | 0.04 |
| First local histogram pass | `wmseg.histo.1.mgz` | 187,772 ambiguous | 0 | 29.53 |
| Second trinary intensity labels | `wmseg.int.2.mgz` | 16,777,216 | 0 | 0.02 |
| Second local histogram pass | `wmseg.histo.2.mgz` | 87,457 ambiguous | 0 | 14.01 |
| Median-curve geometry | `wmseg.medcurv.mgz` | 86,646 ambiguous | 0 | 10.38 |
| Border Gaussian reclassification | `wmseg.reclassified.mgz` | 135,195 candidates | 0 | 3.67 |
| Bright white recovery | `wmseg.recover-brightwm.mgz` | 16,777,216 | 0; 7 restored | 0.66 |
| Wrong-direction removal | `wmseg.wrong-dir.mgz` | 16,777,216 | 0; 13,700 removed | 0.67 |
| One-dimensional removal | `wmseg.rm1d.mgz` | 16,777,216 | 0; 3,250 removed | 0.31 |
| Thin-strand thickening, including planar holes | `wmseg.thicken.mgz` | 16,777,216 | 0; 4,756 added | 35.21 |
| Bright non-WM mask | `wmseg.bright-nonwm.mgz` | 16,777,216 | 0; 87 removed | 2.36 |
| Diagonal morphology | `wmseg.filter.mgz` | 16,777,216 | 0; 897 added | 0.46 after JIT cache |

Each row after the first takes the corresponding saved native preceding stage as input. The diagonal-morphology row is the later Numba implementation; its earlier interpreted version took 33.48 seconds and gave the same output. The complete chained Python run from `antsdn.brain.mgz` to an in-memory WM volume also had 0/16,777,216 voxel mismatches and took 117.33 seconds before that optimization. The earlier file-output wrapper wrote `mri_segment_python_wm.seg.mgz` in 113.59 seconds; after the optimization a new file-output run took 90.40 seconds and again matched all 16,777,216 native voxels and the 284-byte MGH header. The decompressed voxel bytes, affine, shape, and uint8 type match the native diagnostic file exactly. The Python footer is 20 bytes and matches the first 20 bytes of both native files' footers. The diagnostic footer has 2,735 bytes and the frozen official footer 2,493 bytes, reflecting additional native tags/history; compressed-file hashes differ. Most measured work remains on CPU. Times vary between runs; the isolated stage times exclude image I/O.

## Thickening parity and scope

The candidate finder reproduces native largest segment 845 (1,703 voxels); five dilation passes produce 3,643 voxels, also exactly matching the native diagnostic image. Direct thickening and neighborhood filling add 4,545 voxels, and planar hole filling adds the remaining 211. The final labels match native: 4,363 `THICKEN_FILL=200` and 393 `NBHD_FILL=210`.

This evidence covers one 1 mm MP-RAGE T1 and the fixed `-wsizemm 13` call. It does not establish parity for other acquisition profiles or a GPU speedup. Reproduce the complete file comparison on control-host with `PYTHONPATH=src python3 validation/recon_all/python_gpu_port/test_mri_segment_intensity.py <antsdn.brain.mgz> <mri_segment_diag_dir> --full-output <output.mgz>`. Use `--histogram-limit -1 --curve-limit -1 --reclassify-limit -1` to rerun the expensive isolated stages. No full recon-all rerun is needed.

On the same control-host input, one native run without diagnostic writes took 45.93 wall seconds (`/usr/bin/time`, max RSS 151,828 KiB) and reproduced the frozen official voxels/header. The latest Python file-output run took 90.40 seconds, or about 1.97 times that native wall time. The 0.46-second diagonal check can be rerun with `validate_mri_segment_diagonal.py <wmseg.bright-nonwm.mgz> <wmseg.filter.mgz>` and fails on any voxel mismatch. These are single, separately timed runs; they establish the current speed boundary, not a stable throughput estimate.
