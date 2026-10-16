# Second `mri_normalize` pass with aseg and brain mask

The isolated Python/Numba/PyTorch CPU command reproduces the FreeSurfer 8.2.0 (`d932c45b7941662ea380a05efef580568b98d41a`) call

```text
mri_normalize -seed 1234 -mprage -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz
```

on the frozen `fs_sub01` 256³ T1 inputs. The Python command uses those three volumes only; no native executable, native intermediate, license, or template is read during inference. The fixed inputs share a voxel grid. Other subject geometries and the CUDA path remain unvalidated. This stage is not yet connected to `fnit-recon-all`.

## Complete output gate

| Check | Independent Python versus fresh native |
| --- | ---: |
| Aseg WM seed controls | 0 mismatches |
| Medial ridge | 0/16,777,216 voxel mismatches; 45,325 ridge voxels |
| Filtered WM controls and outlier map | 0 mismatches each; 8,352 removed |
| Float32 initial `norm_1` checkpoint | 0 mismatches |
| Final default two-pass `brain.mgz` | 0/16,777,216 voxel mismatches |
| 284-byte MGH header and voxel payload | Exact |

The Python output is an exact prefix of the decompressed fresh native output. Native appended 996 further trailer bytes in the fresh paired run; the earlier completed subject output had 451 more trailer bytes than Python. Thus the complete compressed and decompressed files have different hashes, while image header, voxel data, and preexisting trailer bytes match. The fresh native output also matched the completed subject's `brain.mgz` voxel for voxel and in header and voxel payload. [Machine-readable paired report](normalize_second_fs_sub01_report.json); [validator](validate_normalize_aseg_stage.py). The diagnostic [ridge/initial-bias validator](validate_normalize_aseg_seed.py) additionally checks independent controls and float32 intermediates.

The critical source-order details were Fast Marching boundary values of ±0.5 and the C++ global `::sqrt(float)` promotion to double before converting the quadratic solution back to float32. An isolated C++ source-order probe using `std::sqrt(float)` initially disagreed with the installed binary at 133 ridge voxels; with global `::sqrt`, the probe and Python port both agreed with the installed ridge. The pre-fix Python final brain differed at 33,595 voxels, so final parity was rechecked after this correction.

## Single same-host timing

The candidate and native command each ran once, sequentially on control-host with the same input files. Times include process startup, file I/O, and compression. No warm-up or load matching was applied, so these are a single observation rather than a general speed estimate.

| Command | Wall time |
| --- | ---: |
| Python CPU standalone CLI | 75.37 s |
| Native FreeSurfer CPU CLI | 115.55 s |

The Python function reported 72.52 s resident time. Its internal breakdown was ridge and outliers 7.76 s; initial Voronoi/bias 11.03 s; gentle controls 1.77 s and bias 10.74 s; first 3D controls/bias 9.27/10.53 s; second 3D controls/bias 9.80/10.34 s. Native does not expose directly comparable per-kernel times in its normal CLI.

The source-matched first `mri_normalize -g 1` pass is documented separately in [`NORMALIZE_FIRST_PASS.md`](experimental/NORMALIZE_FIRST_PASS.md). Its focused arithmetic regressions still pass (2/2) after adding this second entry. The complete Python recon-all pipeline remains unvalidated.
