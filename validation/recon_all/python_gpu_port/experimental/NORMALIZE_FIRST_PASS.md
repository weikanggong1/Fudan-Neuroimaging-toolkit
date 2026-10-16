# `mri_normalize`: source translation and matched validation

This experiment covers `mri_normalize -g 1 -seed 1234 -mprage nu.mgz T1.mgz`: its 1D slice spline, initial gentle 3D correction, and two 3D iterations. The same implementation also supports isolated `-n 0` and `-n 1` checkpoints. The production recon-all entry still uses the native program while other stages are being ported.

Reference: FreeSurfer 8.2 source commit `d932c45b7941662ea380a05efef580568b98d41a`, specifically `mri_normalize/mri_normalize.cpp`, `utils/mrinorm.cpp`, `utils/mrifilter.cpp`, `utils/mrihisto.cpp`, `utils/histo.cpp`, and `utils/numerics.cpp`. The standalone Python port is under [`src/fnit/recon_all/normalization`](../../../../src/fnit/recon_all/normalization/), with [`normalize_t1`](../../../../src/fnit/recon_all/normalization/pipeline.py) as its Python API. It needs Python packages PyTorch, NumPy, SciPy, Numba, and NiBabel; it does not load a FreeSurfer binary.

## Frozen input and exactness

The input is `fs_sub01/mri/nu.mgz` (SHA-256 `ed0155a083a64ce3929a7598f61b45875ca7dcfef1000e9de42af62a807c7e3e`), with `mri/transforms/talairach.xfm` (SHA-256 `1e509f0e544614555ef77ba4df6c8f6ebfa4c73c4771d7464589250dfc607e59`). The 256³ image, transform, and native bundle are held on the shared cwStorage filesystem. All comparisons below use the same voxel indices and exact input files.

| Checkpoint | Candidate versus FreeSurfer | Evidence |
|---|---:|---|
| 1D output, before gentle | 0/16,777,216 differing voxels | Native `DIAG=0x8 DIAG_VERBOSE=1` `src0.mgz`; identical float32 voxel SHA-256 `ad61368bb71c438c50b311aff1acb14799696b1e52a12d3d3b511cc26e21f37e` |
| Gentle controls | 0/16,777,216 differing voxels | 7³ candidates 4,337; after 5³ 10,836; after sequential outlier removal 10,775; identical voxel SHA-256 `122ec98e6cdbaca4206b2b810d0e60e98d08cddafe3628842536e81f8d72658a` |
| 26-neighbor Voronoi field | 0/16,777,216 differing float32 voxels | Native `-sigma 0` `bias0.mgz`; identical voxel SHA-256 `3649627259bf7149969e3ba56b93259e89a94b4ec44e5795e04172b2e1a7d793` |
| Sigma-8 Gaussian bias, CPU strict path | 0/16,777,216 differing float32 voxels | Against native `DIAG=0x8` `bias0.mgz`; 65-tap FreeSurfer kernel and ordered float32 accumulation |
| Bias application | 0/16,777,216 differing voxels | Integer-truncated source and bias followed by FreeSurfer half-up rounding; `torch.round` alone differed at 4,327 voxels |
| Complete `-n 0` output, H100 GPU1 | 0/16,777,216 differing voxels; affine and 284-byte MGH header equal | Same-host native and candidate voxel SHA-256 `1e560f9d1a3e4b82715851ac42800d8a6065f8eb08a99c1ce647273a86ad9f75` |
| First 3D pass input/control/bias | 0/16,777,216 differing voxels in each float32/uint8 image | `src0.mgh`, `ctrl0.mgh`, `bias0.mgh`; final controls 63,806; WM/GM histogram peaks 108.841682/79.883804 |
| Complete `-n 1` output, control-host CPU | 0/16,777,216 differing voxels | Native DIAG, native noDIAG, and candidate common voxel SHA-256 `0942786f8ca04e38c5935860dda7f9a67f325cf0a80b9237a2aa1b108f591e55` |
| Second 3D pass input/control/bias | 0/16,777,216 differing voxels in each float32/uint8 image | `src1.mgh`, `ctrl1.mgh`, `bias1.mgh`; final controls 111,059; WM/GM histogram peaks 110.007790/45.003185 |
| Default two-pass final `T1.mgz`, control-host CPU | 0/16,777,216 differing voxels | Native isolated DIAG run, candidate, and existing recon-all `fs_sub01/mri/T1.mgz` common voxel SHA-256 `7634025c33c07f21933b05e98848acac188e4b8b68eaeeb1ad523a11788b3716` |

Native and candidate full decompressed MGH files differ in their trailing command-history/tag bytes. Their 284-byte header and voxel payload match for both the `-n 0` H100 pair and the default two-pass control-host noDIAG pair. The `-n 0` candidate compressed file SHA-256 is `542c6ccaa62022e5b9df48f30b447bbfda1e07c6f3d97d41b8eb52f411d5e47b`; native same-host file SHA-256 is `8f93846b2815a3e53aa67bf6f4c7d2ce36d17153248f4a7a5029f2f9769b8948`. For default two-pass control-host noDIAG, the candidate/native compressed file SHA-256 values are `ae2f5578a991dcb2dc5460c641bc60d38ba0c8d5d96ca1e54cd33e2ff1d4a01f` and `5f5f714eebda12cffed2cc6f6aa7c5f41fe073cee9d3560638ca8a656f6d484b`.

The native `DIAG` run exposed `src0.mgz`, `ctrl0.mgz`, and `bias0.mgz` before and during gentle correction. A second native run with `-sigma 0` exposed the Voronoi field before Gaussian smoothing; its 1D and control volumes were verified equal to the sigma-8 run. The Python intermediate outputs were each compared directly to those native files. The source's 23 histogram peaks and 12 retained spline peaks also match in position and intensity. This subject has a flat retained peak intensity of 112 at all 12 knots, so nonflat spline curves require another subject for validation.

An isolated native default run with `DIAG=0x8 DIAG_VERBOSE=1` exposed `src0/ctrl0/bias0` and `src1/ctrl1/bias1` as uncompressed MGH volumes. The control-host CPU Python diagnostic run emitted the corresponding volumes, and each pair matched exactly across all 16,777,216 voxels. The H100 full-run check compared the final T1 output; it did not emit these six H100 intermediates. The selected-control histogram and all-image histogram matched native counts bin for bin; smoothed histogram counts differed only at the native text plot's decimal printing precision. Retaining the *float32* image after gentle correction and after the first 3D iteration is necessary; converting either intermediate to uint8 changes the following controls.

| Default-run native/Python checkpoint | Common voxel-array SHA-256 | Differing voxels |
|---|---|---:|
| First 3D float32 input, `src0.mgh` | `8f28c1113f75244258e0a925c76e12953c13e26618f78b2a4e888f57251220c1` | 0/16,777,216 |
| First 3D uint8 control, `ctrl0.mgh` | `2bf3d2ccba8e6ab4988fc6bdf59f5652c08840152e98cd7c81f9571987194549` | 0/16,777,216 |
| First 3D float32 bias, `bias0.mgh` | `426b14ccd20a0d1ee50efa202ed7eea75aca4f566e5c30cb5cd5dbe3475d8668` | 0/16,777,216 |
| Second 3D float32 input, `src1.mgh` | `69b6293507d9a484dd8c91b3401faaf8ef2c552a5a9df8f839b0debf884a00db` | 0/16,777,216 |
| Second 3D uint8 control, `ctrl1.mgh` | `edba5db216ad0eb5a41b111bfb33d59eab5257c603da0a5bf57ade5f285ee8bc` | 0/16,777,216 |
| Second 3D float32 bias, `bias1.mgh` | `3bb512e1198468a52119f79cffc84c000a5cb8ed5bf463613edca2b19907be4c` | 0/16,777,216 |
| Final uint8 `T1.mgz`, CPU and H100 pairs | `7634025c33c07f21933b05e98848acac188e4b8b68eaeeb1ad523a11788b3716` | 0/16,777,216 |

## Time on the same input

| Host and scope | Native `-n 0` | Python candidate | Candidate stages |
|---|---:|---:|---|
| accelerator-a-1, one fresh CLI run, H100 GPU1 | 29.77 s | 11.79 s | In-process 7.923 s: 1D 0.271, controls 2.016, Voronoi 2.407, Gaussian 0.304, application/write 0.724 s |
| control-host, one fresh CPU CLI run | 17.93 s | 14.60 s | In-process 12.323 s: 1D 0.098, controls 1.542, Voronoi 7.809, Gaussian 2.212, application/write 0.531 s |
| control-host, isolated `-n 1`, mixed diagnostic settings | Native DIAG 55.08 s | Candidate noDIAG 64.92 s | Candidate in-process 62.52 s, of which 3D controls 39.18 s; this row is **not** a matched diagnostic comparison |
| control-host, isolated default two-pass run, both writing stage diagnostics | 82.64 s | 141.27 s | Candidate in-process 138.64 s: 3D controls 39.15 + 65.20 s, Voronoi 6.86 + 7.07 s, Gaussian 2.29 + 2.39 s |
| accelerator-a-1 H100 GPU1, default two-pass candidate before ROI cropping | Same-host native from later pair 116.31 s | 226.35 s | In-process 221.61 s: CPU control selection 68.74 + 140.64 s, GPU Voronoi 2.35 + 2.19 + 2.70 s, GPU Gaussian 0.24 + 0.07 + 0.07 s; this is an unpaired pre-optimization run |
| control-host, default two-pass after ROI cropping, both noDIAG | 76.94 s | 60.55 s | Same-host CLI 1.27× faster; candidate in-process 58.28 s, controls 9.76 + 17.05 s, final 0 voxel differences |
| accelerator-a-1 H100 GPU1, default two-pass after ROI cropping, both noDIAG | 116.31 s | 75.60 s | Same-host CLI 1.54× faster; in-process 68.87 s: CPU controls 20.24 + 32.60 s, GPU Voronoi 4.47 + 2.50 + 2.63 s, GPU Gaussian 0.13 + 0.07 + 0.07 s; final 0 voxel differences |

The accelerator-a-1 CLI comparison is **2.53× faster for `-n 0` only**, with exact final voxel agreement. The old full default CPU candidate was slower in the diagnostic run; 3D control selection accounted for most of its time. After ROI cropping, the paired control-host noDIAG run is 1.27× faster than native. These are one-subject measurements and are not full recon-all speed estimates. The Python GPU path retains CPU histogram, control selection, chessboard distance transform, and wavefront index sorting; PyTorch CUDA handles 1D voxel scaling, Voronoi averages, sigma-8 smoothing, and bias application. The CPU path uses NumPy/SciPy plus Numba for the strict Gaussian loops. Cropping repeated control-neighbor convolutions to the eligible bounding box reduced control-host isolated per-pass control times from 39.15 to 13.29 s and 65.20 to 22.77 s; both control masks remained exact.

### Matched noDIAG commands and timing scope

The two final matched pairs ran the same `nu.mgz` and `talairach.xfm` on each named host. On accelerator-a-1, `CUDA_VISIBLE_DEVICES=1` mapped physical H100 GPU1 to PyTorch `cuda:0`. No warm-up was subtracted: CLI time is `/usr/bin/time` wall time including Python startup, imports, I/O, and MGH compression. Candidate `total_seconds` and step times are inside the resident Python function. Native substep timings are not exposed by the normal CLI, so the table compares total CLI time while the candidate JSON reports its own stages.

```bash
BASE=/data/workspaces/freesurfer_synth/work/reconall_python_gpu_20260925/normalize_diag_1d
INPUT=/data/workspaces/freesurfer_synth/work/reconall_main_20260924/single_subjects/fs_sub01/mri/nu.mgz
XFM=/data/workspaces/freesurfer_synth/work/reconall_main_20260924/single_subjects/fs_sub01/mri/transforms/talairach.xfm
BUNDLE=/data/workspaces/freesurfer_synth/work/reconall_main_v07_20260924/bundle
LICENSE=/data/workspaces/freesurfer_synth/work/reconall_external_models_20260925/private_license.txt
cd "$BASE"

# control-host CPU candidate, then native; both noDIAG
/usr/bin/time -f 'wall_seconds=%e' env PYTHONPATH="$BASE/overlay" python3 -m fnit.recon_all.normalization --input "$INPUT" --xfm "$XFM" --output n2/T1_python_roi_control-host.mgz --device cpu
/usr/bin/time -f 'wall_seconds=%e' env FS_LICENSE="$LICENSE" FREESURFER_HOME="$BUNDLE" "$BUNDLE/bin/mri_normalize" -g 1 -seed 1234 -mprage "$INPUT" n2/T1_native_control-host_nodiag.mgz

# accelerator-a-1 H100 candidate, then native; both noDIAG
/usr/bin/time -f 'wall_seconds=%e' env CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$BASE/overlay" /data/workspaces/freesurfer_synth/work/reconall_main_20260924/venv/bin/python -m fnit.recon_all.normalization --input "$INPUT" --xfm "$XFM" --output n2/T1_python_roi_gpu1.mgz --device cuda:0
/usr/bin/time -f 'wall_seconds=%e' env CUDA_VISIBLE_DEVICES=1 FS_LICENSE="$LICENSE" FREESURFER_HOME="$BUNDLE" "$BUNDLE/bin/mri_normalize" -g 1 -seed 1234 -mprage "$INPUT" n2/T1_native_accelerator-a-1_n2.mgz
```

The remote `overlay` was a staged copy of the package source; the executed `pipeline.py` SHA-256 matches the source in this repository (`ad9a206a35e6ac2334f1c8006ef3f6245985a14d6169e7c8e308171162e29ec5`). The native executable appears only in the benchmark commands. The production [`normalization` package](../../../../src/fnit/recon_all/normalization/) imports NiBabel, NumPy, SciPy, Numba, PyTorch, and this repository's pure Python MGH writer; it neither invokes nor reads any FreeSurfer executable, license, or model file. Its `normalize_t1` API and `fnit-normalize` CLI perform the same stage; neither is wired into the existing standalone recon-all dispatcher yet.

`PYTHONPATH=src python3 -m pytest -q test_normalize_source.py` passed 2/2 small arithmetic tests on control-host after promotion into the package. Real-subject comparisons and logs are in `/data/workspaces/freesurfer_synth/work/reconall_python_gpu_20260925/normalize_diag_1d/` (`python_T1_n0_gpu1.log`, `native_accelerator-a-1_n0.log`, `n1/`, `n2/`, and intermediate images).

**Remaining gate:** validate on more than one T1 input, including a subject with nonflat 1D spline knots; then assess integration with the larger Python recon-all. This single-subject result establishes default `mri_normalize` voxel parity for `fs_sub01` on CPU and H100, not parity for other inputs or full recon-all.
