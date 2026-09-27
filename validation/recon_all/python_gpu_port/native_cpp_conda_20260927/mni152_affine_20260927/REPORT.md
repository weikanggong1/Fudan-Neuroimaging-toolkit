# Real-T1 Python MNI152 affine registration and auxiliary crop impact

## Input, output, and provenance

This isolated benchmark uses `examples/data/sub-01_T1w.nii.gz` (SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`). The archived FreeSurfer 8.2 subject is `work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official`; the v5 FNIT replay is `work/reconall_wm_agent_20260927/v5_prefix_replay_20260927/subjects/sub01`. Their `mri/orig.mgz` files differ in bytes (SHA-256 `c99c2462...` versus `7ac9286d...`), but **all 256³ voxels and the NIfTI affine are identical**. The tested package modules match the isolated worktree exactly: `src/fnit/synthmorph/pipeline.py` SHA-256 `8d8545628b3e9852eeb222d5afe6b5a0cd218a79d3804a575e907ba8e404a74c`; `src/fnit/recon_all/aux_seg.py` SHA-256 `d6dada2add96bd7e951e3eeb9da71b5450129cbf071533e8033475e67937c11a`.

Input templates on the host were checked against FNIT's asset catalog: cropped MNI152 image SHA-256 `ef89b7aa615dcb4a68a1cad725a9b99314227f9777ed86f42aae61e4e89da19a` (7,745,204 bytes), full MNI152 image `e4e1a25b66fef6b2cde8e4916d4a03d4369de6f24b0d3d4736baffc3d4a8c575` (17,977,963 bytes), cropped/full template LTA `2715856af855b0308350bbe32f8f932e3432d4c96acbbe107eee9dbb1c2b2e07` (1,620 bytes), and affine weight `1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6` (51,455,312 bytes). The additional reverse template LTA `2a80b62a9baceaa8c4312bf46d5468353b10ed2990e1704028907700cb13d790` is used by the official nonlinear warp stage; it is not needed to compose this auxiliary segmentation affine. The live remote scripts and outputs reside only in `work/mni152_affine_codex_20260927`; the report commits JSON, no T1, license, native executable, model or atlas bytes.

| Function | Inputs and call | Complete output and structure | Official equivalent |
| --- | --- | --- | --- |
| Python bounding crop | v5 `mri/orig.mgz`; nonzero voxel bounding box with three-voxel low pad and exclusive high stop | `candidate_crop.nii.gz`, 162×168×196 one-channel NIfTI, float/native intensity voxels and spatial affine; `crop_comparison.json` | `mri_mask -bb 3 orig.mgz orig.mgz invol.crop.nii.gz` |
| `SynthMorph(model="affine", extent=256)` | cropped native T1, `mni152.1.0mm.cropped.nii.gz`, affine checkpoint; `model(crop, template)` on `cuda:1` | `aff_pythoncrop.lta`, one 4×4 RAS-to-RAS transform with source/destination volume geometry; moved/fixed-moved Surfa volumes returned by API | `mri_synthmorph -m affine -t aff.lta invol.crop.nii.gz mni152.1.0mm.cropped.nii.gz -j 4` |
| Python LTA composition | affine world matrix, native `orig.mgz` voxel-to-world, full MNI152 voxel-to-world | `reg_targ_to_invol_pythoncrop_affine.lta`, type-0 voxel-to-voxel 4×4 matrix plus source/destination volume geometry; homogeneous last row forced to `[0,0,0,1]`; `comparison_sanitized.json` | `mri_concatenate_lta -invert1 -invertout aff.lta reg.crop-to-invol.lta reg.invol_to_croptarg.lta`, then `mri_concatenate_lta -invert2 reg.1.0mm.to.1.0mm.cropped.lta reg.invol_to_croptarg.lta reg.targ_to_invol.lta` |
| CPU auxiliary crop check | official and candidate type-0 LTAs, same official `nu.mgz`, two MCA priors and one venous-sinus prior | `aux_crop_impact_affine.json`: three native-grid prior comparisons, threshold hit counts, `80³`/`144³` model-input crop starts and SHA-256; no network inference | `mri_mcadura_seg --i nu.mgz ... --synthmorphdir ...`; `mri_vsinus_seg --s subject --rca-synthseg ... --synthmorphdir ...` |

The crop script and affine script are `work/mni152_affine_codex_20260927/python_crop.py` and `run_mni_affine.py`. The CPU auxiliary check is `compare_aux_crops_affine.py`. These are validation scripts with fixed data paths; no production runner modification was made.

## Numeric comparison

| Stage | Official vs Python on the same real T1 |
| --- | --- |
| Bounding crop | Start `[49,33,25]`, exclusive stop `[211,201,221]`, shape `162×168×196`; **0 mismatched voxels**; spatial affine maximum absolute difference `7.6294e-6` |
| World affine, using frozen official crop | 4×4 matrix maximum absolute element difference **0.0237113**; using Python crop **0.0237135** |
| Direct LTA composition from **official** affine | Official type-0 LTA maximum element difference **3.4961e-5 voxels**; this isolates composition error from inference |
| Python affine plus homogeneous LTA composition | Official LTA maximum element difference **0.068515 voxels**; over a 27-point grid of full-MNI voxels, mapped displacement mean **0.051357 mm**, maximum **0.089175 mm** |
| MCA left/right and venous-sinus model inputs | Resampled priors have small differences and threshold hit counts change (`5300→5285`, `5581→5577`, `72125→72122`), but crop starts stay **exactly** `[130,86,118]`, `[61,81,127]`, `[54,31,8]`; all three extracted model-input cubes have matching SHA-256 |

The affine matrix is not pointwise identical to official. The crop decision is robust **for this T1**: once the prior selects the same crop, the current Python auxiliary model receives the same intensity tensor. The segmentation network was not rerun in this benchmark, so matching final MCA/venous-sinus labels is an inference conditional on deterministic inference, not a measured output claim. The outputs of Python MCA/venous models on frozen official LTA have separate validation; this report tests the additional LTA integration sensitivity only.

## Measured time and memory

| Operation | Wall time | Execution context |
| --- | ---: | --- |
| Official `mri_mask -bb 3` isolated crop | 1.86 s | gpucw1 CPU |
| Python crop | 0.47 s | headcw CPU |
| Official `mri_synthmorph -m affine` archived log | 107.57 s | gpucw1 CPU, four threads, earlier run |
| Python SynthMorph model initialization | 4.32 s | gpucw1 H100 GPU1 |
| Python affine inference, frozen official crop | 1.29 s | gpucw1 H100 GPU1, first call |
| Python affine inference, Python crop | 0.57 s | gpucw1 H100 GPU1, warm second call |
| Python process with both inference calls, composition and serialization | 12.05 s | gpucw1; peak CUDA allocated 4.66 GB, reserved 5.50 GB |
| CPU auxiliary check, six prior resamples | 12.03 s | headcw CPU, four threads |

The runs differ in date, node, warm state and numerical output, so these times do not establish equivalent reconstruction acceleration. They show the affine network is a plausible low-memory GPU replacement whose downstream crop choices survived this measured error on one real T1.

## Minimal integration path and remaining gates

Create the Python nonzero bounding crop and verified target template in the subject's transform directory; call the existing package `SynthMorph(weights=weights_dir, device=device, model="affine", extent=256)` on the crop and MNI152 cropped image; compose world matrices into type-0 `reg.targ_to_invol.lta` using full MNI152 and native MRI voxel geometries, forcing the exact homogeneous row. The current `aux_seg` functions read that LTA. Their model lookup uses `FS_TORCH_MODEL_DIR` or `<assets>/models`, so the runner must explicitly point them to its separately configured weights directory. Then run MCA/dura and venous-sinus segmentation, reproduce the five official mask/edit calls that create `brain.finalsurfs.mgz`, and validate that volume voxelwise before white/pial placement.

The required MNI152 files are cataloged but currently downloaded through a **514,649,342-byte** archive. No small direct official URL has been verified. Keep them opt-in until a verified small-file distribution source is available. The native Python crop still has a `7.6e-6` geometry difference and the affine/LTA still has up to `0.089 mm` mapped displacement, so broader inputs and final auxiliary labels must be checked before claiming numerical parity.
