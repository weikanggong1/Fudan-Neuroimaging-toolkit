# SynthStrip precision and the first downstream voxel difference

## Function and usage

`run_input_talairach_chain(t1, subject_dir, weights_dir, assets_dir, device="cuda:1", threads=4)` imports and conforms one T1, runs the existing PyTorch SynthStrip, then writes the SynthMorph affine. Its CLI is:

```bash
python -m fnit.recon_all.input_talairach_chain T1.nii.gz /empty/sub01 \
  --weights-dir /weights --assets-dir /assets --device cuda:1 --threads 4
```

The external weight is `synthstrip.1.pt`. The FreeSurfer 8.2 stage being replaced is `mri_synthstrip --threads 4 -i orig.mgz -o synthstrip.mgz`. The following native masking operation is `mri_mask T1.mgz synthstrip.mgz brainmask.mgz`. This function changes only the cuDNN precision of the SynthStrip forward pass: its constructor enables TF32; the recon-all caller disables cuDNN TF32 for inference and restores the previous flag afterwards. Other GPU stages retain their own precision settings. It uses float32, with no float16 or bfloat16.

## Same T1 validation on gpucw1

The fixed input T1 has SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`. The already completed candidate's `orig.mgz`, `T1.mgz`, and `nu.mgz` were voxel-exact against the archived official subject. The `orig.mgz` file SHA-256 is `a7944bbdb8618bc3a83dae65c49986b25439cc33d962da5a0b10afec8e361938`; the official SynthStrip image SHA-256 is `f95c4e04b91a5001f88ddbb24af5d33eda4e4aa97d338a86477f71a91bab48de`.

| Frozen input, 256³ | SynthStrip voxels different from official | Brainmask voxels different from official | Single-run SynthStrip time |
| --- | ---: | ---: | ---: |
| CUDA1, cuDNN TF32 enabled (previous caller) | 35 / 16,777,216 | 35 / 16,777,216 | 5.17 s |
| CUDA1, cuDNN TF32 disabled | **0 / 16,777,216** | **0 / 16,777,216** | 10.25 s |
| CPU | **0 / 16,777,216** | — | 12.44 s |
| Archived FreeSurfer `mri_synthstrip` | reference | reference | 62.09 s |

The timing rows are single observations on a shared host, and the official time is archived rather than concurrently paired. An integrated call through the patched `run_input_talairach_chain` on the same T1 took 9.80 s for SynthStrip, produced `orig.mgz` and `synthstrip.mgz` with **zero voxel differences**, and restored the entering cuDNN TF32 flag. Production `mask_volume(T1.mgz, synthstrip.mgz, brainmask.mgz, device="cuda:1")` then produced an official-exact 16,777,216-voxel brainmask. MGZ compressed file hashes can differ despite identical voxel data; the fixed SynthStrip replay file SHA-256 is `d27170d33230cb3b342ca44fce1ad057c02eaccd71b84c57b14d9583f3047412`.

## Propagation check

The existing source-built Conda `mri_em_register` was rerun using the fixed mask and the candidate's exact `nu.mgz`. Its `talairach.lta` 4×4 matrix matched the official matrix in every element; the old candidate differed by up to `5.132e-5`. The replay took 296.09 s; the archived official command took 251.75 s. The existing Python `run_ca_normalize` then used the fixed mask, exact LTA, exact `nu.mgz`, and the same GCA: `norm.mgz` matched all 16,777,216 voxels and `ctrl_pts.mgz` matched all 100,663,296 values. Its replay took 33.20 s; the archived official `mri_ca_normalize` command totaled 65.96 s. These timings are unpaired and show no controlled speed ratio.

The isolated fixed-input test establishes parity through CA normalization. A new full recon-all run has **not** been counted in the old 8/138 output result. A later [same-input SynthMorph precision isolation and connected presurface check](talairach_tf32_isolation_20260927/README.md) reduced its voxel-LTA/eTIV error with a local affine-only TF32 override; the XFM still is not exactly equal, and cortical ROI or vertex parity remains unverified.
