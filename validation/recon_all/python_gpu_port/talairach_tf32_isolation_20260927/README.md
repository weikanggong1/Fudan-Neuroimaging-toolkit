# Talairach affine precision isolation on one real T1

This is a targeted comparison, not an end-to-end recon-all acceptance. The
official reference is FreeSurfer 8.2 on the same T1. Input, weight, template,
source, and candidate output SHA-256 values are in
[`precision_report.json`](precision_report.json) and
[`connected_report.json`](connected_report.json); the bounded downstream
continuation is in [`gca_gate_report.json`](gca_gate_report.json). The reference operation is
`mri_synthmorph -m affine -t aff.lta synthstrip.mgz mni305.cor.stripped.mgz -j 4`.
The candidate calls `fnit.synthmorph.SynthMorph(..., model="affine", extent=256)`.
No official volume or transform was used as a candidate input.

## Fixed-input affine comparison

All rows use the same saved candidate `synthstrip.mgz`, MNI305 template and
affine weight. The source used in the test has the hashes recorded in the
report. Both cuDNN benchmark and deterministic flags were true, matching the
preceding SynthStrip state. Each row has one timed inference; the second
default row is a warm repeat, so these times are not a paired speed benchmark.

| Execution | matmul TF32 | cuDNN TF32 | Inference s | Max voxel-LTA element error | eTIV error mm³ |
|---|---:|---:|---:|---:|---:|
| CPU float32 | off | off | 17.836 | 0.00012255 | +0.48617 |
| GPU default, first | on | on | 2.200 | 0.04733276 | −822.54696 |
| GPU float32, both off | off | off | 4.025 | 0.00008392 | −0.89466 |
| GPU, matmul off | off | on | 1.511 | 0.00392628 | −3.52243 |
| GPU, cuDNN off | on | off | 1.483 | 0.07858276 | −826.74013 |
| GPU default, repeat | on | on | 1.412 | 0.04733276 | −822.54696 |

The default GPU world affine exactly reproduces the earlier v3 candidate
world affine. Disabling both TF32 switches only during the affine inference
reduced the absolute eTIV error from 822.547 to 0.895 mm³, about 919-fold.
It did not make the LTA or eTIV exactly equal to the official output.
No production precision default was changed.

## Connected input prefix

The separate connected trial starts from the same raw T1, runs candidate
SynthStrip and Talairach with both TF32 flags disabled only inside the
SynthMorph affine call, and then runs candidate `nu`, `T1`, and `brainmask`.
It reuses a saved **candidate** N4 `nu0.mgz`, since N4 does not read the
Talairach transform. It does not run GCA registration or any later stage.

| Candidate volume | Differing voxels versus official | Affine / MGH header |
|---|---:|---|
| `orig.mgz` | 0 / 16,777,216 | exact / exact |
| `synthstrip.mgz` | 0 / 16,777,216 | exact / exact |
| `nu.mgz` | 0 / 16,777,216 | exact / exact |
| `T1.mgz` | 0 / 16,777,216 | exact / exact |
| `brainmask.mgz` | 0 / 16,777,216 | exact / exact |

The connected voxel LTA has a maximum element error of 0.00008392 and yields
eTIV 1,310,265.657875 mm³ versus official 1,310,266.552537 mm³. Timings:
input plus Talairach 13.435 s (including SynthStrip 6.540 s and affine
Talairach 3.983 s); `nu` from saved `nu0` 1.818 s; `T1` normalization
62.867 s; `brainmask` 0.983 s. The full MGZ file hashes need not match when
history metadata differs; voxel arrays, affine matrices and the first 284
header bytes were compared separately.

`probe_precision.py` is a reusable CLI with moving image, template, weight
directory, official and prior affine/voxel LTAs, and output directory as
positional arguments. The recorded fixed-input run used an equivalent
scratch script whose SHA-256 is in `precision_report.json`.
`probe_connected.py` is the exact script used for the connected trial; it
accepts T1, empty candidate subject directory, weight directory, asset
directory, saved candidate `nu0.mgz`, and read-only official subject directory.
Both scripts support `--device cuda:0 --threads 4`. Their reference paths are
read for comparison only.

## Bounded GCA and presurface continuation

`probe_gca_gate.py` continues the saved candidate subject above. It runs the
same pinned Conda `mri_em_register` binary as the earlier v3 whole-subject
candidate (binary hash in the JSON), candidate Python `run_ca_normalize`,
and candidate Python callosum segmentation. It reuses saved **candidate**
`synthseg.rca.mgz` from the same T1 to avoid rerunning the network; that
input was independently compared with the official output before use.
No official image, label, transform, or surface is a reconstruction input.
The official equivalent registration command is
`mri_em_register -uns 3 -mask brainmask.mgz nu.mgz RB_all_2020-01-02.gca transforms/talairach.lta`.

| Output | Paired result |
|---|---|
| GCA `talairach.lta` | All 16 matrix elements exact |
| `norm.mgz`, `ctrl_pts.mgz` | Each 0 / 16,777,216 differing voxels; affine and MGH header exact |
| Saved candidate `synthseg.rca.mgz` | 0 / 16,777,216 differing voxels; affine and MGH header exact |
| `aseg.auto_noCCseg.mgz`, `aseg.auto.mgz`, `aseg.presurf.mgz` | Each 0 / 16,777,216 differing voxels; affine and MGH header exact |
| `cc_up.lta` | Maximum matrix element error 0.00000763; not exact |

The registration, CA normalization, and callosum/copy stages took 329.526,
37.143, and 25.454 s on this shared node. The old v3 timings for the same
operations were 251.249, 27.048, and 24.438 s, respectively, under a
different load; these are observations, not controlled speed ratios. The
new SynthMorph LTA/eTIV improvement is independent of GCA registration,
which reads `nu` and `brainmask` and creates a separate `talairach.lta`.
GCA through `aseg.presurf` numerically agrees on this T1, but `cc_up.lta`
still has a small transform difference. WM, surfaces, regional statistics,
and additional subjects remain outside this trial; the repository default
TF32 policy has not been changed.
