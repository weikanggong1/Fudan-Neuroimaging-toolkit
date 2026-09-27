# Conda C++ recon-all: complete-subject output comparison

## Scope and decision

The Conda-built Python/C++ candidate completed 29 stages for `sub-01_T1w.nii.gz` (`run.json`: `status=complete`, 2706.64 s). This report compares its subject directory with the archived FreeSurfer 8.2 official reconstruction of the same T1 input. The [strict comparator](../compare_complete_subject.py) checked 138 specified outputs: **6 pass, 81 exist but fail, and 51 are missing from the candidate**. Its exit status was 1, as expected for a failed parity gate. The completed workflow is not yet equivalent to official `recon-all`.

The strict result is [strict_138.json](strict_138.json); the quantitative diagnostic is [comparison.json](comparison.json). `pass` tests decoded values and required structure, not file SHA-256: even the six passing MGZ files have different compressed-file hashes. For integer volumes the comparator requires exact voxels; for floating volumes it permits 1e-6 absolute error. It also checks data type, affine within 1e-6, and an exact header. Surface coordinates require 1e-5 mm, ordered faces exact; morphology uses 0.001 absolute tolerance for area maps or 0.005 for other maps, plus 0.001 relative tolerance. Annotation vertex IDs and color tables must match; numeric stats fields permit 0.005.

| Strict output group | Checked | Passed | Present, failed | Candidate missing |
|---|---:|---:|---:|---:|
| `mri/` | 39 | 6 | 14 | 19 |
| `surf/` | 64 | 0 | 50 | 14 |
| `label/` | 12 | 0 | 6 | 6 |
| `stats/` | 23 | 0 | 11 | 12 |
| **Total** | **138** | **6** | **81** | **51** |

The six passing files are `mri/T1.mgz`, `mri/nu.mgz`, `mri/orig.mgz`, `mri/orig/001.mgz`, `mri/rawavg.mgz`, and `mri/synthseg.rca.mgz`. Every voxel in each file matches; the first, second, third, and sixth each contain 16,777,216 voxels, while `orig/001.mgz` and `rawavg.mgz` each contain 10,223,616. Affines and headers pass. The equality of the SynthSeg label volume does **not** extend to the downstream `aseg.mgz`.

## Voxel and segmentation differences

Each count below is the number of unequal voxels among 16,777,216 positions in the corresponding same-shaped volume. A large label-value MAE can reflect a different atlas code; the count is more interpretable for categorical maps.

| Output | Unequal voxels | Observation |
|---|---:|---|
| `mri/synthstrip.mgz` | 38 | Matches the scale of the `brainmask` discrepancy. |
| `mri/brainmask.mgz` | 38 | Affine and header match. |
| `mri/aseg.auto.mgz` | 2,263 | Early anatomical segmentation differs despite exact SynthSeg labels. |
| `mri/aseg.presurf.mgz` | 2,263 | Same unequal-voxel count as `aseg.auto`. |
| `mri/aseg.mgz` | 107,249 | Official final segmentation includes postprocessing absent or different in the candidate; candidate data type is float32, official is int32. |
| `mri/norm.mgz` | 494,932 | This full-subject comparison uses different upstream brainmask/LTA inputs; a separate *identical-input* `mri_ca_normalize` test produced exact `norm.mgz` voxels (see [paired test](CA_NORMALIZE_CANDIDATE_SAME_INPUT.md)). |
| `mri/brain.mgz` | 956,162 | Downstream brain image differs. |
| `mri/wm.mgz` | 435,774 | White-matter segmentation differs substantially. |
| `mri/filled.mgz` | 90,420 | Hemisphere fill differs. |
| `mri/aparc+aseg.mgz` | 128,651 | Desikan atlas projection/final aseg combination differs. |
| `mri/aparc.DKTatlas+aseg.mgz` | 126,192 | DKT atlas projection differs. |
| `mri/aparc.a2009s+aseg.mgz` | 146,949 | a2009s atlas projection differs. |
| `mri/wmparc.mgz` | 490,487 | White-matter parcellation differs. |

The exact `synthseg.rca.mgz` followed by 2,263 unequal `aseg.auto` voxels localizes one divergence *after* the SynthSeg output. The much larger 107,249-voxel final `aseg` difference is consistent with the missing/different final postprocessing chain, but this report alone does not assign every voxel to a specific command. The [upstream diagnosis](UPSTREAM_DIAGNOSIS.md) gives a separate transition-level comparison.

## Meshes and vertex maps

The same vertex order is a prerequisite for a pointwise metric test. It fails at the first compared cortical meshes, so **none of the 32 diagnostic vertex maps has valid one-to-one vertex correspondence**. The table reports distributions, not paired vertex errors.

| Hemisphere | Official vertices / faces | Candidate vertices / faces | Vertex difference |
|---|---:|---:|---:|
| Left `orig`, `white`, `pial`, `sphere.reg` | 106,622 / 213,240 | 117,076 / 234,148 | +10,454 (+9.80%) |
| Right `orig`, `white`, `pial`, `sphere.reg` | 105,541 / 211,078 | 120,273 / 240,542 | +14,732 (+13.96%) |

The white-surface candidate-to-official nearest-vertex mean distance is 1.018 mm left and 1.044 mm right. On pial it is 1.434 mm left and 1.475 mm right; the candidate-to-official 95th percentiles are 5.798 and 6.450 mm. These nearest-vertex distances are descriptive spatial distances, not geometric correspondence or evidence of thickness equivalence.

| Vertex map | Left mean: official → candidate | Right mean: official → candidate |
|---|---:|---:|
| `thickness` (mm) | 2.07854 → 2.90596 | 2.02429 → 2.90564 |
| `area` (mm²/vertex) | 0.70626 → 0.50375 | 0.71091 → 0.50605 |
| `area.pial` (mm²/vertex) | 0.87232 → 1.34755 | 0.87610 → 1.30493 |
| `volume` (mm³/vertex) | 1.68205 → 2.51324 | 1.65794 → 2.45516 |
| `curv` | −0.02522 → −0.04381 | −0.02290 → −0.00695 |
| `curv.pial` | 0.05383 → 0.03635 | 0.05410 → −0.00320 |

All 12 examined `orig`/`white`/`pial`/`inflated`/`sphere`/`sphere.reg` geometries and all 32 examined vertex maps fail strict comparison. A spatial-nearest resampling diagnostic exists in `comparison.json`, but it must not be described as pointwise FreeSurfer parity.

## Atlas labels and statistical outputs

All six present Desikan, a2009s, and DKT annotation files fail because their vertex ID vectors have the different mesh lengths above. Their color tables and region names are identical. Spatial-nearest region Dice medians are descriptive only: left/right Desikan 0.931/0.896, a2009s 0.850/0.856, DKT 0.941/0.930. They are **not** vertexwise atlas agreement.

All nine diagnostic stats files (`lh`/`rh` × three cortical atlases, `aseg.stats`, `wmparc.stats`, `brainvol.stats`) differ. Selected parsed official → candidate values:

| Statistic | Official | Candidate | Difference |
|---|---:|---:|---:|
| `lh.aparc.stats` MeanThickness (mm) | 2.23998 | 2.78127 | +0.54129 |
| `rh.aparc.stats` MeanThickness (mm) | 2.18352 | 2.78356 | +0.60004 |
| `lh.aparc.stats` WhiteSurfArea (mm²) | 70,554.5 | 52,592.4 | −17,962.1 |
| `rh.aparc.stats` WhiteSurfArea (mm²) | 70,299.6 | 54,111.4 | −16,188.2 |
| `aseg.stats` BrainSegVol (mm³) | 1,207,086 | 1,268,060 | +60,974 |
| `aseg.stats` CortexVol (mm³) | 353,949.844 | 530,204.123 | +176,254.279 |
| `aseg.stats` eTIV (mm³) | 1,310,266.468 | 1,309,430.973 | −835.495 |

The 34 Desikan regions are present on both sides, but all 44 comparable lines in each `lh.aparc.stats` and `rh.aparc.stats` fail the strict check. For example, left superior frontal `GrayVol` is 18,258 → 24,869 mm³, `SurfArea` 6,702 → 4,856 mm², and `ThickAvg` 2.283 → 2.780 mm. The candidate left a2009s table lacks `G_subcallosal` (73 versus 74 parsed regions); `aseg.stats` lacks 15 official regions, and `wmparc.stats` lacks `Left-UnsegmentedWhiteMatter` and `Right-UnsegmentedWhiteMatter`. These differences remain even though the SynthSeg label volume passes.

## Missing strict outputs

The strict manifest expects all 51 files below; each exists in the official reference and is absent from the candidate. The diagnostic comparator checks a smaller subset and lists only 23 missing paths, so its missing count must not replace this strict count.

- **MRI (19):** `mri/antsdn.brain.mgz`, `mri/aseg.presurf.hypos.mgz`, `mri/brain.finalsurfs.manedit.mgz`, `mri/brain.finalsurfs.mgz`, `mri/entowm.mgz`, `mri/filled.auto.mgz`, `mri/lh.ribbon.mgz`, `mri/mca-dura.mgz`, `mri/mrisps.white.mgz`, `mri/mrisps.wpa.mgz`, `mri/rh.ribbon.mgz`, `mri/ribbon.mgz`, `mri/surface.defects.mgz`, `mri/transforms/synthmorph.1.0mm.1.0mm/test.nii.gz`, `mri/transforms/synthmorph.1.0mm.1.0mm/warp.to.mni152.1.0mm.1.0mm.inv.nii.gz`, `mri/transforms/synthmorph.1.0mm.1.0mm/warp.to.mni152.1.0mm.1.0mm.nii.gz`, `mri/vsinus.mgz`, `mri/wm.asegedit.mgz`, `mri/wm.seg.mgz`.
- **Surface (14):** `surf/lh.w-g.pct.mgh`, `surf/rh.w-g.pct.mgh`; for each hemisphere, `avg_curv`, `jacobian_white`, `smoothwm.BE.crv`, `smoothwm.C.crv`, `smoothwm.FI.crv`, and `smoothwm.S.crv`.
- **Atlas (6):** for each hemisphere, `BA_exvivo.annot`, `BA_exvivo.thresh.annot`, and `mpm.vpnl.annot`.
- **Statistics (12):** for each hemisphere, `aparc.pial.stats`, `BA_exvivo.stats`, `BA_exvivo.thresh.stats`, `curv.stats`, and `w-g.pct.stats`; plus `stats/entowm.stats` and `stats/vsinus.stats`.

The most consequential present mismatches are the filled/white segmentation, white and pial placement, topology and vertex count, and official final aseg processing. C++ source parity on an *identical intermediate input* can establish that a ported command is faithful, but it cannot by itself make the end-to-end outputs equivalent while preceding inputs and omitted stages differ. The observed total runtime must therefore not be called an equivalent-reconstruction speedup.
