# Paired vertex error after exact bilateral `orig`

This is a **diagnostic replay on one real T1**, not a complete recon-all run. The candidate starts from independently produced FNIT `lh/rh.orig` surfaces whose ordered coordinates and faces were proven exact against FreeSurfer 8.2 ([topology report](../../../../../docs/recon_all/TOPOLOGY_CONDA_GA.md)). The MRI/aseg files come from the same T1's completed v3 candidate prefix; the official subject is read only. The scan is `examples/data/sub-01_T1w.nii.gz`, SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`. No official white, pial, label, or vertex map is used to generate candidate outputs.

The [probe](probe.py) creates an isolated FreeSurfer-style subject directory. For each hemisphere it copies the exact `orig`, runs the current `smooth_surface`, copies `smoothwm` to white, applies the current segmentation-normal-ray pial estimator, calls Conda `mris_place_surface` for thickness, area and curvature, and computes mid-area and vertex volume. Its inputs are `--orig-dir`, `--prefix-subject`, `--official-subject`, `--native-bin`, and `--assets`; its output is `--output-subject/{surf,label,scripts,mri}` plus `exact-orig-vertex-probe.json`. The JSON has `hemispheres.{lh,rh}.surfaces` with same-index coordinate errors and `maps` with one value per vertex, mean values, MAE, P50/P95/P99/max errors, and exact-value counts. It refuses an existing output directory or an `orig` that differs from official ordered geometry. The external license is passed only through `FS_LICENSE`.

The official comparison stages are `mris_smooth`/surface preparation, intensity-driven `mris_place_surface --white` and `--pial`, followed by `mris_place_surface --thickness WHITE PIAL 20 5`, `--area-map`, and `--curv-map`. Current FNIT only calls the latter metric commands on its approximate white/pial geometry; see [Conda stage commands](../../../../../docs/recon_all/CONDA_CPP_STAGES.md). This probe deliberately stops before registration, annotations, ribbon and ROI statistics.

| Map | LH official mean → candidate mean | LH paired MAE | RH official mean → candidate mean | RH paired MAE |
| --- | ---: | ---: | ---: | ---: |
| Thickness, mm | 2.079 → 2.967 | **1.080 mm** | 2.024 → 2.979 | **1.103 mm** |
| White area, mm²/vertex | 0.706 → 0.498 | 0.229 | 0.711 → 0.503 | 0.227 |
| Pial area, mm²/vertex | 0.872 → 1.286 | 0.696 | 0.876 → 1.263 | 0.678 |
| Mid-area, mm²/vertex | 0.789 → 0.892 | 0.369 | 0.794 → 0.883 | 0.361 |
| Gray volume, mm³/vertex | 1.682 → 2.548 | 1.269 | 1.658 → 2.559 | 1.264 |
| White curvature, 1/mm | −0.025 → −0.045 | 0.042 | −0.023 → −0.043 | 0.041 |
| Pial curvature, 1/mm | 0.054 → 0.048 | 0.155 | 0.054 → 0.049 | 0.147 |

The meshes retain exact official vertex IDs and ordered faces (LH 106,622/213,240; RH 105,541/211,078), so these **are paired per-vertex errors**. Their P99 and maximum errors are in [report.json](report.json). Mean absolute coordinate component differences from official are 0.319/0.314 mm for `smoothwm`, 0.443/0.432 mm for white, and 1.137/1.116 mm for pial (LH/RH). The first new error after exact `orig` is therefore surface preparation; white/pial placement magnifies it. Curvature, area and thickness calculations can match on identical input meshes, but their outputs here are inaccurate because their input geometry is different. Volume additionally depends on the candidate cortex label.

On gpucw1 CPU, this two-hemisphere probe exited 0 in **97.29 s**, peak RSS 656,436 KiB. Geometry generation took 7.11/9.39 s and the metric/label/volume stage took 37.50/38.25 s (LH/RH); these are nested in the 97.29 s process wall and must not be added to it. Conda native thickness calls alone took 28.01/28.48 s. The archived official recon-all executed more stages and used different inputs; these timings cannot be interpreted as an equivalent reconstruction speed comparison. [time.txt](time.txt) and the JSON retain raw evidence. The probe was run from the same pinned source/Conda environment as the v5 downstream replay, but is an independent controlled diagnostic.
