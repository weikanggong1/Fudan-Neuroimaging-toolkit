# Conda C++ recon-all: same-T1 end-to-end observation

Candidate profile: `experimental-native-gca-topology-metrics-sphere-registration-core-v2`; device `cuda:1`; 4 CPU threads.
Input SHA-256: `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`. Candidate status: **complete**.
Strict outputs: **8/138** passed; all passed: **False**.

## Observed total wall time

| Conda candidate | Archived FreeSurfer 8.2 | Candidate / official |
|---:|---:|---:|
| 2649.0 s | 6789.6 s | 0.390 |

These are one-run observations on a shared host at different times. The official result is an archived same-T1 run, not a paired cold-start trial. The fixed outputs do not all match, so this ratio is not an equivalent-reconstruction speedup.

## Candidate stages (inclusive wall time)

| Stage | Seconds |
|---|---:|
| `input_talairach` | 11.80 |
| `n4` | 198.52 |
| `nu` | 1.79 |
| `T1_normalize` | 64.25 |
| `brainmask` | 0.78 |
| `SynthSeg` | 7.80 |
| `mri_em_register` | 349.81 |
| `mri_ca_normalize` | 34.07 |
| `mri_cc` | 25.94 |
| `surface_lh` | 555.18 |
| `surface_rh` | 507.32 |
| `register_lh` | 218.80 |
| `register_rh` | 298.53 |
| `annot_lh_aparc` | 47.12 |
| `annot_lh_aparc.a2009s` | 60.29 |
| `annot_lh_aparc.DKTatlas` | 52.95 |
| `annot_rh_aparc` | 45.61 |
| `annot_rh_aparc.a2009s` | 58.11 |
| `annot_rh_aparc.DKTatlas` | 48.59 |
| `project_aparc_volumes` | 5.00 |
| `project_wmparc` | 1.81 |
| `brain_volume_stats` | 2.92 |
| `aseg_stats` | 14.22 |
| `wmparc_stats` | 13.89 |
| `stats_lh_aparc` | 2.33 |
| `stats_lh_aparc.a2009s` | 3.55 |
| `stats_lh_aparc.DKTatlas` | 2.22 |
| `stats_rh_aparc` | 2.56 |
| `stats_rh_aparc.a2009s` | 3.74 |
| `stats_rh_aparc.DKTatlas` | 2.49 |
| Other overhead | 6.98 |

`surface_lh/rh` include their nested topology, sphere, and metric steps; do not add the next table to this total.

## Candidate hemisphere substeps

| Hemisphere | Step | Seconds |
|---|---|---:|
| lh | `topology_python_seconds` | 8.89 |
| lh | `topology_native_seconds` | 43.69 |
| lh | `sphere.inflate_nofix` | 11.83 |
| lh | `sphere.qsphere_nofix` | 68.54 |
| lh | `sphere.inflate` | 9.89 |
| lh | `sphere.sphere` | 357.46 |
| lh | `metric.thickness` | 30.82 |
| lh | `metric.area` | 0.62 |
| lh | `metric.area.pial` | 0.75 |
| lh | `metric.curv` | 1.63 |
| lh | `metric.curv.pial` | 1.79 |
| rh | `topology_python_seconds` | 6.64 |
| rh | `topology_native_seconds` | 50.75 |
| rh | `sphere.inflate_nofix` | 11.65 |
| rh | `sphere.qsphere_nofix` | 80.47 |
| rh | `sphere.inflate` | 12.57 |
| rh | `sphere.sphere` | 284.73 |
| rh | `metric.thickness` | 34.61 |
| rh | `metric.area` | 0.81 |
| rh | `metric.area.pial` | 0.85 |
| rh | `metric.curv` | 1.90 |
| rh | `metric.curv.pial` | 2.11 |

## Archived official command timing

Each row is the sum of that command's `@#@FSTIME` elapsed entries. Wrapper commands can contain other commands, so these rows must not be summed as an end-to-end total.

| Command | Calls | Seconds |
|---|---:|---:|
| `mri_convert` | 3 | 13.59 |
| `mri_add_xform_to_header` | 2 | 0.64 |
| `lta_convert` | 3 | 0.23 |
| `mri_synthstrip` | 1 | 62.09 |
| `mri_synthseg` | 1 | 192.32 |
| `mri_synthmorph` | 3 | 518.67 |
| `rca-talairach` | 1 | 83.88 |
| `mri_concatenate_lta` | 2 | 0.24 |
| `mri_warp_convert` | 1 | 15.58 |
| `mri_ca_register` | 1 | 97.68 |
| `mri_vol2vol` | 1 | 1.22 |
| `fs-synthmorph-reg` | 1 | 563.29 |
| `mri_nu_correct.mni` | 1 | 225.30 |
| `mri_normalize` | 2 | 202.97 |
| `mri_em_register` | 1 | 251.75 |
| `mri_ca_normalize` | 1 | 65.96 |
| `mri_cc` | 1 | 130.45 |
| `seg2cc` | 1 | 131.25 |
| `mri_mask` | 3 | 4.32 |
| `mri_edit_wm_with_aseg` | 5 | 32.35 |
| `AntsDenoiseImageFs` | 1 | 31.82 |
| `mri_segment` | 1 | 64.38 |
| `mri_pretess` | 3 | 5.59 |
| `mri_fill` | 1 | 92.91 |
| `mri_tessellate` | 2 | 2.76 |
| `mris_extract_main_component` | 2 | 1.68 |
| `defect2seg` | 1 | 35.50 |
| `mris_remesh` | 2 | 64.50 |
| `mris_remove_intersection` | 2 | 9.50 |
| `mris_autodet_gwstats` | 2 | 8.27 |
| `mris_place_surface` | 16 | 1505.58 |
| `label-cortex` | 2 | 68.47 |
| `mri_label2label` | 2 | 38.08 |
| `mris_register` | 2 | 833.10 |
| `rca-surfreg` | 1 | 834.67 |
| `vertexvol` | 2 | 5.20 |
| `mris_curvature_stats` | 2 | 6.36 |
| `mris_volmask` | 1 | 244.68 |
| `mri_relabel_hypointensities` | 1 | 27.13 |
| `mri_surf2volseg` | 5 | 244.62 |
| `mri_brainvol_stats` | 1 | 6.80 |
| `mri_segstats` | 2 | 159.13 |
| `mris_label2annot` | 6 | 6.67 |
| `mris_anatomical_stats` | 4 | 16.62 |

## Strict 138-output comparison

| Output class | Passed | Checked | Missing candidate |
|---|---:|---:|---:|
| mri | 8 | 39 | 19 |
| vertex map | 0 | 46 | 14 |
| surface | 0 | 18 | 0 |
| label | 0 | 12 | 6 |
| stats | 0 | 23 | 12 |

Selected outputs (details and every region/vertex remain in the strict and diagnostic JSON):

| Output | Pass | Error summary |
|---|---|---|
| `mri/orig.mgz` | True | voxels: 16777216/16777216 exact; max |Δ| 0.0 |
| `mri/nu.mgz` | True | voxels: 16777216/16777216 exact; max |Δ| 0.0 |
| `mri/T1.mgz` | True | voxels: 16777216/16777216 exact; max |Δ| 0.0 |
| `mri/synthseg.rca.mgz` | True | voxels: 16777216/16777216 exact; max |Δ| 0.0 |
| `mri/aseg.mgz` | False | voxels: 16672230/16777216 exact; max |Δ| 42.0 |
| `mri/aparc+aseg.mgz` | False | voxels: 16649938/16777216 exact; max |Δ| 2035.0 |
| `mri/wmparc.mgz` | False | voxels: 16287082/16777216 exact; max |Δ| 4998.0 |
| `surf/lh.orig` | False | xyz shape: [116148, 3] candidate vs [106622, 3] official |
| `surf/lh.white` | False | xyz shape: [116148, 3] candidate vs [106622, 3] official |
| `surf/lh.pial` | False | xyz shape: [116148, 3] candidate vs [106622, 3] official |
| `surf/lh.sphere` | False | xyz shape: [116148, 3] candidate vs [106622, 3] official |
| `surf/lh.sphere.reg` | False | xyz shape: [116148, 3] candidate vs [106622, 3] official |
| `surf/lh.thickness` | False | vertex shape: [116148] candidate vs [106622] official |
| `surf/lh.area` | False | vertex shape: [116148] candidate vs [106622] official |
| `surf/lh.volume` | False | vertex shape: [116148] candidate vs [106622] official |
| `surf/lh.curv` | False | vertex shape: [116148] candidate vs [106622] official |
| `surf/lh.sulc` | False | vertex shape: [116148] candidate vs [106622] official |
| `surf/rh.orig` | False | xyz shape: [118913, 3] candidate vs [105541, 3] official |
| `surf/rh.white` | False | xyz shape: [118913, 3] candidate vs [105541, 3] official |
| `surf/rh.pial` | False | xyz shape: [118913, 3] candidate vs [105541, 3] official |
| `surf/rh.sphere` | False | xyz shape: [118913, 3] candidate vs [105541, 3] official |
| `surf/rh.sphere.reg` | False | xyz shape: [118913, 3] candidate vs [105541, 3] official |
| `surf/rh.thickness` | False | vertex shape: [118913] candidate vs [105541] official |
| `surf/rh.area` | False | vertex shape: [118913] candidate vs [105541] official |
| `surf/rh.volume` | False | vertex shape: [118913] candidate vs [105541] official |
| `surf/rh.curv` | False | vertex shape: [118913] candidate vs [105541] official |
| `surf/rh.sulc` | False | vertex shape: [118913] candidate vs [105541] official |
| `label/lh.aparc.annot` | False | vertex IDs shape: [116148] candidate vs [106622] official |
| `label/rh.aparc.annot` | False | vertex IDs shape: [118913] candidate vs [105541] official |
| `stats/aseg.stats` | False | outlier rows: 51; max numeric |Δ| 171632.66774700006 |
| `stats/wmparc.stats` | False | outlier rows: 74; max numeric |Δ| 9640.0 |
| `stats/lh.aparc.stats` | False | outlier rows: 44; max numeric |Δ| 171028.66774700006 |
| `stats/rh.aparc.stats` | False | outlier rows: 44; max numeric |Δ| 171028.66774700006 |

## Provenance

- Candidate run SHA-256: `b8e8b10ac72354b35b60991f43ef92dcf8c95f8595222589f0283e88372d9b11`.
- Strict report SHA-256: `165f57b93b3bb0cef59e2a71b5a35387eec87920f27c6d38d1a1a5e43aa69934`.
- Archived official log SHA-256: `6badf1a4c26e564a29f78a173547f4151920400ff4bda1047d15135c87998a66`.
- Archived benchmark JSON SHA-256: `c41953438ccc6317e40ca953ed08844e2c18ca99542e2f0cd78d3e6728b19cde`.
