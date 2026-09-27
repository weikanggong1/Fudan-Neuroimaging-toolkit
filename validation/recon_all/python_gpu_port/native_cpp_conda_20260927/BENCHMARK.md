# Conda C++ recon-all: same-T1 end-to-end observation

Candidate profile: `experimental-native-gca-topology-metrics-sphere-registration-core-v1`; device `cuda:1`; 4 CPU threads.
Input SHA-256: `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`. Candidate status: **complete**.
Strict outputs: **6/138** passed; all passed: **False**.

## Observed total wall time

| Conda candidate | Archived FreeSurfer 8.2 | Candidate / official |
|---:|---:|---:|
| 2706.6 s | 6789.6 s | 0.399 |

These are one-run observations on a shared host at different times. The official result is an archived same-T1 run, not a paired cold-start trial. The fixed outputs do not all match, so this ratio is not an equivalent-reconstruction speedup.

## Candidate stages (inclusive wall time)

| Stage | Seconds |
|---|---:|
| `input_talairach` | 14.31 |
| `n4` | 195.65 |
| `nu` | 1.77 |
| `T1_normalize` | 67.12 |
| `brainmask` | 0.88 |
| `SynthSeg` | 8.26 |
| `mri_em_register` | 358.19 |
| `mri_ca_normalize` | 32.78 |
| `surface_lh` | 509.76 |
| `surface_rh` | 640.12 |
| `register_lh` | 269.10 |
| `register_rh` | 231.17 |
| `annot_lh_aparc` | 47.23 |
| `annot_lh_aparc.a2009s` | 63.99 |
| `annot_lh_aparc.DKTatlas` | 61.98 |
| `annot_rh_aparc` | 41.22 |
| `annot_rh_aparc.a2009s` | 57.61 |
| `annot_rh_aparc.DKTatlas` | 48.68 |
| `project_aparc_volumes` | 5.58 |
| `project_wmparc` | 1.97 |
| `brain_volume_stats` | 2.88 |
| `aseg_stats` | 16.73 |
| `wmparc_stats` | 13.51 |
| `stats_lh_aparc` | 1.40 |
| `stats_lh_aparc.a2009s` | 2.26 |
| `stats_lh_aparc.DKTatlas` | 1.54 |
| `stats_rh_aparc` | 1.69 |
| `stats_rh_aparc.a2009s` | 2.50 |
| `stats_rh_aparc.DKTatlas` | 1.34 |
| Other overhead | 5.39 |

`surface_lh/rh` include their nested topology, sphere, and metric steps; do not add the next table to this total.

## Candidate hemisphere substeps

| Hemisphere | Step | Seconds |
|---|---|---:|
| lh | `topology_python_seconds` | 9.30 |
| lh | `topology_native_seconds` | 51.99 |
| lh | `sphere.inflate_nofix` | 10.21 |
| lh | `sphere.qsphere_nofix` | 69.08 |
| lh | `sphere.inflate` | 12.19 |
| lh | `sphere.sphere` | 296.20 |
| lh | `metric.thickness` | 32.55 |
| lh | `metric.area` | 0.71 |
| lh | `metric.area.pial` | 0.74 |
| lh | `metric.curv` | 1.83 |
| lh | `metric.curv.pial` | 1.97 |
| rh | `topology_python_seconds` | 8.82 |
| rh | `topology_native_seconds` | 51.71 |
| rh | `sphere.inflate_nofix` | 11.91 |
| rh | `sphere.qsphere_nofix` | 80.39 |
| rh | `sphere.inflate` | 13.61 |
| rh | `sphere.sphere` | 409.25 |
| rh | `metric.thickness` | 35.62 |
| rh | `metric.area` | 0.76 |
| rh | `metric.area.pial` | 0.73 |
| rh | `metric.curv` | 1.80 |
| rh | `metric.curv.pial` | 1.99 |

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
| mri | 6 | 39 | 19 |
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
| `mri/aseg.mgz` | False | voxels: 16669967/16777216 exact; max |Δ| 253.0 |
| `mri/aparc+aseg.mgz` | False | voxels: 16648565/16777216 exact; max |Δ| 2035.0 |
| `mri/wmparc.mgz` | False | voxels: 16286729/16777216 exact; max |Δ| 4998.0 |
| `surf/lh.orig` | False | xyz shape: [117076, 3] candidate vs [106622, 3] official |
| `surf/lh.white` | False | xyz shape: [117076, 3] candidate vs [106622, 3] official |
| `surf/lh.pial` | False | xyz shape: [117076, 3] candidate vs [106622, 3] official |
| `surf/lh.sphere` | False | xyz shape: [117076, 3] candidate vs [106622, 3] official |
| `surf/lh.sphere.reg` | False | xyz shape: [117076, 3] candidate vs [106622, 3] official |
| `surf/lh.thickness` | False | vertex shape: [117076] candidate vs [106622] official |
| `surf/lh.area` | False | vertex shape: [117076] candidate vs [106622] official |
| `surf/lh.volume` | False | vertex shape: [117076] candidate vs [106622] official |
| `surf/lh.curv` | False | vertex shape: [117076] candidate vs [106622] official |
| `surf/lh.sulc` | False | vertex shape: [117076] candidate vs [106622] official |
| `surf/rh.orig` | False | xyz shape: [120273, 3] candidate vs [105541, 3] official |
| `surf/rh.white` | False | xyz shape: [120273, 3] candidate vs [105541, 3] official |
| `surf/rh.pial` | False | xyz shape: [120273, 3] candidate vs [105541, 3] official |
| `surf/rh.sphere` | False | xyz shape: [120273, 3] candidate vs [105541, 3] official |
| `surf/rh.sphere.reg` | False | xyz shape: [120273, 3] candidate vs [105541, 3] official |
| `surf/rh.thickness` | False | vertex shape: [120273] candidate vs [105541] official |
| `surf/rh.area` | False | vertex shape: [120273] candidate vs [105541] official |
| `surf/rh.volume` | False | vertex shape: [120273] candidate vs [105541] official |
| `surf/rh.curv` | False | vertex shape: [120273] candidate vs [105541] official |
| `surf/rh.sulc` | False | vertex shape: [120273] candidate vs [105541] official |
| `label/lh.aparc.annot` | False | vertex IDs shape: [117076] candidate vs [106622] official |
| `label/rh.aparc.annot` | False | vertex IDs shape: [120273] candidate vs [105541] official |
| `stats/aseg.stats` | False | outlier rows: 46; max numeric |Δ| 176858.279316 |
| `stats/wmparc.stats` | False | outlier rows: 74; max numeric |Δ| 9699.0 |
| `stats/lh.aparc.stats` | False | outlier rows: 44; max numeric |Δ| 176254.279316 |
| `stats/rh.aparc.stats` | False | outlier rows: 44; max numeric |Δ| 176254.279316 |

## Provenance

- Candidate run SHA-256: `2340e726c329c55927c1a0c5d6f4124d5fc53c374824b6b10277969b295a3b46`.
- Strict report SHA-256: `780faa4322964499f7da7eac1b1adc8693c9917a40e524c7d1750310e6f0abb1`.
- Archived official log SHA-256: `6badf1a4c26e564a29f78a173547f4151920400ff4bda1047d15135c87998a66`.
- Archived benchmark JSON SHA-256: `c41953438ccc6317e40ca953ed08844e2c18ca99542e2f0cd78d3e6728b19cde`.
