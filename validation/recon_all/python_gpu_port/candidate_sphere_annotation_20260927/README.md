# Real-T1 LH sphere continuation from saved candidate smoothwm

This is a **saved-stage continuation**, not a new single-process T1-to-recon-all run. The input is the repository's deidentified real `sub-01_T1w.nii.gz` (SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`). The [upstream connected prefix](../white_connected_prefix_20260927/README.md) generated the 106,622-vertex LH `smoothwm` after candidate MRI, topology, white placement, and three smoothing passes. The candidate subject is at `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_finalsurfs_stage_20260927/white_candidate_no_official_topology_fslicense`; the completed FreeSurfer 8.2 subject at `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official` was read only by the comparison script. No official surface was used as a reconstruction input.

An earlier saved v5 subject has 106,695 LH vertices and 213,386 faces on `smoothwm`, `inflated`, `sphere`, and `sphere.reg`; its `sulc` and `aparc.annot` each have 106,695 entries. This candidate and the official subject have 106,622 vertices and 213,240 faces. The old downstream files cannot be reused on the new mesh; see [the saved-v5 audit](saved_v5_audit.json).

## Inputs, commands, and outputs

Conda `mris_inflate` read candidate `surf/lh.smoothwm` and wrote candidate `surf/lh.inflated` plus `surf/lh.sulc`. It used the private external FreeSurfer license. The native equivalent is `mris_inflate LH.SMOOTHWM LH.INFLATED` from the same subject's `scripts` directory. Python `run_standard_sphere(inflated, smoothwm, output, finish_device="cpu")` then read the two same-order triangle surfaces, optimized a conventional sphere on CPU, wrote `surf/lh.sphere`, and returned projection, metric, ordered update, finish, and total timing fields. Its CLI and FreeSurfer equivalent are:

```bash
python -m fnit.recon_all.sphere_standard_run \
  CANDIDATE/surf/lh.inflated CANDIDATE/surf/lh.smoothwm CANDIDATE/surf/lh.sphere \
  --finish-device cpu --report sphere_report.json
mris_sphere -threads 4 -seed 1234 CANDIDATE/surf/lh.inflated CANDIDATE/surf/lh.sphere
```

The run used the saved source snapshot under the same `reconall_finalsurfs_stage_20260927` scratch directory. SHA-256 of `sphere_standard_run.py` and `sphere_standard_line_search.py` was `7544ccb77153a7ac242b88d1c9752bdd08ce5765ba483cd644a4075e94e90405` and `8777cb96ca93ad0584d34e3f2d2f13827b1302dbaf600fc9454fdd39cd323296`, matching the current repository files. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMBA_NUM_THREADS` were each 4. The comparison helper [audit.py](audit.py) takes `CANDIDATE_SUBJECT OFFICIAL_SUBJECT OUTPUT.json` and writes resolved candidate paths, file hashes, ordered face/coordinate errors, morph errors, and annotation overlap when available. It reads but does not modify subjects.

## Paired saved-stage result

| Candidate output | Same-index comparison with saved official |
| --- | --- |
| `lh.smoothwm` input | Ordered faces exact; Euclidean mean / P99 / max **0.000311208 / 0.006172390 / 0.188170 mm**; 18 vertices >0.1 mm |
| `lh.inflated` | Ordered faces exact; **0.001061059 / 0.002213085 / 0.004201193 mm**; 0 vertices >0.1 mm |
| `lh.sulc` | Same vertex count; mean / P99 / max absolute scalar error **0.000505999 / 0.008345132 / 0.580831528**; 191 exact values |
| Python `lh.sphere` | Ordered faces exact; **3.480691977 / 5.641300784 / 6.873138365 mm**; **106,606 / 106,622** vertices >0.1 mm |

The [paired file audit](white_candidate_new_sphere_audit.json) contains realpaths and SHA-256 for every row. The candidate sphere SHA-256 is `7748a5108c4d722f90f22c584f82659db46b6d57bd2eb825e76ee1429174de9a`. The [full 331-update trajectory](white_candidate_lh_sphere_report.json) took **1144.18 s** inside the API and **1149.97 s** [shell wall](white_candidate_lh_sphere_time.txt), peak RSS **749,556 KiB**. Conda inflation took **12.72 s** [shell wall](white_candidate_lh_inflate_time.txt), peak RSS **166,052 KiB**. A prior Python sphere on frozen official surfaces took 243 updates and 372.82 s; an archived native sphere on different frozen inputs and node load took 240.64 s. Those earlier observations are not paired timing ratios.

The initial negative-area fraction was 0.022116% here versus 0.022112% in the [frozen-input Python run](../standard_sphere_lh_api_headcw.json). The first two update step sizes were 1027.937 and 18450.785 here, versus 1027.952 and 18396.798 on frozen inputs. At update index 2 they became **399.365 versus 132.050**; the first schedule difference occurs at index 8. The candidate run had 331 updates versus 243 frozen-input updates. Its [runtime warnings](white_candidate_lh_sphere_warnings.log) include NumPy float32 quadratic-fit overflow/invalid warnings; the warnings alone do not identify the cause of the later geometry error.

## Same-candidate-input official control

One isolated FreeSurfer 8.2 `mris_sphere` run on gpucw1 read copies of only the candidate `lh.inflated` and `lh.smoothwm`. `cmp` before and after execution and the [input hashes](candidate_sphere_official_control/input_sha256.txt) confirm both copies were byte-identical to the candidate inputs: `f3a2128de469dc3c8d06e2649f85f420b26c84f95aea3c7912f677b4e57afd30` and `6d1a5d30639f3678c2e825d69b816f9bc35d5411546145f6c0bcab29a5fd0b91`. The official binary SHA-256 was `c34ca308a7fa03acdb3f689bf6125cf3d0198c37c3a62992a29f68e631c73612`. `FS_LICENSE` used a private external file; no license content is archived. The output was written under `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_finalsurfs_stage_20260927/candidate_sphere_official_control/subjects/sub01/surf/lh.sphere`. The command used `-threads 4 -seed 1234` and a 600-second timeout. It exited successfully in **328.61 s**, peak RSS **217,164 KiB**; see the [time](candidate_sphere_official_control/time.txt) and [native log](candidate_sphere_official_control/mris_sphere.log).

The exact native executable and copied-input layout were:

```bash
CONTROL=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_finalsurfs_stage_20260927/candidate_sphere_official_control
/public/software/apps/Freesurfer/8.2.0-1/bin/mris_sphere -threads 4 -seed 1234 \
  "$CONTROL/subjects/sub01/surf/lh.inflated" "$CONTROL/subjects/sub01/surf/lh.sphere"
```

`FREESURFER_HOME`, `SUBJECTS_DIR`, `OMP_NUM_THREADS=4`, and the private `FS_LICENSE` were set before the command; `/usr/bin/time` and `timeout 600` wrapped it.

| Comparison of ordered sphere vertices | Mean / P99 / max 3D distance, mm | Vertices >0.1 mm |
| --- | ---: | ---: |
| Official binary on candidate inputs vs saved official recon-all sphere | 2.959344 / 5.437959 / 7.066581 | 106,579 / 106,622 |
| Python on candidate inputs vs official binary on **the same candidate inputs** | 1.652636 / 2.952841 / 3.557925 | 106,495 / 106,622 |

Both comparisons have identical ordered faces and counts. Full surface realpaths, hashes, and per-vertex summaries are in the [three-way control audit](candidate_sphere_official_control_audit.json). The Python/native same-input shell observations are 1149.97/328.61 s (3.50×) on gpucw1, but they ran sequentially with uncontrolled shared-node load; this is a one-run observation, not a stable speed ratio. The official binary's 2.96 mm shift on candidate inputs shows that the small upstream surface perturbation itself changes the optimizer's result substantially. The additional 1.65 mm Python-versus-official difference on those **identical inputs** shows that the Python implementation is also not parity-stable under this perturbation. Neither effect can be assigned solely to the other.

This candidate sphere does **not** meet same-index surface parity. Per the first-difference gate, no candidate `lh.sphere.reg` or `lh.aparc.annot` was generated from it, and no candidate final-white trial used an old, incompatible annotation. The next correction needs to address the upstream smoothwm/inflated discrepancy and the candidate-input Python/native sphere divergence before registration, annotation, and final-white metrics can be accepted.
