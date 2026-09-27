# FNIT v3 end-to-end recon-all: same-T1 comparison with FreeSurfer 8.2

## Scope and reproducibility

This is **one completed reconstruction of a real T1**, `examples/data/sub-01_T1w.nii.gz` (SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`). The FreeSurfer 8.2 reference and FNIT v3 candidate used this same file, but ran at different times on a shared node. FNIT source commit was `c925c3c19dd0a8df8c5f9ab9c0438c8b02d966e8` (source archive SHA-256 `06374902a292f7a9b502a741b0eee5776c27cfeb59c04792a5aa7fca60e1ef7b`). FNIT completed with exit code 0 and a `complete` run report. The candidate is a **hybrid Conda Python/C++ implementation**: eight source-built FreeSurfer binaries are recorded in `provenance.txt`; the surface placement and later projection still use approximations.

Reference command:

```bash
/public/software/apps/Freesurfer/8.2.0-1/bin/recon-all -i /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/examples/data/sub-01_T1w.nii.gz -s a_official -sd /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_benchmark_pair_ac_20260924/official_subjects -all -parallel -openmp 4 -itkthreads 1
```

Candidate command (shown with its actual paths and options):

```bash
/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_cpp_conda_20260927/conda_env/bin/python -m fnit.recon_all.native_free /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/examples/data/sub-01_T1w.nii.gz /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_wm_agent_20260927/v3_e2e/subjects/sub01 --weights-dir /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_wm_agent_20260927/v3_weights --assets-dir /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_wm_agent_20260927/v3_assets --device cuda:0 --threads 4 --n4-python /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_cpp_conda_20260927/conda_env/bin/python --native-bin-dir /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_wm_agent_20260927/fs_cpp_eight/bin --native-topology --native-sphere --native-surface-metrics --native-registration
```

The evaluator reads the two FreeSurfer subject directories, the candidate stage JSON, the fixed 138-file strict comparator, and the spatial comparator. It writes `benchmark_summary.json`, a dictionary with `strict.by_class`, `volume_failures`, `first_surface_divergence`, `vertex_map_distributions_not_pointwise`, `roi`, `stages_seconds`, and `nested_surfaces`. Geometry arrays are `V × 3` coordinates and `F × 3` ordered face indices. Vertex maps are one value per vertex. The isolated analysis script is `summarize.py` beside this report; the per-file checks and all 34-region rows remain in the raw comparator JSON files. This script is diagnostic and runs on CPU using nibabel, NumPy, and SciPy. It is not a production pipeline stage.

## Completeness and strict accuracy

The 138-file comparison passed **19/138** outputs. Another **47** reference outputs were absent from the candidate, and **72** existed but failed. Passing means the comparator's strict numerical/metadata contract, **not byte-identical files**. In this run, all 19 passing MRI volumes have exactly equal voxels, affines and checked headers; every one has a different compressed-file SHA-256. The comparator also checks dtype, geometry and numeric tolerances for other file types.

| Output class | Checked | Passed | Missing candidate | Present but failed |
| --- | ---: | ---: | ---: | ---: |
| MRI volumes | 39 | 19 | 15 | 5 |
| Surfaces | 18 | 0 | 0 | 18 |
| Vertex maps | 46 | 0 | 14 | 32 |
| Annotations | 12 | 0 | 6 | 6 |
| Statistics | 23 | 0 | 12 | 11 |
| **Total** | **138** | **19** | **47** | **72** |

The 19 passing MRI files include `T1.mgz`, `norm.mgz`, `brainmask.mgz`, `aseg.presurf.mgz`, `wm.seg.mgz`, `wm.mgz`, and `filled.mgz`. These establish meaningful agreement in much of the volumetric input chain. The final `aseg.mgz` differs at **104,986 / 16,777,216** voxels (99.374% label agreement), and its stored dtype/header differ (`>i4` official versus `>f4` candidate). The final `aparc+aseg.mgz` differs at **122,770** voxels (99.268% agreement); `aparc.DKTatlas+aseg.mgz` differs at **120,789**, `aparc.a2009s+aseg.mgz` at **139,648**, and `wmparc.mgz` at **482,292**. The independent spatial comparator has **18 failed, 28 blocked, and 6 missing** checks; topology mismatch blocks a valid paired-vertex metric on final surfaces.

The 33-column `synthseg.vol.csv` differs only slightly: 19 columns match exactly and the largest absolute difference is **0.04 mm³**. The eTIV column matches at **1,280,294.2 mm³**. This rounding-scale difference should be kept separate from the large cortical discrepancies below.

## First surface divergence and final geometry

In both hemispheres, `orig.nofix`, `smoothwm.nofix`, and `inflated.nofix` have the **same ordered faces and exactly equal coordinates** (left 102,764 vertices/205,560 faces; right 101,454/202,936). `qsphere.nofix` keeps the ordered faces but is the first divergent checkpoint among these compared surface files: paired displacement has mean **0.240 mm** (P99 1.060, maximum 3.369) on the left and mean **0.717 mm** (P99 3.057, maximum 10.014) on the right. Both runs used `mris_sphere -q -p 6 -a 128 -seed 1234` at this step, and the incoming mesh coordinates/faces matched; surface metadata was not independently proven identical. The discrepancy changes the topology repair input but does not alone explain every later failure.

| Final mesh | FreeSurfer vertices / faces | FNIT vertices / faces | Descriptive nearest-vertex distance, official → FNIT |
| --- | ---: | ---: | ---: |
| Left `orig` | 106,622 / 213,240 | 102,153 / 204,302 | mean 0.262 mm, P95 0.531 mm |
| Right `orig` | 105,541 / 211,078 | 101,079 / 202,154 | mean 0.267 mm, P95 0.536 mm |
| Left `white` | 106,622 / 213,240 | 102,153 / 204,302 | mean 0.843 mm, P95 2.242 mm |
| Right `white` | 105,541 / 211,078 | 101,079 / 202,154 | mean 0.842 mm, P95 2.281 mm |
| Left `pial` | 106,622 / 213,240 | 102,153 / 204,302 | mean 1.109 mm, P95 3.504 mm |
| Right `pial` | 105,541 / 211,078 | 101,079 / 202,154 | mean 1.134 mm, P95 3.513 mm |

These nearest-vertex distances measure geometric coverage only. They **are not matched-vertex position errors**: vertex counts and topology differ, so native vertex IDs cannot be paired. In the v3 source, `white.preaparc` and `white` are copies of `smoothwm`; `pial` is estimated by a normal ray through segmentation. The reference uses `mris_place_surface` geometry placement with intensity, WM, aseg, cortical labels, and atlas constraints. The archived v3 command runs `mris_fix_topology` with `-out orig` but omits the official `-ga -seed 1234`; the official run writes `orig.premesh` and then calls `mris_remesh --remesh --iters 3` to produce `orig`. These are separate topology-sequence differences in addition to the changed `qsphere.nofix` input. The native metric routines therefore measure different white/pial surfaces. `sphere.reg` and atlas assignment inherit these inputs.

The per-vertex files can only be compared as **distributions** until topology and vertex correspondence match:

| Vertex metric, file-wide mean | Official left → FNIT left | Official right → FNIT right |
| --- | ---: | ---: |
| `thickness`, mm | 2.079 → 2.940 | 2.024 → 2.956 |
| `area`, mm² | 0.706 → 0.525 | 0.711 → 0.530 |
| `area.pial`, mm² | 0.872 → 1.408 | 0.876 → 1.374 |
| `volume`, mm³ | 1.682 → 2.730 | 1.658 → 2.735 |
| `curv`, 1/mm | −0.0252 → −0.0518 | −0.0229 → −0.0475 |
| `curv.pial`, 1/mm | 0.0538 → −0.0641 | 0.0541 → −0.0042 |

These values are distribution means, not vertexwise MAEs. The paired final vertex error is **not measurable without correspondence**. Any apparent vertexwise error by raw array index would be invalid.

## Cortical ROI and subcortical metrics

All 34 named `aparc.stats` regions per hemisphere were compared. The table reports the mean absolute error across regions, followed by median absolute relative error. Gray matter ROI volume and the global `Cortex.CortexVol` header measure have different definitions; they must not be summed or substituted for one another.

| ROI field | Left MAE; median relative | Right MAE; median relative |
| --- | ---: | ---: |
| `ThickAvg` | 0.712 mm; 28.75% | 0.775 mm; 34.84% |
| `SurfArea` | 619.3 mm²; 28.41% | 618.3 mm²; 30.06% |
| `GrayVol` | 2,715.6 mm³; 48.43% | 2,766.1 mm³; 52.84% |
| `MeanCurv` | 0.05685 1/mm; 43.81% | 0.05794 1/mm; 42.93% |

Global `Cortex.MeanThickness` is **2.23998 → 2.85687 mm** left (+27.54%) and **2.18352 → 2.87176 mm** right (+31.52%). `Cortex.WhiteSurfArea` is **70,554.5 → 49,500.0 mm²** left (−29.84%) and **70,299.6 → 49,391.0 mm²** right (−29.74%). The repeated global `Cortex.CortexVol` measure is **353,949.84 → 496,248.73 mm³** (+40.20%) in both stats headers. The ROI `ThickAvg` maximum relative error reaches 65.69% left and 69.29% right; `GrayVol` reaches 138.54% and 131.28%, respectively.

The 45 named `aseg.stats` region volumes have MAE **507.65 mm³**. Among nonzero official regions, median absolute relative error is **0.170%**, with maximum **6.64%** for CSF (332,652.7 → 354,741.5 mm³). This is materially closer than the cortical ROI metrics, although `aseg.mgz` itself fails the exact voxel comparator.

## Where the errors arise

1. **Volumetric chain is mostly stable.** Nineteen MRI targets pass, including the filled WM and `aseg.presurf` used before surfaces. The final `aseg` and projected parcellations still differ; those errors need their own downstream checks.
2. **The earliest compared surface mismatch is `qsphere.nofix`.** Its 0.240/0.717 mm mean paired displacement changes the topology-repair input. Archived v3 omits the official topology `-ga -seed 1234` flags and `orig.premesh` → `mris_remesh --remesh --iters 3` sequence; its final `orig` has thousands fewer vertices. The source-built `mris_sphere` used the same printed q-sphere arguments, so the earlier q-sphere discrepancy needs its own frozen-input numerical/metadata test. Restore identical input and command sequence before treating any later vertex index as matched.
3. **Geometry placement is the dominant cortical metric gap.** v3 copies smoothed tessellation to `white` and traces pial along normals. This is consistent with the directional pattern of lower white area and higher thickness/gray volume; the isolated effects have not been apportioned. A separate **frozen-input** C++ test of source-built `mris_place_surface` found mean `white.preaparc` displacement **0.000421 mm** versus the installed binary, but its small floating differences amplified through a connected final white call to mean **0.0221 mm**. That isolated result is not the cause of the much larger v3 end-to-end white/pial gaps; v3 did not call the placement binary for those meshes.
4. **Registration, annotation, projection, and statistics inherit the geometry.** They cannot meet vertex/ROI parity before repaired topology, white/pial geometry, and `sphere.reg` agree. Later differences should be retested on frozen identical surfaces to separate their own implementations from upstream error.

This causal ordering is based on checkpoint equality, source inspection, and the frozen-input experiment. The sizes of downstream effects may involve interacting factors; the current single case cannot attribute each ROI error to one source line.

## Runtime and valid speed interpretation

| Run | `/usr/bin/time` process wall | Internal/launcher wall |
| --- | ---: | ---: |
| FNIT v3 candidate | 2,903.79 s (48:23.79) | 2,898.52 s across 39 top-level stages |
| Archived FreeSurfer 8.2 | 6,795 s (1:53:15) | 6,805.18 s launcher record |

The raw process-wall ratio is **2.34×** (official / FNIT) for these two runs. It **must not be called an equivalent-reconstruction speedup**: the candidate has missing files, divergent topology and cortical values, and its long white/pial placement steps are approximated. Runs were separate observations on a shared host, not a paired performance trial. Internal stage times are inclusive wall times; nested surface timings should not be added again.

| Candidate stage | Candidate seconds | Relevant archived official command timing |
| --- | ---: | --- |
| `SynthSeg` | 4.57 | `mri_synthseg` 192.32 s, one call; segmentation target passes, soft volumes differ ≤0.04 mm³ |
| `n4` + `nu` | 172.12 | `mri_nu_correct.mni` 225.30 s, different implementation/stage boundaries |
| `mri_em_register` | 251.25 | `mri_em_register` 251.75 s, one call |
| `mri_segment` | 77.37 | `mri_segment` 64.38 s, one call |
| `mri_fill` | 74.55 | `mri_fill` 92.91 s, one call |
| `surface_lh` / `surface_rh` | 494.76 / 814.67 | Official `mris_place_surface` 16 calls, 1,505.58 s combined; **not stage-equivalent** |
| `register_lh` / `register_rh` | 194.45 / 187.93 | Official `mris_register` two calls, 833.10 s combined; upstream spheres differ |

The v3 surface totals contain native topology repair (9.70 s left, 15.10 s right), `qsphere.nofix` (75.19/154.49 s), final sphere generation (333.24/560.87 s), and native thickness metrics (24.83/27.09 s). The right sphere stages dominate its surface time. The official per-command times are archived `@#@FSTIME` entries; wrappers can include children, so those rows must not be added to form an official total. Stage comparisons are descriptive until their inputs, output contracts, and boundaries match.

## Release and next verification gate

The end-to-end process **runs**, but it does **not** reproduce official recon-all cortical outputs. Keep the current v3 comparison labelled experimental. For a parity claim, first make both hemispheres' `qsphere.nofix` and final `orig` topology match on frozen inputs; then integrate and validate white/preaparc, final white, pial.T1 and pial placement with all official auxiliary inputs; then retest sphere registration, annotation, volume projection, every ROI row, and every vertex map. When ordered topology matches, require same-index vertex geometry and metrics within prespecified tolerances. Rerun the full 138-file comparator on the same real T1, then use paired timing trials for any speed claim.

Published evidence: [strict 138-file result](strict_138.json), [benchmark summary](benchmark_summary.json), [per-stage report](fnit-native-free-run.json), [process time](e2e.time), [source and input hashes](provenance.txt), and [analysis script](summarize.py). The full spatial comparator and official `/usr/bin/time` remain in the remote scratch directories named above. The [frozen-input white placement diagnosis](../WHITE_PREAPARC_FIRST_DIVERGENCE_20260927.md) records the isolated geometry tests and raw evidence paths.
