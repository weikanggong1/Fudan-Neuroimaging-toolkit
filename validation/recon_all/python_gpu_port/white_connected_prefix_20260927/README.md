# Connected candidate prefix through LH smoothwm

This is a **saved-stage connection test**, not a fresh one-process
T1-to-recon-all run. The real input scan is
`examples/data/sub-01_T1w.nii.gz` (SHA-256
`f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`).
The reference is the completed FreeSurfer 8.2 `a_official` subject on gpucw1.
No official auxiliary label or surface is passed to placement. The saved
patched topology GA output in this test **did use official `brain.mgz` and
`wm.mgz`**; its scratch subject also symlinked official `filled.mgz` and
`norm.mgz`. Corresponding v5 candidate MRI volumes were separately measured
voxel-exact, but this replay has an official-MRI topology input boundary.

## Candidate chain and file boundaries

The saved v5 candidate MRI inputs `orig`, `nu`, `synthseg.rca`, `brain`,
`brainmask`, `entowm`, and `aseg.presurf` each have zero voxel differences
against the official subject ([upstream comparison](../mni_aux_connected_20260927/input_comparison.json)).
The [CPU MNI/auxiliary run](../mni_aux_connected_20260927/README.md) uses
those images and external weights/priors to generate its own MNI152 LTA,
`mca-dura.mgz`, `vsinus.mgz`, and `brain.finalsurfs.mgz`. The [initial
surface chain](../../../../src/fnit/recon_all/initial_surface_chain.py)
starts from candidate `filled` and `norm` and uses Python pretess,
tessellation, smoothing, **Python inflate**, and Python quick sphere. The
[patched Conda topology GA](../../../../docs/recon_all/TOPOLOGY_CONDA_GA.md)
and Python remesh then generate LH `orig.premesh` and `orig` using the
scratch topology subject's official `brain` and `wm`. They are
linked into an isolated subject with the candidate MRI outputs. The new
Conda `mris_place_surface --white` wrapper produces `lh.white.preaparc`;
Python `smooth_surface(..., iterations=3, device="cpu")` consumes that
placed surface to produce `lh.smoothwm`.

The stages were run at different times and their saved candidate outputs
were linked for this test. The [raw paired report](lh_report.json) records
the resolved candidate and official paths and full SHA-256 hashes for
`wm`, `aseg.presurf`, both auxiliary labels, `brain.finalsurfs`, `orig`,
`orig.premesh`, `white.preaparc`, `smoothwm`, the diagnostic volume, and
the threshold statistics. The source-to-target link and the four official topology MRI symlinks are
auditable in the report. This mixed-input check does **not** establish a
fully candidate-derived topology chain, fresh production CLI completion,
or RH connected placement. The prior [RH candidate placement test](../../../../docs/recon_all/WHITE_PREAPARC_CONDA_CHAIN.md)
used borrowed official auxiliary labels and has a different upstream
boundary. Corrected Python inflate and quick sphere were separately exact
on this T1's frozen inputs ([inflation](../INFLATE_STATUS.md),
[quick sphere](../SPHERE_QUICK_STATUS.md)); this is not a guarantee for other
subjects.

## Same-T1 accuracy

| Candidate output | Comparison with saved FreeSurfer 8.2 output |
| --- | ---: |
| `mca-dura`, `vsinus`, `brain.finalsurfs` | Each 0 / 16,777,216 differing voxels; affine equal |
| LH `orig.premesh`, conditional on official topology MRI inputs | 101,689 ordered vertices and 203,374 faces exact |
| LH `orig`, same conditional input | 106,622 ordered vertices and 213,240 faces exact |
| LH `white.preaparc` | 106,622 faces ordered exact; mean/P99/max 3D vertex displacement **0.000420574 / 0.006998961 / 0.630004 mm**; **50** vertices >0.1 mm |
| Placement diagnostic `mrisps.wpa` | 0 / 16,777,216 differing voxels; affine equal |
| Threshold statistics | Text byte-identical |
| LH `smoothwm`, 3 CPU passes | 106,622 faces ordered exact; mean/P99/max 3D vertex displacement **0.000311208 / 0.006172390 / 0.188170 mm**; **18** vertices >0.1 mm |

Distances are Euclidean between corresponding vertices, not absolute
coordinate-component errors. The compressed volume/surface hashes need not
match when geometry and voxels do: metadata and provenance differ. The
nonzero vertex tails remain a precision gap for the downstream white/pial
and vertex-map stages. No downstream regional thickness, area, volume,
curvature, atlas, or complete 138-output acceptance is inferred here.

## Observed time and official context

| Stage | Candidate CPU wall, s | Archived official stage, s |
| --- | ---: | ---: |
| MNI152 crop + affine | 15.63 | crop 1.86 + SynthMorph 107.57 |
| MCA/dura segmentation | 6.49 | 118 |
| Venous-sinus segmentation | 12.09 | 55 |
| Five finalsurfs edits | 8.58 | 8.21 |
| LH initial Python inflate / quick sphere | 27.57 / 56.29 | See [stage reports](../INFLATE_STATUS.md) and [sphere report](../SPHERE_QUICK_STATUS.md) |
| LH Conda topology GA and Python remesh | See [topology report](../../../../docs/recon_all/TOPOLOGY_CONDA_GA.md) | See same report |
| LH gray/white thresholds | 4.41 | Archived 3.98 |
| LH Conda white pre-aparc placement | 261.51 | Archived 242.65 |
| LH final Python smoothwm | 7.22 | Paired official-input median 3.153 |

The MNI/aux/finalsurfs sum was **42.79 s**; its full earlier probe,
including imports and volume comparisons, took 57.25 s. The threshold,
placement and final smoothing sum in this separate saved-stage run was
**273.14 s**. Load, inputs, process boundaries and dates differ from the
archived official timings; these are stage observations, not an equivalent
end-to-end speed comparison. The [same-input final smoothing benchmark](../../../../docs/recon_all/SMOOTHWM_FINAL_PARITY.md)
found exact output from an official `white.preaparc` input and is separate
from the candidate-connected errors above.

## Reproduce the comparison

Use the FNIT Conda Python with nibabel and NumPy, the saved candidate
subjects and official reference, then run:

```bash
python compare.py CANDIDATE_PLACEMENT_SUBJECT OFFICIAL_SUBJECT \
  CANDIDATE_AUX_SUBJECT CANDIDATE_AUX_REPORT INITIAL_SURFACE_REPORT \
  PLACEMENT_TIME_JSON lh_report.json
```

The command compares the saved outputs; it does not rerun image inference,
topology repair or placement. Source input/output paths and hashes needed to
repeat those stage calls are in the JSON report and the linked stage reports.
