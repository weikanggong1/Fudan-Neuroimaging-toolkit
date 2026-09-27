# Real-T1 saved-stage prefix through LH smoothwm

This is a **connected replay from saved candidate stages**, not a fresh
single-process T1-to-recon-all run. The deidentified input is
`examples/data/sub-01_T1w.nii.gz` (SHA-256
`f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`);
the comparison subject is completed FreeSurfer 8.2 `a_official` on gpucw1.
The new **fully candidate-input LH replay** passes no official image,
label, or surface into reconstruction operators. The official subject is
read by the probes only for paired comparison. The older mixed-input
replay is retained below with its distinct provenance.

## Fully candidate-input LH chain

The saved v5 candidate `orig`, `nu`, `synthseg.rca`, `brain`, `brainmask`,
`entowm`, and `aseg.presurf` MRI outputs were each voxel-exact against
official ([seven-input comparison](../mni_aux_connected_20260927/input_comparison.json)).
The [CPU MNI152/auxiliary run](../mni_aux_connected_20260927/README.md)
used those candidate files plus external weights/priors to generate
its own LTA, `mca-dura`, `vsinus`, and `brain.finalsurfs`. The initial
surface files `orig.nofix`, `inflated.nofix`, and `qsphere.nofix` came from
FNIT's Python [initial surface chain](../../../../src/fnit/recon_all/initial_surface_chain.py)
on separately saved candidate `filled`/`norm` from this T1; they were all
ordered-coordinate/face exact against official before topology repair.

The new [candidate topology probe](probe_candidate_topology.py) links the
v5 candidate **`brain`, `wm`, `filled`, and `norm`** into a new scratch
subject. It refuses any of their resolved paths under the official
subject. All four have 0/16,777,216 differing voxels and affine difference
0 against their reference counterparts. The probe runs Python centered
sphere, patched Conda `mris_fix_topology_fnit`, Python three-iteration
remesh, and Python intersection check. Its [input and output report](candidate_topology_lh_report.json)
records realpaths, SHA-256, elapsed times, and ordered vertex/face
comparisons. `orig.premesh` and `orig` were exact. In this scratch subject,
`wm`, `aseg.presurf`, `brain.finalsurfs`, `mca-dura`, and `vsinus` are
candidate outputs as well.

The [placement probe](probe_candidate_place.py) then runs Python gray/white
thresholds and Conda `mris_place_surface --white` on that new `orig`,
followed by Python three-pass CPU smoothing. The [full LH paired report](candidate_full_lh_report.json)
records each input/output realpath and SHA-256, per-surface ordered
coordinates/faces, per-volume voxels/affines, threshold text equality,
and stage times. It also compares the current runner's proposed final
`white` (a copy of `smoothwm`) with official final `white`. The stages
were executed in separate processes and linked by saved files. RH has
separate segmental placement evidence, but this fully candidate-input
connected replay was LH only.

| Output on the new candidate subject | Difference against saved official |
| --- | ---: |
| `mca-dura`, `vsinus`, `brain.finalsurfs` | Each 0 / 16,777,216 differing voxels; affine equal |
| LH `orig.premesh` | 101,689 ordered vertices, 203,374 faces, all coordinates exact |
| LH `orig` | 106,622 ordered vertices, 213,240 faces, all coordinates exact; zero intersecting faces |
| LH `white.preaparc` | Ordered faces exact; mean / P99 / max **0.000420574 / 0.006998961 / 0.630004 mm**; **50** vertices >0.1 mm |
| `mrisps.wpa` diagnostic | 0 / 16,777,216 differing voxels; affine equal |
| Gray/white threshold statistics | Text byte-identical |
| LH `smoothwm`, three CPU passes | Ordered faces exact; mean / P99 / max **0.000311208 / 0.006172390 / 0.188170 mm**; **18** vertices >0.1 mm |
| Current runner's `white = smoothwm` versus official final `white` | Mean / P99 / max **0.288067 / 1.263745 / 3.322477 mm**; **89,203** vertices >0.1 mm |

All distances are 3D Euclidean distances at corresponding vertex indices.
The last row measures the **next unresolved final white placement stage**;
it does not reflect an additional placement command. The nonzero
`white.preaparc`/`smoothwm` tails are a smaller precision gap, still
relevant to vertex-level acceptance. Byte hashes of compressed MGZ and
surface files can differ despite exact voxel or ordered geometry because
metadata/provenance differs. Final pial geometry, thickness, area, volume,
curvature, atlas, ROI statistics, and complete 138-output acceptance remain
unverified for this connected prefix.

## Times and official context

| Stage | Candidate saved-run wall, s | Archived official context |
| --- | ---: | ---: |
| MNI152 crop + affine, MCA/dura, vsinus, five finalsurfs edits | 15.63, 6.49, 12.09, 8.58 | 1.86 + 107.57, 118, 55, 8.21 s respectively; [MNI/aux report](../mni_aux_connected_20260927/README.md) |
| LH initial Python pretess, tessellation, main component, smoothing, inflate, quick sphere | 3.64, 0.37, 0.08, 3.90, 27.57, 56.29 | Separate [inflate](../INFLATE_STATUS.md) and [sphere](../SPHERE_QUICK_STATUS.md) benchmarks |
| LH centered sphere / Conda topology GA / Python remesh / intersection check | 9.94 / 104.77 / 181.66 / 1.45 | Separate [topology benchmark](../../../../docs/recon_all/TOPOLOGY_CONDA_GA.md) |
| LH thresholds / Conda pre-aparc placement / Python final smoothwm | 8.24 / 237.16 / 5.86 | 3.98 / 242.65 archived for first two; 3.153 s official-input smoothing median in [separate paired benchmark](../../../../docs/recon_all/SMOOTHWM_FINAL_PARITY.md) |

These are one-run observations on a shared node, with separate process
boundaries and differing load. The official measurements were archived
on other runs. They do not give an end-to-end time or a controlled speed
ratio. The MNI/aux/finalsurfs stage sum was 42.79 s in its earlier probe;
that probe's full wall, including repeated 256³ comparisons, was 57.25 s.

## Earlier mixed-input LH replay

The [earlier comparison](lh_report.json) linked the same candidate
auxiliary labels and finalsurfs to an `orig` generated by the same FNIT
Python/Conda topology algorithms, **but its topology scratch subject used
official `brain` and `wm` MRI files**. Its `filled` and `norm` symlinks
also pointed to the official subject, even though the independent initial
Python surface chain read candidate `filled`/`norm`. This test therefore
could not establish a fully candidate-input topology chain. Its placed
`white.preaparc` and smoothed `smoothwm` had the same geometry errors as
the subsequent fully candidate replay. The older [bilateral placement
report](../../../../docs/recon_all/WHITE_PREAPARC_CONDA_CHAIN.md) also
borrowed official MCA/dura and venous-sinus labels, so its RH result has a
separate upstream boundary. Both trial types are kept to show precisely
which inputs were checked.

## Reproduce the checks

Run the [candidate topology probe](probe_candidate_topology.py) with an
empty scratch subject, v5 MRI directory, candidate auxiliary subject,
saved FNIT initial surface subject, official comparison subject, patched
Conda `mris_fix_topology_fnit`, external data assets, and output JSON.
It writes the candidate MRI manifest and stops before placement if the
first topology surface differs. Then run [candidate placement](probe_candidate_place.py)
with that subject, Conda `mris_place_surface`, assets, and timing JSON.
Finally use the saved-output comparator:

```bash
python compare.py CANDIDATE_SUBJECT OFFICIAL_SUBJECT CANDIDATE_AUX_SUBJECT \
  CANDIDATE_AUX_REPORT INITIAL_SURFACE_REPORT PLACEMENT_TIME_JSON \
  candidate_full_lh_report.json
```

The probe scripts require the FNIT Conda Python, nibabel, NumPy, the
pinned Conda binaries and data-only assets; `FS_LICENSE` points to a
private external license when required by those binaries. No license,
patient image, binary, or model weight is committed.
