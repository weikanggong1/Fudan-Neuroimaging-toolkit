# v3 final `aseg.mgz`: first postprocessing divergence

This is an isolated CPU diagnosis on the real `sub-01_T1w.nii.gz` (SHA-256
`f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`).
The frozen v3 subject and FreeSurfer 8.2 subject are **read-only**. The script
writes only to a fresh scratch directory. It runs with `CUDA_VISIBLE_DEVICES=`;
there is no GPU timing or full `recon-all` rerun.

## First divergence

The archived official `recon-all.log` runs `mris_volmask` to build the ribbon,
then `mri_relabel_hypointensities` to build `aseg.presurf.hypos.mgz`, then
`mri_surf2volseg --fix-presurf-with-ribbon` to build the final `aseg.mgz`.
Archived v3 performs none of these final-volume stages: `_segment_callosum`
copies `aseg.auto.mgz` to both `aseg.presurf.mgz` and `aseg.mgz`. Thus the
**first missing postprocessing output is the ribbon**, after the already
divergent v3 white/pial surfaces. `aseg.presurf.mgz` itself matches official
for all 16,777,216 voxels, with the same float32 MGH header and affine.

On v3's fixed surfaces, the existing Python ports generated:

| Output | Unequal voxels vs official | Stored dtype vs official | MGH header exact |
| --- | ---: | --- | --- |
| `ribbon.mgz` | 197,941 | uint8 / uint8 | yes |
| `aseg.presurf.hypos.mgz` | 195 | float32 / float32 | yes |
| final `aseg.mgz` | **144,469** | int32 / int32 | yes |

The official hypointensity step changes exactly one `aseg.presurf` voxel
(`3 -> 77` at `[146,126,184]`); the v3 surfaces make the Python/native step
change 194 (`42 -> 77`: 136; `3 -> 77`: 58). The ribbon disagreement is much
larger and includes 39,483 candidate cortex voxels where the official ribbon
is zero. These are upstream geometry differences, not evidence that the
Python relabeling or final ribbon-fix rules are wrong.

The archived v3 `aseg.mgz` is a float32 copy of `aseg.presurf.mgz` and differs
from the official int32 final file at **104,986** voxels. On identical v3
inputs, the generated Python hypointensity volume matched a fresh installed
`mri_relabel_hypointensities` output at **0** unequal voxels, with exact dtype,
header, and affine. The generated int32 final volume matched a fresh installed
`mri_surf2volseg --fix-presurf-with-ribbon` output at **0** unequal voxels,
again with exact dtype, header, and affine. This native control consumed the
Python-generated ribbon, so it validates the final fix operation on identical
inputs, while the prior frozen-official-input test validates the Python
ribbon operation. A fresh native `mris_volmask` on v3 surfaces was not run.

Applying this otherwise exact chain to v3 changes 123,461 candidate voxels:
40,944 previous errors become correct, 80,427 previously correct voxels
become wrong, and 2,090 wrong voxels change to another wrong label. The net
official mismatch rises by **39,483** to **144,469**. The production-safe
decision is to **leave v3 final-aseg generation unchanged** until white/pial
surfaces and cortex labels meet their own official-input gates. Then rerun
these three stages and this same-T1 check; do not infer final parity from
same-input operation parity alone.

## CPU timing and provenance

| Stage | Measured seconds | Boundary |
| --- | ---: | --- |
| Python ribbon | 8.29 | In-process Python/Numba, v3 inputs |
| Python relabel | 5.09 | In-process Python/SciPy/PyTorch CPU, v3 inputs |
| Installed native relabel | 11.25 | Fresh subprocess, same v3 inputs |
| Python final fix | 5.16 | In-process Python/SciPy, v3 inputs |
| Python int32 MGH save | 0.45 | Separate file write |
| Installed native final fix | 3.81 | Fresh subprocess, same Python hypos/ribbon and v3 surfaces/labels |
| Archived official ribbon / relabel / fix | 244.68 / 27.13 / 10.32 | Original official run, different time/load |

The in-process Python and fresh-process native times are observational and
not a paired speed benchmark. The official timings are from the archived
`@#@FSTIME` log. Input and implementation SHA-256 values, per-file voxel,
dtype, header, affine results, first unequal voxel, and unchanged-source
checks are in [report.json](report.json). The frozen candidate inputs remained
SHA-256 unchanged. Native startup initially required `FREESURFER_HOME` and
`SUBJECTS_DIR`; the probe resumed from checksum-verified Python outputs rather
than recomputing them. The final report status is `complete`.

Reproduce the isolated check with [probe.py](probe.py) in the archived v3
Python environment, `PYTHONPATH` pointing to v3 source, the installed
FreeSurfer binaries, and fresh `--out`. The actual reference roots were
`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official`
and
`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_wm_agent_20260927/v3_e2e/subjects/sub01`.
Remote scratch output is at
`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_wm_agent_20260927/aseg_v3_postprocess_20260927/run1`.

This establishes only the v3 final-volume branch boundary on one T1. It does
not establish final cortical surface, vertex metric, ROI, or end-to-end parity.
