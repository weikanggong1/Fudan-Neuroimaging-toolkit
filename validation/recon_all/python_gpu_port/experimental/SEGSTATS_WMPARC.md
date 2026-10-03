# `mri_segstats` wmparc statistics: fs_sub01

This page preserves the original fixed-input experiment. Current production
API, CLI and the complete real-data warning regression are documented in
[segmentation statistics](../../../../docs/recon_all/SEGMENTATION_STATS.md).

## Fixed command and source

The actual `recon-all` call is:

```bash
mri_segstats --seed 1234 --seg mri/wmparc.mgz --sum stats/wmparc.stats \
  --pv mri/norm.mgz --excludeid 0 --brainmask mri/brainmask.mgz \
  --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR \
  --subject fs_sub01 --surf-wm-vol --ctab WMParcStatsLUT.txt \
  --etiv --stiv stats/synthseg.tiv.dat
```

Source is FreeSurfer 8.2.0 commit `d932c45b7941662ea380a05efef580568b98d41a`:
[`mri_segstats/mri_segstats.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_segstats/mri_segstats.cpp) SHA-256 `3347ee0f4e679f293e2ad3a38ece2479df12e6eba98cdb405261f6747e46623d`, [`utils/mri.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mri.cpp) SHA-256 `2d2d71d4e15d339f92c1eab4072761bb82dfe97d2adde76cef587bc983645c59`, [`utils/mri2.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mri2.cpp) SHA-256 `f6f01c07a2e44c127eb1b22bca61d0e6e87a6065f538fbafc8de343bcec84ed4`, and [`utils/stats.cpp`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/stats.cpp) SHA-256 `0627fa0ddb0ca6c7f9c11433bc2b4e372de9a10460a4a4d72ca3d26489798989`.

`segstats_wmparc_python.py` implements the fixed branch with NumPy, Numba and nibabel. For each border voxel it reproduces FreeSurfer's 6-face border, 3×3×3 candidate labels, clamped 15×15×15 intensity neighborhood, ascending-label tie rule, and float32 PV accumulation. Unlike the native routine's one full-volume scan per segmentation, it visits each voxel once and updates all affected labels in the same spatial order. It also reproduces the source's float32 intermediate in sample standard deviation. It reads the cached 16-element `brainvol.stats`, the Talairach XFM for eTIV, the SynthSeg sTIV scalar, and the small WM LUT. The fixed `--brainmask` only triggers a `MaskVol` summary from the cached brain-volume statistics in this command. The Python production path does not execute or read a FreeSurfer binary/runtime package. This stage computes on CPU and is called by the current recon-all entry point.

## Frozen inputs

All subject paths are under `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_main_20260924/single_subjects/fs_sub01`.

| Input | SHA-256 |
| --- | --- |
| `mri/wmparc.mgz` | `fc764c8396989984c550319240a96a06134e4a7708579a8dfae1321a9e77da2c` |
| `mri/norm.mgz` | `e8875a30cb63dcdfdbda6a830fa4a8a30141e1f0da3058d33b491253abe74390` |
| `mri/brainmask.mgz` | `6d31b57e4f13006aab34a0f667643da477a63d3d8b514f0c4e1a0e7215141539` |
| `mri/transforms/talairach.xfm` | `1e509f0e544614555ef77ba4df6c8f6ebfa4c73c4771d7464589250dfc607e59` |
| `stats/brainvol.stats` | `42abea3af25d90be3b1815d1c07ed65472984f152989078cbb2501a711336627` |
| `stats/synthseg.tiv.dat` | `b78b3e547b49550024c49f39eae9ccd483d235098560b53a0f2cc50f2a0aede6` |
| `WMParcStatsLUT.txt` | `1eb362b9d929797179106ba37eecaa8eff2830f1509be8e7b56238e6ac7c6dbf` |

The isolated results are under `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_python_gpu_20260925/wmparc_stats_ref`. A fresh native headcw rerun produced `wmparc.native.stats` with **0 differences in 70 rows, 700 row fields and 7 Measure lines** against the original recon-all `stats/wmparc.stats` (original SHA-256 `d80766f123223a6ac078edac2cf7a7b1b7efe15b7ed59378cadf4bb8c4433eb3`). The Python `wmparc.python.stats` also has **0 differences** in those fields and in all TableCol/NRows/NTableCols/ColHeaders schema tokens against both native outputs. Thus the tolerance is exact at the statistics file's printed precision: 0.1 mm³ for PV volume, 0.0001 for norm mean/std/min/max/range, and 0.000001 for summary volumes. The Python file SHA-256 is `6b9f00b68e134fd8edb733595f8b7a5098ad7bf594d3ad6604db6ff1b32c1136`; complete-file bytes differ because the Python header identifies its own generating program and has different path, host and timestamp provenance.

On the same headcw host with the same inputs, the fresh native rerun took **100.34 s**. The final Python step took **10.14 s** immediately after its last code change and **9.35 s** on a repeat with a warm Numba cache. These are complete CLI wall times, not a multi-subject benchmark. The repeated output retained 0 numerical differences. Two synthetic unit tests passed. These timings only cover this wmparc statistics step, not recon-all or GPU acceleration.

To reproduce the fixed-input check:

```bash
BASE=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work
SUBJ=$BASE/reconall_main_20260924/single_subjects/fs_sub01
LUT=$BASE/reconall_main_20260924/bundle/WMParcStatsLUT.txt
REF=$BASE/reconall_python_gpu_20260925/wmparc_stats_ref
PYTHONPATH=src python -m fnit.recon_all.segstats_wmparc_python \
  "$SUBJ" "$LUT" "$REF/wmparc.python.stats"
python validation/recon_all/python_gpu_port/experimental/check_segstats_wmparc.py \
  "$SUBJ/stats/wmparc.stats" "$REF/wmparc.python.stats"
```

The exact-field gate covers this one fixed FreeSurfer 8.2.0 subject and command. Other `mri_segstats` branches and other subjects have not yet been validated.
