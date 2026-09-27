# LH cortical labels from a saved candidate white pre-aparc surface

## Inputs, outputs and order

This is a saved-stage check on the same deidentified T1 as the
[fully candidate LH surface prefix](../white_connected_prefix_20260927/README.md),
not a fresh one-process `recon-all`. The candidate subject on gpucw1 is
`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_finalsurfs_stage_20260927/white_candidate_no_official_topology_fslicense`.
The tested reconstruction inputs are its FNIT/Conda-generated
`surf/lh.white.preaparc`, v5-generated `mri/aseg.presurf.mgz`, and
v5-generated `mri/entowm.mgz`. All resolved paths and SHA-256 hashes are
in [lh_report.json](lh_report.json); the two MRI inputs each have 0/16,777,216
voxel differences and affine difference 0 from official. No official
label, surface, or volume is passed into either label generator.

In official recon-all order, the existing
`label_cortex_fix_ga(surface, aseg, entowm, "lh", output)` writes
`label/lh.cortex.label` and returns ordered base and appended gyrus-ambiens
vertex IDs. Then `label_cortex(surface, aseg, output,
keep_hip_amyg=True)` writes `label/lh.cortex+hipamyg.label` and returns
its ordered selected IDs. Both functions use NumPy/SciPy on CPU. The first
label is needed by final white placement and later segmentation; the second
is needed by pial placement. The latter is not an input to the first.
The [probe](probe.py) runs both functions, then reads FreeSurfer 8.2
reference labels only for comparison.

Equivalent official commands from the same subject's
`scripts/recon-all.log`, with the latter executed from `mri/`, are:

```bash
label-cortex --s a_official --lh --fix-ga
mri_label2label --label-cortex ../surf/lh.white.preaparc \
  aseg.presurf.mgz 1 ../label/lh.cortex+hipamyg.label
```

Standalone Python commands corresponding to the two outputs are:

```bash
python -m fnit.recon_all.label_cortex_fix_ga_python \
  surf/lh.white.preaparc mri/aseg.presurf.mgz mri/entowm.mgz lh \
  label/lh.cortex.label
python -m fnit.recon_all.label_cortex_python \
  surf/lh.white.preaparc mri/aseg.presurf.mgz \
  label/lh.cortex+hipamyg.label --keep-hip-amyg
```

## Paired real-T1 result

| Candidate output | Rows | Ordered vertex ID / membership | Bytes and coordinate text |
| --- | ---: | --- | --- |
| `lh.cortex.label` | 99,048 = 98,883 base + 165 GA | All ordered IDs exact; vertex-set difference 0; 98,937 unique IDs in both | Header/count and ID/stat fields exact; 7,136 XYZ rows differ |
| `lh.cortex+hipamyg.label` | 101,000 | All ordered IDs exact; vertex-set difference 0 | Header/count and ID/stat fields exact; 7,284 XYZ rows differ |

The duplicate vertex IDs in `cortex.label` are deliberate native
GA-label concatenation and occur in the same order in the candidate and
official files. The output files are **not byte-identical** because label
rows include surface XYZ coordinates rounded to three decimals. Every
textual mismatch is in those XYZ fields; none changes vertex ID or stat.
The XYZ text's mean/P99/max 3D difference is
`0.000400/0.007000/0.575074 mm` for `cortex.label` and
`0.000411/0.007141/0.575074 mm` for `cortex+hipamyg.label`. This follows
the measured nonzero coordinate tail in candidate `white.preaparc`
(50 vertices above 0.1 mm), despite exact membership for this subject.
The first different data row is index 13 in both files. The raw report
includes the candidate/reference label hashes, row counts, unique counts,
first row difference, and timing.

The candidate CPU wall times on gpucw1 were **6.14 s** for fixed GA
cortex and **4.34 s** for the hippo/amygdala-preserving label. A separate
frozen-official-input run on headcw measured 3.89/3.46 s for Python and
10.73/7.72 s for native child programs; see the
[isolated stage benchmark](../LABEL_CORTEX_FIX_GA_STATUS.md). Different
inputs, hosts and load prevent a paired speed ratio. The label IDs are
accepted for this one LH saved-stage chain, while byte equality remains
blocked by surface coordinates. RH connected candidate labels and the
later final white/pial geometry are not established here.

## Reproduce

With the FNIT Conda Python and repository source available:

```bash
PYTHONPATH=src python probe.py CANDIDATE_SUBJECT \
  V5_CANDIDATE_ENTOWM OFFICIAL_SUBJECT lh_report.json
```

The probe creates `mri/entowm.mgz` as a symlink to the specified candidate
file if needed, then writes both labels under the candidate `label/`
directory. It refuses existing outputs. `--compare-existing` rechecks
already generated labels and preserves the original measured times without
rerunning the generators. The official subject is used only for the
read-only comparisons after candidate generation.
