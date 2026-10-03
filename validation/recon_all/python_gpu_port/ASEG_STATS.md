# Fixed `mri_segstats` aseg statistics

This page preserves the 2026-09-28 same-input measurement. The production
runner now calls this function; current Chinese API, CLI, inputs, outputs and
the real single-voxel warning regression are documented in
[segmentation statistics](../../../docs/recon_all/SEGMENTATION_STATS.md).

`segstats_aseg_python.py` replays the fixed FreeSurfer 8.2 command using
`aseg.mgz`, `norm.mgz`, `brainvol.stats`, the original uncorrected bilateral
surfaces, Talairach XFM, SynthSeg sTIV scalar, and the downloaded
`ASegStatsLUT.txt`. It shares the independently verified partial-volume and
intensity-statistics calculation with the Python `wmparc.stats` implementation.
It keeps the 12 empty atlas rows required by `--empty`, excludes cortical
gray/white labels, and computes `lh/rhSurfaceHoles` from each uncorrected
surface's Euler characteristic. This stage runs on CPU without a native
FreeSurfer executable.

For `fs_sub01`, the Python output, the original official `aseg.stats`, and a
new official replay matched at **45/45 table rows, 450/450 table fields,
21/21 Measure lines, and every table-schema field**. There were zero printed
numeric differences: partial-volume volume is printed to 0.1 mm³, intensity
fields to 0.0001, and global volume measures to 0.000001. The measured
uncorrected-surface Euler values were -16/-14, giving 9/8 holes as in the
official output. File headers that identify the command, host, time, and
paths differ by design.

| Same-input headcw CLI | Time |
| --- | ---: |
| Fresh FreeSurfer 8.2 `mri_segstats` | 22.66 s |
| Python/Numba CPU | 10.34 s |

This is one same-host step comparison. The Python Numba cache was available.
It does not measure the complete reconstruction. The frozen input hashes were
`aseg.mgz` `87b396428a367b6ca8bce624e80de5ca204ff3febdb87b1109e938e19512f48e`,
`lh.orig.nofix` `ef00440f651026e4d099bd282f3e4e2cfbfffdc1e0877684fe22aa2948542b79`,
`rh.orig.nofix` `45d53fa51554b6f58f04ac37ee8eff1814fd31c001178e7648af2eb97b9f01cf`,
and the atlas LUT `36eb822e91174a7a4f99f22f17f19e6692225476b815aa0fdaf0622fd32e2511`.
Other common input hashes are listed in the [wmparc stats
report](experimental/SEGSTATS_WMPARC.md).

```bash
python -m fnit.recon_all.segstats_aseg_python \
  /path/to/subjects/fs_sub01 /path/to/assets/ASegStatsLUT.txt \
  /path/to/subjects/fs_sub01/stats/aseg.stats
```

This historical isolated validation consumes official upstream surfaces and
segmentation. Its same-input fields and timings remain distinct from the
current raw-input whole-subject measurements.
