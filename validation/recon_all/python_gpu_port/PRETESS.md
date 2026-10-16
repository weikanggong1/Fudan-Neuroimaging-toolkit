# Pretess topology repair

`pretess_python.py` translates the three fixed FreeSurfer 8.2.0
`mri_pretess` calls in the subject's recon-all trace. It follows the source
order of six edge passes, four foreground corner passes, and four background
corner passes; edits made early in a pass affect later checks. The `wm` mode
preserves existing white-matter values and writes 215 for newly added voxels.
This stage currently runs on NumPy/CPU. Source reference:
`mri_mc/mri_pretess.cpp` at
`d932c45b7941662ea380a05efef580568b98d41a`.

| Fixed call | Native median | Python median | Topology edits | Voxel differences |
| --- | ---: | ---: | ---: | ---: |
| `wm.asegedit.mgz wm norm.mgz` | 1.389 s | 3.542 s | 336 | 0 / 16,777,216 |
| `filled.mgz 255 norm.mgz` | 0.999 s | 2.822 s | 18 | 0 / 16,777,216 |
| `filled.mgz 127 norm.mgz` | 0.745 s | 1.896 s | 9 | 0 / 16,777,216 |

The [three-trial report](pretess_cpu_report.json) uses the same frozen input
files on `control-host`, alternates native and resident Python runs, and records
their SHA-256 values. The decompressed MGH header and all voxel bytes match
for every paired run, with identical affines. The complete files differ in
FreeSurfer command-history metadata; the Python output preserves the input
tags and does not forge a native command string. The two small
[behavior tests](test_pretess_python.py) check brightness selection, equal
intensity tie behavior, and the white-matter fill value.

The two Python-produced `filled-pretess` volumes were fed directly to
`tessellate_gpu.py`; each produced the same vertex and quad core bytes as a
fresh native `mri_tessellate` run. This checks the adjacent stage boundary
without running another full reconstruction. The longer bilateral
[six-stage](SMOOTH_SURFACE.md) replay now starts from frozen `filled.mgz` and
`norm.mgz` and matches the official quick spheres; upstream white-matter
generation and later topology/surface stages remain separate gates.
