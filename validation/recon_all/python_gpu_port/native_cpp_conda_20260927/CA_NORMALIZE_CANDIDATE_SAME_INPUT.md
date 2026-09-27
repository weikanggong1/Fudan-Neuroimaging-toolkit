# CA intensity normalization: current candidate, frozen same inputs

## Function and data

`run_ca_normalize` estimates GCA guided control points and a smooth bias field, then writes the conformed uint8 `norm.mgz` and six frame float32 `ctrl_pts.mgz`. It reads `nu.mgz`, `brainmask.mgz`, the 2020 GCA, and a voxel to voxel `talairach.lta`. The Python implementation is [`ca_normalize_python.py`](../../../../src/fnit/recon_all/ca_normalize_python.py). No new external data, C++ binary, FreeSurfer installation, or GPU memory is required for this stage. The GCA already belongs to the verified external assets.

Python API:

```python
from fnit.recon_all.ca_normalize_python import run_ca_normalize

timing = run_ca_normalize(
    "mri/nu.mgz", "mri/brainmask.mgz",
    "assets/average/RB_all_2020-01-02.gca",
    "mri/transforms/talairach.lta",
    "mri/norm.mgz", "mri/ctrl_pts.mgz",
)
```

Equivalent FreeSurfer 8.2 command, run with the same four frozen input files:

```bash
mri_ca_normalize -c ctrl_pts.mgz -mask brainmask.mgz \
  nu.mgz average/RB_all_2020-01-02.gca \
  transforms/talairach.lta norm.mgz
```

The command is the default command in the official same T1 `recon-all.log`. Only this reference command used the installed FreeSurfer executable. The paired Python call used the independent Conda environment; the candidate subject and active end to end run were not modified.

## Accuracy and elapsed time

The current candidate's `nu.mgz`, `brainmask.mgz`, and `talairach.lta` were copied to a separate validation directory. The external GCA had the same SHA256 as the official installed GCA. The independent official command and Python API read those identical frozen files.

| Output | Header bytes | Different values | Maximum absolute error |
| --- | --- | ---: | ---: |
| `norm.mgz` (256³ uint8) | 284/284 equal | 0 / 16,777,216 | 0 |
| `ctrl_pts.mgz` (256³ × 6 float32) | 284/284 equal | 0 / 100,663,296 | 0 |

The candidate end to end run's existing Python outputs also matched this fresh official run at every voxel/value and in the 284 byte MGH header. Compressed file hashes differed, so complete byte identity was not established. In this paired gpucw1 run, `/usr/bin/time` recorded **91.44 s** for the official executable and **35.47 s** for the Python process including startup; the candidate integrated Python stage recorded **32.78 s**. Node load varied, so these single timings are descriptive rather than a throughput speedup claim. The earlier independent frozen input check likewise found zero output differences and recorded Python 17.49 s; see [`CA_NORMALIZE.md`](../CA_NORMALIZE.md).

## Source of the complete subject difference

Compared with the prior official complete subject, the candidate had `nu.mgz` identical in all 16,777,216 voxels, but `brainmask.mgz` differed at 38 voxels and the 4×4 `talairach.lta` differed in 9 elements (maximum absolute difference 0.0002322644). Their final `norm.mgz` volumes differed at 494,932 voxels and `ctrl_pts.mgz` at 6,273 values. Since the same input reference command reproduces the candidate Python output exactly, these complete subject differences arise before this normalization call. The 38 mask voxels could alter the GCA registration and therefore the LTA; that causal link is an inference, not a tested mask intervention.

The exact same input result and lower observed Python time do not justify adding a compiled `mri_ca_normalize` target to the Conda runtime. The input hashes and full numerical comparison are in [`ca_normalize_same_input.json`](ca_normalize_same_input.json).
