# Fixed T1 `mri_ca_normalize` port

The isolated Python implementation is in `src/fnit/recon_all/ca_normalize_python.py`. It uses NumPy, SciPy, Numba, nibabel, and the repository's one-channel GCA reader. No FreeSurfer executable or library is called by `run_ca_normalize`. This result covers the fixed 256³ uint8 T1 profile with the 2020 one-channel GCA and a frozen voxel-to-voxel Talairach LTA. It is not yet wired into the main recon-all entry point.

Reference source: FreeSurfer 8.2 at commit `d932c45b7941662ea380a05efef580568b98d41a`, especially `mri_ca_normalize/mri_ca_normalize.cpp`, `utils/gca.cpp`, `utils/mrinorm.cpp`, `utils/mrifilter.cpp`, and `utils/mri.cpp`.

## Frozen same-input validation

Host: `control-host`, subject `fs_sub01`. Inputs are under `/data/workspaces/freesurfer_synth/work/reconall_main_20260924`:

| Input | SHA-256 |
| --- | --- |
| `single_subjects/fs_sub01/mri/nu.mgz` | `ed0155a083a64ce3929a7598f61b45875ca7dcfef1000e9de42af62a807c7e3e` |
| `single_subjects/fs_sub01/mri/brainmask.mgz` | `6d31b57e4f13006aab34a0f667643da477a63d3d8b514f0c4e1a0e7215141539` |
| `single_subjects/fs_sub01/mri/transforms/talairach.lta` | `d5996cbdb5b1fc547a5ca71d3c8bc797f1cc259df177e8deba871c3125b82b1d` |
| `bundle/average/RB_all_2020-01-02.gca` | `2fcd276a39800f01f93a4c8828ae6d0a8cea3d8b8b9fe1599d4ee54e806be93e` |

The freshly run official command was:

```bash
source /etc/profile
module load freesurfer
export FS_LICENSE=/data/workspaces/freesurfer_synth/work/reconall_external_models_20260925/private_license.txt
B=/data/workspaces/freesurfer_synth/work/reconall_main_20260924
S=$B/single_subjects/fs_sub01/mri
mri_ca_normalize -c ctrl_pts.mgz -mask "$S/brainmask.mgz" "$S/nu.mgz" \
  "$B/bundle/average/RB_all_2020-01-02.gca" "$S/transforms/talairach.lta" norm.mgz
```

The Python call is:

```python
from fnit.recon_all.ca_normalize_python import run_ca_normalize

timing = run_ca_normalize(nu, brainmask, gca, talairach_lta,
                          "norm.mgz", "ctrl_pts.mgz")
```

`benchmark_ca_normalize.py` reruns the Python call and streams both decompressed MGH files to compare the 284-byte header, every voxel in each frame, and the remaining trailer. Its measured output is `ca_normalize_fs_sub01_report.json`.

## Numerical result

| Output | Header bytes different | Voxel values different | Largest voxel error |
| --- | ---: | ---: | ---: |
| `norm.mgz`, 256³ uint8 | 0 / 284 | 0 / 16,777,216 | 0 |
| `ctrl_pts.mgz`, six 256³ float32 frames | 0 / 284 | 0 / 100,663,296 | 0 |

The independently produced intermediate `norm1.mgz`, `norm2.mgz`, and `norm3.mgz` arrays also had zero voxel differences from the official intermediate files. The selected control labels and atlas means match all six official frames exactly. The scaled input matched the official `-n 0` output exactly. The initial 246,437-sample atlas raster differed at 12 voxel locations due to source-coordinate rounding; none changed the selected controls or normalized output on this case.

The complete decompressed MGH files differ only after the voxel payload. Python writes a 20-byte trailer; the separately timed official run writes a 2,309-byte `norm.mgz` trailer and a 78,689-byte `ctrl_pts.mgz` trailer. The older and earlier fresh official outputs also have identical headers and voxels, while their `norm.mgz` trailers differ by three bytes. The two earlier official `ctrl_pts.mgz` files are identical across the full decompressed MGH content. Therefore the demonstrated parity is the image header and all voxel values, not trailer metadata.

The decisive source details were the 100-element cap on the `sigma=16` Gaussian kernel and the C double threshold `.75 * .8 * 110`, which evaluates just above 66 and excludes voxels with intensity 66. Using an integer threshold of 66 admitted an extra third-pass region and changed 56,379 final voxels by one gray level.

## Time on the same host

The fresh official full `-n 3` command took 42.68 s (`/usr/bin/time -p`); an earlier official full run reported 42 s in its own log. The final Python call took 18.00 s from function invocation through both MGH writes, including first-call Numba compilation. Its internal breakdown was:

| Python step | Seconds |
| --- | ---: |
| GCA read, mask, histogram scale, sample coordinates | 4.01 |
| Pass 1 control selection / bias correction | 0.93 / 3.18 |
| Pass 2 control selection / bias correction | 0.98 / 1.94 |
| Pass 3 control selection / bias correction | 1.59 / 1.89 |
| Write `norm.mgz` and `ctrl_pts.mgz` | 2.97 |

Separate official `-n 1` and `-n 2` runs took 22.47 s and 43.71 s. Their nonmonotonic relationship to the 42.68 s `-n 3` run shows host/runtime variation, so these single runs do not support reliable per-pass native timing differences. The full-stage timing is a single-subject observation, not a throughput estimate.
