# Conda final white placement

`fnit.recon_all.final_white_conda.run_final_white(subject_dir, hemi, binary, assets_dir, threads=4)` runs one FreeSurfer 8.2 final white placement using the source-built Conda `mris_place_surface` executable. It is a standalone stage. The `fnit-recon-all` scheduler does **not** call it yet because the required candidate cortex labels and annotation must precede final white placement.

## Inputs and outputs

`subject_dir` is a FreeSurfer-style subject directory, `hemi` is `lh` or `rh`, `binary` is the executable Conda-built `mris_place_surface`, `assets_dir` is the external FreeSurfer data directory, and `threads` is a positive CPU thread count. The function checks each input before running:

| Input below `subject_dir` | Role |
| --- | --- |
| `mri/brain.finalsurfs.mgz`, `mri/wm.mgz`, `mri/aseg.presurf.mgz` | Placement intensity, white matter and border-search segmentation |
| `surf/H.white.preaparc` | Ordered input mesh and rip reference |
| `surf/autodet.gw.stats.H.dat` | Subject gray/white thresholds |
| `label/H.cortex.label`, `label/H.aparc.annot` | Cortical rip mask and atlas annotation |

It writes `surf/H.white` and `mri/mrisps.white.mgz`, then returns `{"output": path, "outvol": path, "seconds": wall_time}`. `mrisps.white.mgz` is shared between hemispheres and overwritten by the later call, matching recon-all. `FS_LICENSE` remains external; the wrapper sets `FREESURFER_HOME` to `assets_dir` and `SUBJECTS_DIR` to the subject's parent. No official FreeSurfer installation is required beyond the separately source-built executable.

Python:

```python
from fnit.recon_all.final_white_conda import run_final_white

result = run_final_white(
    "/path/to/subjects/sub01", "lh",
    "/path/to/recon-cpp-build/bin/mris_place_surface",
    "/path/to/assets", threads=4,
)
```

Command line:

```bash
python -m fnit.recon_all.final_white_conda /path/to/subjects/sub01 lh \
  --binary /path/to/recon-cpp-build/bin/mris_place_surface \
  --assets-dir /path/to/assets --threads 4
```

The equivalent command in the archived FreeSurfer 8.2 `recon-all.log`, run from `subject_dir/mri`, is:

```bash
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat \
  --seg aseg.presurf.mgz --threads 4 --wm wm.mgz \
  --invol brain.finalsurfs.mgz --lh \
  --i ../surf/lh.white.preaparc --o ../surf/lh.white \
  --white --nsmooth 0 --rip-label ../label/lh.cortex.label \
  --rip-bg --rip-surf ../surf/lh.white.preaparc \
  --aparc ../label/lh.aparc.annot --restore-255 --restore-255 \
  --outvol mrisps.white.mgz --rip-bg-lof
```

The duplicate `--restore-255` and final `--rip-bg-lof` reproduce the logged command. Substitute `rh` for the right hemisphere. The source order is in [pinned recon-all](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/scripts/recon-all); the archived command and first-difference analysis are in [the earlier diagnostic](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/WHITE_PREAPARC_FIRST_DIVERGENCE_20260927.md).

## Same-input real T1 benchmark

The input is the repository's deidentified real `examples/data/sub-01_T1w.nii.gz` (SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`). On gpucw1, an isolated scratch subject symlinked **all seven stage inputs** from the archived official FreeSurfer 8.2 subject. Thus this run tests only the Conda C++ final white command and wrapper, not candidate T1-to-white reconstruction. The executable SHA-256 was `9a42f5d7b70a066daf12e67fb6a0924048b4778186ea772e8976722b226b35d5`; the private license was inherited through `FS_LICENSE` and was not copied.

| LH output versus archived official | Observed result |
| --- | ---: |
| Ordered vertices / faces | 106,622 / 213,240 in both; ordered faces exact |
| Volume geometry | Exact |
| Vertex Euclidean displacement mean / P99 / maximum | 0.000358662 / 0.005435218 / 0.751783 mm |
| Vertices with displacement >0.1 mm | 57 / 106,622 |
| `mrisps.white.mgz` | 0 / 16,777,216 differing voxels; affine and full compressed-file SHA-256 exact |
| Conda wrapper wall / max RSS | 225.64 s / 1,142,784 KiB |
| Archived official command wall | 225.85 s, different day and node load |

The two times are observations, not a controlled speed ratio. Conda output is **near**, but not vertex exact. Its numerical errors reproduce the older frozen-input diagnostic. The [raw comparison](../../validation/recon_all/python_gpu_port/final_white_conda_20260927/official_frozen_lh.json) records all input SHA-256 values, resolved paths, output metrics and wrapper time; the [run log](../../validation/recon_all/python_gpu_port/final_white_conda_20260927/official_frozen_lh.log) records optimizer decisions and process resources with the private license path redacted. Run [compare.py](../../validation/recon_all/python_gpu_port/final_white_conda_20260927/compare.py) with `CANDIDATE_SUBJECT OFFICIAL_SUBJECT WRAPPER_LOG OUTPUT_JSON` to reproduce the numerical comparison. A full candidate-input run, RH run and official-binary self-repeat are separate validation gates.
