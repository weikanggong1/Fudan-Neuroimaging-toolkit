# Conda pial.T1 placement

`fnit.recon_all.pial_t1_conda.run_pial_t1(subject_dir, hemi, binary, assets_dir, threads=4)` runs one FreeSurfer 8.2 `mris_place_surface --pial` command with the Conda source-built C++ executable. It is a standalone stage; the current `fnit-recon-all` scheduler does not call it. The eight prerequisite files must be produced by the preceding subject stages.

## Inputs, output, and use

`subject_dir` is a FreeSurfer-style subject directory, `hemi` is `lh` or `rh`, `binary` is executable Conda-built `mris_place_surface`, `assets_dir` is the external FreeSurfer data directory, and `threads` is a positive CPU thread count. The function checks these inputs before starting:

| Input below `subject_dir` | Role |
| --- | --- |
| `mri/brain.finalsurfs.mgz`, `mri/wm.mgz`, `mri/aseg.presurf.mgz` | Intensity, white-matter and border-search volumes |
| `surf/H.white` | Ordered final white mesh, also used as repulsion and pinning reference |
| `surf/autodet.gw.stats.H.dat` | Subject gray/white thresholds |
| `label/H.cortex+hipamyg.label`, `label/H.cortex.label` | Cortical rip and medial-wall pinning masks |
| `label/H.aparc.annot` | Atlas annotation passed to the native command |

It writes `surf/H.pial.T1` and returns `{"output": path, "seconds": wall_time}`. It does not write final `pial`, thickness, area, volume, curvature or atlas statistics. `FS_LICENSE` remains external; the wrapper passes `FREESURFER_HOME=assets_dir` and `SUBJECTS_DIR=subject_dir.parent` to the executable. The Conda build instructions are in [Conda C++ stages](CONDA_CPP_STAGES.md).

Python:

```python
from fnit.recon_all.pial_t1_conda import run_pial_t1

result = run_pial_t1(
    "/path/to/subjects/sub01", "lh",
    "/path/to/recon-cpp-build/bin/mris_place_surface",
    "/path/to/assets", threads=4,
)
```

Command line:

```bash
python -m fnit.recon_all.pial_t1_conda /path/to/subjects/sub01 lh \
  --binary /path/to/recon-cpp-build/bin/mris_place_surface \
  --assets-dir /path/to/assets --threads 4
```

The equivalent command in the archived FreeSurfer 8.2 `recon-all.log`, run from `subject_dir/mri`, is:

```bash
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat \
  --seg aseg.presurf.mgz --threads 4 --wm wm.mgz \
  --invol brain.finalsurfs.mgz --lh --i ../surf/lh.white \
  --o ../surf/lh.pial.T1 --pial --nsmooth 0 \
  --rip-label ../label/lh.cortex+hipamyg.label \
  --pin-medial-wall ../label/lh.cortex.label \
  --aparc ../label/lh.aparc.annot \
  --repulse-surf ../surf/lh.white --white-surf ../surf/lh.white \
  --restore-255
```

For the right hemisphere, use `rh` for file names and the hemisphere flag. The official stage order is in the [pinned recon-all source](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/scripts/recon-all).

## Same-input real T1 benchmark

The reference is the repository's deidentified real `examples/data/sub-01_T1w.nii.gz` (SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`) reconstructed by FreeSurfer 8.2. The isolated LH scratch subject symlinks all eight prerequisite files from the archived official subject; no official pial geometry is an input. This experiment tests the Conda C++ `pial.T1` command on correct upstream inputs, not candidate end-to-end reconstruction. The [comparison JSON](../../validation/recon_all/python_gpu_port/pial_t1_conda_20260927/official_frozen_lh.json) preserves every input path and SHA-256. Its [run log](../../validation/recon_all/python_gpu_port/pial_t1_conda_20260927/official_frozen_lh.log) preserves optimizer progress and process resources. Run [compare.py](../../validation/recon_all/python_gpu_port/pial_t1_conda_20260927/compare.py) with `CANDIDATE_SUBJECT OFFICIAL_SUBJECT WRAPPER_LOG OUTPUT_JSON` to reproduce the comparison.

| LH output versus archived official | Observed result |
| --- | ---: |
| Ordered vertices / faces | 106,622 / 213,240 in both; ordered faces exact |
| Volume geometry | Exact |
| Exact coordinate components | 28,062 / 319,866 |
| Vertex Euclidean displacement mean / P99 / maximum | 0.0296608 / 0.229365 / 1.669266 mm |
| Vertices with displacement >0.1 mm | 5,571 / 106,622 |
| Conda wrapper wall / max RSS | 245.32 s / 828,564 KiB |
| Archived official `recon-all.log` stage wall | 3.68 min (220.8 s), separate run and load |

The Conda executable SHA-256 was `9a42f5d7b70a066daf12e67fb6a0924048b4778186ea772e8976722b226b35d5`, and the wrapper source SHA-256 was `61450061862bf4b77d42b879e59c4f220c45826a1fa806317b16b149d011e1a0`. All eight input hashes and resolved paths match the archived official subject. The Conda optimizer took 42 accepted steps in this log; the official and Python same-input checks took 41. This Conda output does not meet vertex parity and should not replace the exact Python stage on the current evidence.

The independent [Python pial placement](PYTHON_PIAL_PLACEMENT.md) previously matched the same archived official LH geometry exactly on correct official inputs and took 1,240.8 s on headcw. A separate official C++ repeat took 116.05 s on headcw. The archived 220.8 s, fresh official 116.05 s and present Conda 245.32 s observations span different runs and changing host loads; they are not a controlled speed ratio. The Conda C++ stage is much faster than the exact Python CPU implementation in these narrow observations, but its geometry is inaccurate. A full candidate-input run, RH run and downstream metric/atlas comparisons remain separate gates.
