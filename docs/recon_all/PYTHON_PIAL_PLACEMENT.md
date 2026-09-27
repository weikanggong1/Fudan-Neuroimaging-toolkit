# Python pial.T1 placement

`fnit.recon_all.place_pial_python.place_pial_t1` implements the four-pass FreeSurfer 8.2 `mris_place_surface --pial` geometry optimizer and its medial-wall pinning/intersection repair. It uses NumPy, Numba and nibabel on CPU. It is a standalone stage API; the default `fnit-recon-all` runner still makes an approximate pial surface and does not call this function. There is no CUDA validation or end-to-end equivalence claim for this stage.

## Inputs, output, and call

`subject` is a FreeSurfer-style subject directory; `hemisphere` is `lh` or `rh`. All seven required files must exist and come from their correct preceding stages:

| Input relative to `subject` | Role |
| --- | --- |
| `surf/{hemi}.white` | Ordered white vertices/faces, volume geometry and surface tags; the pial optimizer begins from these coordinates. |
| `surf/autodet.gw.stats.{hemi}.dat` | Subject-specific gray/white intensity thresholds. |
| `label/{hemi}.cortex+hipamyg.label` | Vertex rip mask for placement and intersection repair. |
| `label/{hemi}.cortex.label` | Medial-wall pinning to white after optimization. |
| `mri/brain.finalsurfs.mgz` | Placement intensity volume and voxel geometry. |
| `mri/wm.mgz` | White matter class for intensity preprocessing. |
| `mri/aseg.presurf.mgz` | Border-search segmentation. |

```python
from fnit.recon_all.place_pial_python import place_pial_t1

report = place_pial_t1(
    "/path/to/subjects/sub01", "lh",
    "/path/to/subjects/sub01/surf/lh.pial.T1",
)
```

The optional `output` defaults to `subject/surf/{hemi}.pial.T1`. The function writes one FreeSurfer triangular surface with the input white's **ordered faces, full volume-geometry tag and auxiliary footer**, replacing its vertex coordinates with placed pial coordinates. It returns a dictionary containing `output` (path string), `hemisphere`, `steps` (accepted optimizer steps), `pass_ends` (four cumulative step numbers), `cleanup` (intersection counts, trace and smoothing counts), and `seconds` (wall time). `max_steps` defaults to 200 as a non-convergence guard and raises rather than writing a partial surface.

No thickness, area, curvature, volume or atlas statistics are written by this function. The post-pial metric and atlas stages must follow. It does not create any prerequisite white/label/MRI input. In particular, the current runner's `smoothwm` copy used as `white` cannot support the same-input exact result reported below; `white.preaparc` placement, cortex/hippocampal label creation, sphere registration/aparc annotation, and final white placement must happen first. This pial optimizer does not directly read `white.preaparc` or `aparc.annot`, but both belong to that preceding official order.

The official equivalent, run from `subject/mri` with FreeSurfer 8.2 and `FS_LICENSE`, is:

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

For RH, replace `lh` with `rh` in names and the hemisphere flag. The Python function reproduces the fixed pial behavior of this command; it does not need the annotation because this fixed branch uses the explicit rip labels.

## Frozen real-T1 same-input validation

The reference is the completed FreeSurfer 8.2 `a_official` subject from a real T1 scan (`mri/orig.mgz` SHA-256 `c99c246200cc35479b6b8cd691457985b66f2062d4a37a0c3592ff1b678b985c`). The exact subject files remain on headcw under `work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official`; no patient image is stored in Git. The candidate reads the original subject's white, brain.finalsurfs, WM, aseg, labels and threshold file without reading any official pial geometry, optimizer log, decision, or RAM checkpoint. Input file hashes are in the [manifest](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/input_manifest.json).

| Same-input stage | Python output vs official | Observed time |
| --- | --- | ---: |
| LH pial.T1 | 41 independent steps; pass ends 26/32/36/41; 106,622/106,622 ordered vertices, 319,866/319,866 float32 coordinates, 213,240/213,240 ordered faces exact. Volume geometry fields and full footer bytes exact after source-preserving write. | Python 1,240.8 s on headcw; fresh official C++ pial 116.05 s on headcw in a separate run. These were under different concurrent loads, so no controlled speed ratio is claimed. |
| RH pial.T1 | 41 independent steps; pass ends 26/31/35/41; 105,541/105,541 ordered vertices, 316,623/316,623 float32 coordinates, 211,078/211,078 ordered faces exact. Volume geometry fields and full footer bytes exact. | Python 1,154.2 s; fresh official C++ RH repeat 91.77 s on headcw, in separate runs under changing load. |

The output SHA-256 may differ because the first surface comment records creation provenance. The [bilateral geometry, footer and timing reports](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/python_pial/) give the exact comparisons. The output of this stage can be fed into the existing Python vertex-map functions with the same official white and cortex label. In that controlled check, p99 absolute errors versus the archived official maps are at most 2.38e-7 mm for thickness, 2.38e-7 mm² for pial area, and 4.77e-7 mm³ for vertex volume; corresponding maxima are 4.77e-7 mm, 9.54e-7 mm² and 1.91e-6 mm³ for either hemisphere. The curvature function has small but nonzero algorithmic error: LH/RH p99 2.71e-5/2.15e-5, maxima 4.54e-4/2.03e-4. All eight bilateral maps have zero vertices beyond the established tolerance (area: 0.001 + 0.001 × |reference|; other maps: 0.005 + 0.001 × |reference|). The [map comparator](../../validation/recon_all/python_gpu_port/compare_pial_metric_maps.py) and [LH/RH JSON reports](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/place_geometry_pair/python_pial/) retain per-map counts. This does not establish current default runner metrics or throughput. The CPU pial optimizer is substantially slower in these narrow observations; collision/KDTree work is the main optimization target after the full upstream chain matches.

The required Python dependencies (`nibabel`, `NumPy`, `SciPy`, `PyTorch`, `Numba`) are installed by the repository's `environment.yml` through the `recon-all-python-stages` package extra. No external FreeSurfer executable is called by `place_pial_t1`.
