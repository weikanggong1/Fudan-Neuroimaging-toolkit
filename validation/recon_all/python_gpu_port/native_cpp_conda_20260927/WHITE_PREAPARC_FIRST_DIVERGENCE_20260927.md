# FreeSurfer 8.2 white surface placement: bounded same-input diagnosis

## Real image and exact input contract

The case is the real `examples/data/sub-01_T1w.nii.gz` T1 scan, SHA-256
`f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`.
The historical FreeSurfer 8.2 `recon-all -all -parallel -openmp 4 -itkthreads 1`
reference is `.../work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official`.
The following tests kept its `mri/brain.finalsurfs.mgz`, `mri/wm.mgz`,
`mri/aseg.presurf.mgz`, `surf/autodet.gw.stats.lh.dat`, `label/lh.cortex.label`,
and `label/lh.aparc.annot` fixed. MGZ inputs are conformed 256 x 256 x 256
volumes; `lh.orig`/`lh.white.preaparc` are ordered 106,622-vertex,
213,240-face FreeSurfer geometries. The only changed input in the connected
white test was `lh.white.preaparc`: the Conda source-built result from the
same frozen official `lh.orig` instead of the archived official result.

Source is pinned FreeSurfer commit `d932c45b7941662ea380a05efef580568b98d41a`.
The installed binary was built with GCC 4.8.5; the Conda source binary with
Conda GCC 11.4.0. Both run on gpucw1 with `--threads 4`, the same authorized
external license, and FreeSurfer 8.2 command arguments. The installed binary
is used only as an accuracy reference, never a packaged dependency.

## Command and outputs

Official `white.preaparc` command, run from the subject's `mri` directory:

```
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --wm wm.mgz --threads 4 --invol brain.finalsurfs.mgz --lh --i ../surf/lh.orig --o ../surf/lh.white.preaparc --white --seg aseg.presurf.mgz --restore-255 --nsmooth 5 --rip-bg-no-annot --rip-bg --rip-bg-lof --restore-255 --outvol mrisps.wpa.mgz
```

Official final `white` command:

```
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --seg aseg.presurf.mgz --threads 4 --wm wm.mgz --invol brain.finalsurfs.mgz --lh --i ../surf/lh.white.preaparc --o ../surf/lh.white --white --nsmooth 0 --rip-label ../label/lh.cortex.label --rip-bg --rip-surf ../surf/lh.white.preaparc --aparc ../label/lh.aparc.annot --restore-255 --restore-255 --outvol mrisps.white.mgz --rip-bg-lof
```

Conda calls replace only the executable path with
`.../work/reconall_cpp_conda_20260927/fs_cpp/bin/mris_place_surface` and write
to isolated subject/output paths. Each call produces a FreeSurfer geometry
file (`V x 3` float coordinates, `F x 3` ordered face indices and volume
geometry metadata) and one 256-cubed uint8 MGZ diagnostic volume. It also
prints optimizer progress to a text log. The binary does not compute cortical
thickness, area or volume in these two geometry calls.

## Paired accuracy and time

| Comparison, left hemisphere | Same input? | Mean displacement | P99 | Maximum | Vertices >0.1 mm |
| --- | --- | ---: | ---: | ---: | ---: |
| Official vs Conda `white.preaparc`, frozen official `orig` and MRI | Yes | 0.0004206 mm | 0.006999 mm | 0.6300 mm | 50 / 106,622 |
| Official vs Conda `white`, frozen official `white.preaparc` and all auxiliaries | Yes | 0.0003587 mm | 0.005435 mm | 0.7518 mm | 57 / 106,622 |
| Installed vs Conda final `white`, identical **Conda** `white.preaparc` and official auxiliaries | Yes | 0.0004806 mm | 0.007325 mm | 0.9820 mm | 68 / 106,622 |
| Installed final `white` from Conda `white.preaparc` vs archived official final `white` | No, one controlled upstream mesh changed | 0.0221263 mm | 0.21981 mm | 2.7519 mm | 3,899 / 106,622 |
| Conda continuous `white.preaparc` to final `white` vs archived official final `white` | Connected chain | 0.0221146 mm | 0.22003 mm | 2.7520 mm | 3,887 / 106,622 |

All pairs have identical ordered face indices. The `mrisps.wpa.mgz` and
`mrisps.white.mgz` outputs from frozen-input Conda and installed calls are
identical at all 16,777,216 voxels **and** by complete file SHA-256. Of the
50 preaparc outliers above 0.1 mm, 34 fall in the archived cortex label;
3,871 of 3,887 connected-final-white outliers do. Therefore the amplified
white difference directly affects cortical vertex metrics.

The isolated installed final-white call with Conda preaparc took 174.85 s
(user 350.72 s, max RSS 1,326,720 KiB). The prior Conda connected call
took 200.05 s. These are separate wall-clock runs under different shared-node
load and do not establish a speed advantage. The Conda frozen preaparc and
white calls took 149.46 s and 150.91 s, respectively; the historical official
preaparc and white calls were 242.65 s and 225.85 s. These historical
timings are descriptive, not matched performance trials.

## First visible divergence

An isolated source probe wrote the Conda mesh immediately after
`MRISaverageVertexPositions(surf, 5)` and immediately after
`MRISremoveIntersections(surf, 0)`. Both outputs matched the already captured
installed FreeSurfer RAM checkpoint in all 106,622 vertex coordinates and all
ordered faces. The preaparc segmentation/outvol MGZ is also byte exact.

Two independent `--debug-vertex 38359` preaparc runs used exactly the same
frozen input files. Both debug runs reproduced the respective ordinary-run
mesh coordinates exactly; output file SHA changed because command/path
metadata changed. All 42 accepted optimizer step indices and `dt` values
were identical. The first printed force discrepancy for this selected
high-difference vertex occurred after accepted step 5, before step 6:
`dxyz.z = -0.030328` (installed) versus `-0.0303281` (Conda), with the printed
position and normal still equal. The first visible global SSE discrepancy
occurred at accepted step 7: `543344.0` (installed) versus `543343.9`
(Conda), relative difference approximately 1.8e-7. Later geometry diverges.
The debug runs took 223.44 s installed and 303.76 s Conda, but verbose
diagnostics and concurrent shared-node load make those times unsuitable for
production benchmarking.

This confines the *observed* first difference to the C++ deformation/force
and integration chain, after initial smoothing, intersections, and input
volume/label handling. `utils/mrisurf_integrate.cpp` accumulates gradient
terms and invokes `MRISmomentumTimeStep` in `utils/mrisurf_timeStep.cpp`;
both have floating-point reductions and branching near zero. The current
evidence does not identify a single incorrect expression or prove a safe
source patch. A general compiler-flag change or unconditional tolerance clamp
would be speculative. No production code was modified.

## Reproducible artifacts

Raw commands, logs, outputs, elapsed time and mesh JSON are in
`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/white_binary_pair_codex_20260927`
and `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/white_step7_probe_codex_20260927`.
The latter's `buildprobe/commands.txt` records the sole diagnostic C++ object
compile and link commands, and `buildprobe/lh.after_*.comparison.json` records
preoptimizer exactness. No private license bytes were copied into artifacts.
