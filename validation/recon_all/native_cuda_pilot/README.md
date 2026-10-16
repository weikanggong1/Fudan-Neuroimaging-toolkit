# `mris_place_surface` intensity-term CUDA pilot

This isolated experiment reproduces the central per-vertex MRI sampling and
gradient calculation in FreeSurfer 8.2 commit `d932c45`,
`utils/mrisurf_compute_dxyz.cpp`, `mrisComputeIntensityTerm` (lines 1941–2148).
The caller is `utils/mrisurf_mri.cpp`, `MRISpositionSurface` (line 603).
The fixture uses the completed `fs_sub01` `mrisps.wpa.mgz` (the native command's
saved preprocessed intensity volume) and `lh.orig` surface. Normals are computed
from that mesh; target intensities are the real
nearest-voxel values shifted by alternating ±3 because the native per-vertex
CBV targets are not saved in the surface file. It tests the math and execution cost,
not equality to the official whole-surface output.

Generate and compile on `control-host` (GCC 8.5), then run the shared binary on accelerator-a-1:

```sh
python make_case.py /path/to/fs_sub01 case.bin
/opt/reference/CUDA/cuda-12.1/bin/nvcc -O2 -arch=sm_90 \
  -std=c++17 --fmad=false --prec-div=true -Xcompiler=-fopenmp -lgomp \
  -o intensity_term intensity_term.cu
./intensity_term case.bin 0
```

The printed `cuda_end_to_end_ms` includes allocation, host-to-device copies of
the real 16 MiB MRI and vertex records, kernel launch, synchronization, and
device-to-host copy. `cuda_context_ms` is separate. The GPU output is compared
with a serial CPU implementation of the same source formula for every gradient
component. This pilot does not modify recon-all. A successful microbenchmark
only justifies an instrumented `mris_place_surface` build and a full mesh,
vertex metric, volume, and stats comparison against the official output.
