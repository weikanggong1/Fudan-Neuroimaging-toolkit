# Final smoothwm: exact same-input stage

The FreeSurfer 8.2 `recon-all.log` for the real `sub-01_T1w.nii.gz` case has two distinct smoothing stages:

| Output | Official command | Input |
| --- | --- | --- |
| `surf/H.smoothwm.nofix` | `mris_smooth -nw -seed 1234 H.orig.nofix H.smoothwm.nofix` | Unrepaired `orig.nofix`; default 10 passes |
| `surf/H.smoothwm` | `mris_smooth -n 3 -nw -seed 1234 H.white.preaparc H.smoothwm` | Placed `white.preaparc`; three passes |

`H` is `lh` or `rh`. The final command is recorded at lines 4675 and 4681 of that official subject's `scripts/recon-all.log`. The current runner still calls `smooth_surface(H.orig, H.smoothwm)` with its default 10 passes, then copies `smoothwm` to `white.preaparc`. This stage wiring is the cause of the final `smoothwm` discrepancy even when `orig` has exact geometry. It cannot be corrected by changing the number of passes on `orig`: genuine `white.preaparc` placement must precede final smoothing. The nofix output has a separate, correct ten-pass input.

## Python stage

`fnit.recon_all.smooth_surface_python.smooth_surface(input_path, output_path, iterations=3, device="cpu")` reads a FreeSurfer triangular surface and writes a triangular surface with the same ordered faces and volume geometry. It returns `None`. For the final `smoothwm`, set `input_path=surf/H.white.preaparc`, `output_path=surf/H.smoothwm`, and `iterations=3`. `device` accepts `cpu` or a CUDA device string; the benchmark below uses CPU float32. The function performs ordered one-ring vertex averaging without `mris_smooth`'s write of additional curvature files (`-nw`). Output bytes differ in the creator comment; ordered coordinates, faces and volume geometry are the numeric contract.

Standalone Python command:

```bash
python -m fnit.recon_all.smooth_surface_python \
  surf/lh.white.preaparc surf/lh.smoothwm --iterations 3 --device cpu
```

Official equivalent:

```bash
mris_smooth -n 3 -nw -seed 1234 \
  surf/lh.white.preaparc surf/lh.smoothwm
```

The corresponding `rh` invocation substitutes `rh` for `lh`. This isolated stage does not create `white.preaparc` and does not by itself fix the current end-to-end reconstruction.

## Same-input real-data benchmark

On `gpucw1`, both implementations read the **same saved official** bilateral `white.preaparc` surfaces from the deidentified real T1. The T1 SHA-256 is `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`. The FreeSurfer binary is version 8.2.0. Three alternating paired runs included file I/O and native process startup; Python import time was excluded. Comparison used ordered float32 coordinates, ordered faces, and FreeSurfer volume geometry.

| Hemisphere | Vertices / faces | Exact coordinate components | Max difference | Python wall median | Official wall median |
| --- | ---: | ---: | ---: | ---: | ---: |
| LH | 106,622 / 213,240 | 319,866 / 319,866 | 0 mm | 7.889 s | 3.153 s |
| RH | 105,541 / 211,078 | 316,623 / 316,623 | 0 mm | 6.634 s | 3.609 s |

All ordered faces and volume geometry fields also matched. The three-run timing arrays, input hashes, and native replay comparisons are in [`report.json`](../../validation/recon_all/python_gpu_port/smoothwm_final_same_input_20260927/report.json). The same-input result establishes the Python operator's parity; it does not establish parity for current final `smoothwm` or downstream metrics until `white.preaparc` is generated from the correct surface placement chain.

Reproduce the paired benchmark from an environment containing the FreeSurfer binary and this Python package:

```bash
python validation/recon_all/python_gpu_port/benchmark_smooth_surface.py \
  --left-surface SUBJECT/surf/lh.white.preaparc \
  --right-surface SUBJECT/surf/rh.white.preaparc \
  --native-binary /path/to/mris_smooth \
  --output-dir /path/to/output --iterations 3 --device cpu --repeats 3
```
