# Fixed `mri_binarize` stage

`mri_binarize_gpu.py` replaces the observed FreeSurfer 8.2 call that creates
`mri/ctxsegmask.mgz`:

```text
mri_binarize --i synthseg.rca.mgz --match 3 42 --inv --o ctxsegmask.mgz
```

It reads the int32 SynthSeg MGH volume, evaluates the label mask with PyTorch,
and writes int32 MGZ. It preserves the source MGH geometry and emits FreeSurfer's
tag and color-table ordering for this input. This is one fixed stage; other
`mri_binarize` options and the surrounding reconstruction remain unported.

The frozen `fs_sub01` input SHA-256 is
`34e2fa57d5f118b3d6de4ed66730d493f4a53b4c32d3c201ad6ef686a570123f`.
On `accelerator-a-1` H100 GPU 0, the saved CUDA and native outputs were both
`256 × 256 × 256` int32, with 16,370,114 positive voxels and zero differing
voxels. Their affines, 284-byte MGH headers, footers, and complete decompressed
MGH byte streams matched. Both decompressed SHA-256 values were
`723f0dffd95b916e418ab4e780b96e5eb65652bd8a8117189d5852de9e04c98f`.
Three unit tests passed on control-host and accelerator-a-1. The same CPU Python stage also
matched the official decompressed MGH byte for byte.

After correcting the report serialization, three alternating same-host paired
trials on accelerator-a-1 passed **complete decompressed MGH byte equality in every
trial**, both on CPU and H100 CUDA 1:

| Execution | Native median (s) | Python/Torch median (s) |
| --- | ---: | ---: |
| CPU | 2.892 | 6.184 |
| CUDA 1 | 3.261 | 8.947 |

These are full fresh command times: every Python call imports Torch, creates
its own process and, for CUDA, initializes a new GPU context. This short step
is slower as a standalone Python command; no resident-process or end-to-end
speedup is claimed. The input hash and individual trial times are in
[`binarize_cpu_report.json`](binarize_cpu_report.json) and
[`binarize_cuda1_report.json`](binarize_cuda1_report.json). Reproduce with
[`benchmark_binarize.py`](benchmark_binarize.py).
