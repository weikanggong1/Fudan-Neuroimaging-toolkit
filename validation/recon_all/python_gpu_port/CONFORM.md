# Python/PyTorch conform stage: FreeSurfer 8.2 parity

The fixed `fs_sub01` recon-all sequence is `mri_convert rawavg.mgz orig.mgz --conform`, followed by `mri_add_xform_to_header -c transforms/talairach.xfm orig.mgz orig.mgz`. This module covers the default single-volume path. It follows the [pinned `mri_convert` implementation](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_convert/mri_convert.cpp), [`MRIchangeType`/`MRIresampleFill`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mri.cpp), and [`MRIconformedTemplate`](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mri_conform.cpp).

The implementation scales float voxels to `uint8` using FreeSurfer's 1,000-bin histogram and 99.9% high cutoff, forms the 1 mm coronal target, and resamples after the type conversion. FreeSurfer's float32 VNL cofactor inverse and matrix multiplication determine source coordinates; using a float64 inverse left 904 one-level voxel errors near half-integer rounding thresholds on this subject. The port also reproduces the float32 post-resampling direction cosines and center that FreeSurfer writes into the MGH header.

## Frozen-subject result

Input: `fs_sub01/mri/rawavg.mgz`, SHA256 `2d9fb24d64cbcd9183489ad2ea1e4e93da9d3970e52e645c5280229b98a44e19`. Reference: fresh FreeSurfer 8.2.0 build `d932c45` on this input. The stored `orig.mgz` has the same first 284 MGH header bytes and all voxel bytes; recon-all subsequently adds a transform tag to its footer.

| Comparison | CPU on control-host | H100 CUDA 1 on accelerator-a-1 |
| --- | ---: | ---: |
| Mismatched voxels, 256³ | 0 | 0 |
| MGH header, 284 bytes | identical | identical |
| Scanner voxel-to-RAS maximum error | 0 mm | 0 mm |
| Complete decompressed MGH | differs in footer | differs in footer |

The Python writer emits FreeSurfer's 20-byte scan-parameter footer. Native `mri_convert` additionally emits command history and other metadata tags, whose contents vary with the invoked command. `add_xform_to_header` writes the required `TAG_MGH_XFORM` (31) with the NUL-terminated transform path. Its tag ID, length, and payload matched the stored official `orig.mgz` exactly for this subject. A complete CPU CLI call with `--xform` reproduced the official image plus this tag as a single exact byte prefix. Native `mri_info` with `DIAG_VERBOSE=1` reported `loaded talairach xform` for the Python output and printed the same path and voxel-to-RAS matrix as for official `orig.mgz`. Both read the same Talairach file (SHA256 `1e509f0e544614555ef77ba4df6c8f6ebfa4c73c4771d7464589250dfc607e59`). [Tag check](conform_xform_tag_report.json). Remaining native footer tags and command history are not reproduced; full MGH byte identity is therefore not claimed.

## Paired stage timing

| Host and scope | Native median | Python/PyTorch median |
| --- | ---: | ---: |
| control-host CPU, 3 alternate-order fresh CLI pairs | 1.483 s | 3.064 s |
| accelerator-a-1 H100 CUDA 1, one same-host fresh CLI pair | 4.06 s | 5.73 s |

Timing includes process startup, image read, and MGZ write. The control-host pairs did not clear OS caches. The accelerator-a-1 single pair is only a pilot. The goal of this stage is exact output parity; it has not shown a CLI speedup. Reports: [control-host CPU pairs](conform_cpu_control-host_report.json), [accelerator-a-1 CUDA pair](conform_cuda1_accelerator-a-1_report.json). `test_conform_gpu.py` passed 3 tests on control-host; the full subject comparison is recorded by the reports. This stage remains isolated from the published full recon-all entry point.

```bash
python -m fnit.recon_all.conform_gpu rawavg.mgz orig.mgz --device cuda:0 \
  --xform /subject/mri/transforms/talairach.xfm
```
