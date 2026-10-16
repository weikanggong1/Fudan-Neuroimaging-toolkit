# Raw quad tessellation

`tessellate_gpu.py` translates the default FreeSurfer 8.2.0
`mri_tessellate.cpp` path at source commit
`d932c45b7941662ea380a05efef580568b98d41a`. It builds the old-format
quad surface in the same z/y/x vertex scan and directed-face creation order.
It uses PyTorch tensors and accepts `--device cpu` or a CUDA device. This is
an isolated stage; `mri_pretess` and later surface stages are not supplied by
this module.

The completed `fs_sub01` reconstruction had deleted its temporary
`filled-pretess255.mgz` and `filled-pretess127.mgz`. We recreated each input
from that subject's retained `filled.mgz` and `norm.mgz` using the original
`mri_pretess` command, then ran **fresh** FreeSurfer 8.2.0
`mri_tessellate` calls before `mris_extract_main_component`. The replayed
vertex/face counts matched the archived official log. Input SHA-256 values
and all paired trial times are in
[`tessellate_cpu_report.json`](tessellate_cpu_report.json).

| Hemisphere | Vertices | Quads | Native median | Python/Torch CPU median |
| --- | ---: | ---: | ---: | ---: |
| Left | 102,764 | 102,780 | 0.739 s | 0.243 s |
| Right | 101,454 | 101,468 | 0.806 s | 0.241 s |

Each hemisphere had three alternating paired file-to-file trials on
`control-host`, using the same `filled-pretess` input. Native times include process
startup; Python timing uses an already loaded process. The complete vertex
float32 bytes, quad vertex-index bytes, and volume geometry tags matched
the native file **byte for byte** in all trials. The command-history tag has
different text and length, so the whole files differ. Native
`mris_extract_main_component` loaded both Python outputs and found one
component in each. A single-voxel synthetic case also matched a fresh native
surface exactly and is retained as
[`test_tessellate_gpu.py`](test_tessellate_gpu.py).

The CUDA path was then replayed in three paired trials on `accelerator-a-1` H100 GPU 1
(`CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`). Both hemispheres again had
byte-identical vertex/quad cores and volume geometry tags to fresh native
outputs. Left native/CUDA medians were **1.398/0.255 s**; right medians were
**1.437/0.162 s**. The first CUDA left call took 1.017 s because it included
initial kernel work within the resident process. All times include file I/O;
Python import and CUDA context creation were excluded. Individual times and
input hashes are in [`tessellate_cuda1_report.json`](tessellate_cuda1_report.json).

A later subject-layout check called both native and Python tools from `surf/` with
`../mri/filled-pretess255.mgz` as the input argument. The Python writer now
retains that argument string in its volume-info `filename` field, as native
does, instead of resolving it to an absolute scratch path. The full 476-byte
volume geometry tag, 102,764 ordered vertices and 205,560 triangle indices
read from the quad file matched native exactly. This checks path metadata for
future runner integration; the command-history footer still differs. The
[subject-layout report](tessellate_relative_filename_report.json) records
all 476 exact volume-geometry bytes and ordered geometry counts.

These isolated stage results do not establish end-to-end recon-all parity or speed.
The stage can only be connected to a Python reconstruction after its upstream
pretess input and downstream surface stages are independently matched.
