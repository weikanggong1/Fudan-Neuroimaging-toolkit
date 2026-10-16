# Fixed cortical labels: FreeSurfer 8.2.0 comparison

The frozen `fs_sub01` recon-all log (FreeSurfer build `8.2.0-20260314-d932c45`) calls `label-cortex --fix-ga` before the separate `mri_label2label --label-cortex ... 1` call. The first command produces `lh/rh.cortex.label` from `white.preaparc`, `aseg.presurf.mgz`, and `entowm.mgz`. The later call produces `lh/rh.cortex+hipamyg.label` from `white.preaparc` and `aseg.presurf.mgz`. Thus the latter label is not the input to the former. `mri_surf2volseg` consumes `cortex.label`.

The pinned `scripts/label-cortex` sequence is: no-GA cortex label; `entowm` 3201/4201 binarization; 11 nearest samples from −1.0 to 0.0 mm along each vertex normal with a maximum; scanner-space normal x and z thresholds; intersection; `mri_cor2label`; ordered concatenation by `mri_mergelabels`. The Python port implements the fixed sequence in `label_cortex_fix_ga_python.py` and reuses `label_cortex_python.py` for the no-GA and separate hippo/amygdala calls. Native concatenation retains duplicate vertex IDs, as does the port.

## Frozen-subject result

Input: `/data/workspaces/freesurfer_synth/work/reconall_reference_accelerator-a-1/fs_sub01` on control-host. Outputs were written only to isolated scratch. The full recon-all was not rerun.

| Output | No-GA rows | Appended GA rows | Final rows | Python vs frozen official |
|---|---:|---:|---:|---|
| `lh.cortex.label` | 98,883 | 165 | 99,048 | Byte identical, SHA-256 `524245d2c800a7ff3fc8ba1d1fe04c02d09354bcc29e99cf3f64182c2f312506` |
| `rh.cortex.label` | 97,833 | 111 | 97,944 | Byte identical, SHA-256 `609cce84a6a0488c3333af213639ab55bc0312a95485baddf5f4d58e82fd6630` |
| `lh.cortex+hipamyg.label` | — | — | 101,000 | Byte identical, SHA-256 `03dfc76e15857a0e2b2f4e7ae751e6f657ae511df408016abe0430e6bd8ca6c2` |
| `rh.cortex+hipamyg.label` | — | — | 99,899 | Byte identical, SHA-256 `e0842a3bea859666a099ad83a07db60b94624ad4b0f0bcad17463612b94e4bc2` |

Fresh native replay of each child executable also matched the frozen official label rows. The native GA projection had identical values at every vertex to the Python nearest-neighbor sampling, and the combined GA masks were identical at every vertex bilaterally. The final official header has one space around the comma because the `tcsh` `mri_mergelabels` wrapper collapses the no-GA header's spaces; the Python writer reproduces that header exactly.

## control-host CPU wall time, seconds

| Fixed operation | Native LH | Native RH | Python LH | Python RH |
|---|---:|---:|---:|---:|
| No-GA `mri_label2label` | 6.44 | 6.81 | Included below | Included below |
| GA binarization | 2.10 | 2.07 | Included below | Included below |
| GA surface sampling | 1.05 | 1.09 | Included below | Included below |
| Scanner normals | 0.62 | 0.52 | Included below | Included below |
| x/z masks, intersection, GA label | 0.52 | 0.43 | Included below | Included below |
| Final `cortex.label`, total | 10.73* | 10.92* | **3.89** | **3.67** |
| Separate `cortex+hipamyg.label` | 7.72 | 6.77 | **3.46** | **2.93** |

`*` Sum of timed native child programs. control-host lacks `tcsh`, so the original wrapper could not run there. For native replay the binary-mask intersection used equivalent `mri_binarize` logic, and final concatenation used the pinned script's exact line ordering; wrapper startup and concatenation time are excluded. These timings compare the fixed stage on one CPU host, not full recon-all throughput. The Python port is NumPy/SciPy on CPU and is not yet wired into the main pipeline.

## Reproduce the output check

```bash
PYTHONPATH=src python validation/recon_all/python_gpu_port/benchmark_label_cortex_fix_ga.py \
  /path/to/fs_sub01 /path/to/scratch/labels
```

The script raises an error unless all four Python labels equal the supplied subject's official labels byte for byte. The isolated synthetic merge test passed on control-host (`1 passed`).
