# Fixed `mri_label2label --label-cortex` replacement

The frozen FreeSurfer 8.2 recon-all command inside `label-cortex --fix-ga`
reads `surf/{lh,rh}.white.preaparc` and `mri/aseg.presurf.mgz`, passes
`KeepHipAmyg01=0`, and writes the temporary no-GA cortex label. The pinned
source commit is `d932c45b7941662ea380a05efef580568b98d41a`.
`mri_label2label.cpp` dispatches to `MRIScortexLabelDECC(mris, aseg, 4, 4,
-1, 0)` in `utils/mrisutils.cpp`.

`label_cortex_python.py` implements this fixed call with NumPy, SciPy, and
NiBabel. It samples `aseg` along the white-surface normal, applies FreeSurfer's
ventricle, hippocampus/amygdala, thalamus, putamen, and cortical-gray tests,
then performs one-ring closing, largest-component filtering, four erosions,
four dilations, and another largest-component filter. It writes the same
ordered ASCII `.label` format without invoking FreeSurfer.

## Paired validation on control-host

Input: official frozen `reconall_reference_accelerator-a-1/fs_sub01` subject. Native
FreeSurfer 8.2.0-1 was freshly replayed with the same bilateral calls. The
Python API and `python -m fnit.recon_all.label_cortex_python`
CLI were each checked. Complete output text, including header, vertex IDs,
coordinates, and stat values, is byte-for-byte equal on both hemispheres.

| Hemisphere | Label vertices | Full-text SHA-256, native = Python | Native CPU | Python CPU API |
| --- | ---: | --- | ---: | ---: |
| LH | 98,883 | `30abd2d99f598e9659aae29aaa8c14334adc603655d7814a7156e66d4691c815` | 7.24 s | 2.98 s |
| RH | 97,833 | `40541d50e4e13c7cc6bd73ee801b0fd4fa9e6dcf001d15b2268edea0876b8547` | 6.97 s | 2.79 s |

Times are one sequential same-host process run per hemisphere with
`OMP_NUM_THREADS=4` for native, including input and output I/O. Two focused
operation tests pass on control-host. `benchmark_label_cortex.py` reproduces the
Python API comparison against a native `.label` and reports exact byte and
ordered-vertex checks.

This module replaces the no-GA `mri_label2label` call only. The enclosing
`label-cortex --fix-ga` script also samples `entowm.mgz`, evaluates surface
normal orientation, and merges the gyrus-ambiens correction into the final
`{lh,rh}.cortex.label`; that separate part is not implemented here. Parity was
checked on the frozen T1 subject, including its default
`FS_NUCACC_IS_MEDIAL_WALL=0` behavior.
