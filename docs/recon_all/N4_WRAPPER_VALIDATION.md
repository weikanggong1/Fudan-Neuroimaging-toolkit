# Python N4 wrapper replay

FreeSurfer 8.2 calls `mri_nu_correct.mni --ants-n4` to make `nu.mgz`. The Python implementation combines [`n4_sitk.py`](../../src/fnit/recon_all/n4_sitk.py) with [`n4_wrapper.py`](../../src/fnit/recon_all/n4_wrapper.py). The latter reproduces the five-decimal global mean ratio, `mris_calc` float32 scaling, the Talairach-centered 50 mm intensity histogram, and `mri_make_uchar`'s 1%/90% mapping. It also preserves the final MGH header and XFORM metadata state after `mri_add_xform_to_header`. This stage uses Python and compiled SimpleITK on CPU and is called by the current `run_recon_all_python` entry point. The latest v3 same-T1 connected run matched the archived official `nu.mgz` at all 16,777,216 voxels, dtype, affine and 284-byte MGH header; see the [v3 summary](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/v3_e2e_20260927/benchmark_summary.json).

The port was checked against the pinned FreeSurfer [wrapper script](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/scripts/mri_nu_correct.mni) and [`mri_make_uchar` source](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_convert/mri_make_uchar.cpp).

## Exactness on the frozen T1

The fixed input was the completed sub01 `orig.mgz` from the main validation run. Its compressed SHA-256 is `7dde820d02968c9fe18056a9cc5cc776c1a6395c48dc4d3a47eb6ba9c399f518`; its 256³ decoded uchar voxel SHA-256 is `84da8a990ef60c6ba30a4ddfbaba597a90e32bca720f96c6d741fcb08c504825`. The native executable bundle identifies itself as FreeSurfer 8.2.0-1, build `d932c45`. Both fresh replays ran on headcw with the same input, Talairach transform, and wrapper options `--uchar ... --n 2 --ants-n4`. The native script was followed by `mri_add_xform_to_header -c`, as in recon-all.

| Fresh output | Python vs native result |
| --- | --- |
| Final `nu.mgz` shape/dtype | 256³ / uchar in both |
| Voxel mismatches | 0 of 16,777,216 |
| Affine and 284-byte MGH header | Identical |
| Full decompressed MGH bytes, including XFORM footer | **Identical** |
| Full decompressed MGH SHA-256 | `6370e1ab0c888ec4d574a244ed745c640442c6ff6a2b54093d6353e89af07dc7` |
| Mean scale and histogram bins | `1.13755989346540527642`; `(4, 45)` in both |

The `nu0.mgz` metadata gate was independently checked on the reference copy of `orig.mgz`: after [`normalize_n4_footer`](../../src/fnit/recon_all/n4_wrapper.py), the entire decompressed `nu0.mgz` matched the official N4 output byte for byte. That reference input has the same voxel SHA and 284-byte MGH header as the main input, but a different XFORM footer path (1,346 versus 1,398 footer bytes); files from the two input paths should not be compared bytewise. The normalization corrects a single known FreeSurfer XFORM `UNKNOWN` tag encoding: a source length of eight including the trailing zero becomes the native N4 output's length of seven.

The earlier gpucw1 full-reconstruction archive has 34 differing `nu.mgz` voxels relative to **both** fresh headcw replays (maximum absolute difference 2); the archived values are higher by one at 9 voxels and by two at 25. Its log records the same scale and histogram mapping as the fresh native replay. Raising the fresh N4 uchar value by one at those 34 locations reproduces all archived final values. The archived intermediate `nu0.mgz` was removed by the original wrapper, so the exact upstream cause cannot be established from retained artifacts. The fresh Python output is therefore verified against a fresh official run on the same input, not against the archived `nu.mgz`.

## Timing and use

One same-host full-stage run measured 157.84 s for native `mri_nu_correct.mni` plus 0.26 s for native `mri_add_xform_to_header`, versus 125.56 s for Python N4 correction, footer normalization, and final `nu.mgz` generation. These are wall times for one run each on headcw; no GPU was used. The native script includes N4 and all postprocessing; the Python measurement includes SimpleITK import and file IO. The earlier isolated N4 timings in [N4_SITK_VALIDATION.md](N4_SITK_VALIDATION.md) are separate runs and should not be added to these totals.

For the postprocessing alone, three alternating paired trials on headcw used the same fixed `nu0.mgz`. The native median was **14.71 s** across `mri_binarize`, two `mri_segstats` calls, `mris_calc`, `mri_convert`, `mri_make_uchar`, and `mri_add_xform_to_header`; the resident Python function median was **0.99 s**. All three native/Python output pairs were identical across the full decompressed MGH. Native median substep times were 1.34, 2.32, 3.10, 2.02, 2.19, 3.34, and 0.26 s respectively. The Python timing excludes module import, whereas the native timing includes each subprocess launch. The raw [paired report](../../validation/recon_all/python_gpu_port/n4_wrapper_headcw_report.json) includes all trial times and output hashes.

To replay only the post-N4 wrapper from an existing `nu0.mgz`:

```bash
python -m fnit.recon_all.n4_wrapper \
  --orig orig.mgz --nu0 nu0.mgz --tal transforms/talairach.xfm --out nu.mgz
```

The postprocessing paired benchmark is reproducible with [`benchmark_n4_wrapper.py`](../../validation/recon_all/python_gpu_port/benchmark_n4_wrapper.py); it does not rerun N4. The remote stage logs and byte-comparison artifacts are retained under `/tmp/reconall_n4_sub01_20260925/` on headcw. Two focused tests in [`test_n4_wrapper.py`](../../tests/recon_all/test_n4_wrapper.py) passed.
