# Fixed-T1 `mri_em_register` search precision and timing

This report covers the isolated translation and nine-parameter searches for
`fs_sub01`. The current Python stage command calls the validated search
candidate. Its final EM optimization and LTA passed a separate fixed-subject
acceptance gate, documented in
[`MRI_EM_REGISTER_VALIDATION.md`](MRI_EM_REGISTER_VALIDATION.md).

## Reference and arithmetic

- FreeSurfer source: commit `d932c45b7941662ea380a05efef580568b98d41a`,
  `mri_em_register/findtranslation.cpp`,
  `mri_em_register/mri_em_register.cpp`, and `utils/matrix.cpp`.
- Installed reference binary BuildID:
  `2f7de2059bf6773b3feebab0d2a5bf8801d58c4f`.
- Input: fixed `fs_sub01` `nu.mgz` and `brainmask.mgz`; 2020 single-T1 GCA,
  SHA-256 `2fcd276a39800f01f93a4c8828ae6d0a8cea3d8b8b9fe1599d4ee54e806be93e`.
- The native matrices were read as 16 float32 values from the installed binary
  using bounded GDB breakpoints. The translation and rotation origin were
  captured before the first linear iteration; iteration 0 was captured at its
  `MatrixPrint` call; final pre-EM was captured before EM. GDB runs stopped
  after their target snapshots and did not modify the native reference LTA.

`findtranslation.cpp` computes its grid delta using float operands and then
iterates in double. The earlier Python code computed that delta in double,
giving a few float32 ULPs of translation error. The native `MatrixMultiply`
uses float accumulation over the four inner products in order. The search
recomputes its selected rotation as `Z @ (Y @ X)`. Both orders matter for the
final float32 matrix. The Numba likelihood kernel retains the source-order
coordinate math and reduces the per-sample float64 values with NumPy's
pairwise sum, matching the previous NumPy scorer at 100/100 random nearby
candidate transforms. No reduced-precision arithmetic is used.

## Fixed-subject gates

| Gate | Result |
| --- | --- |
| Eight translation grids, native vs Python | 16/16 float32 matrix elements exact; final translation `[-4.934213161468506, 11.513154983520508, -21.381580352783203]` |
| Nine-parameter iteration 0, native vs Python | 16/16 float32 matrix elements exact; score `-3.5403575897216797` |
| Nine iterations, native vs Python | All nine scores match the native log at its printed precision; final pre-EM 16/16 float32 matrix elements exact, maximum absolute error 0 |

The final pre-EM matrix is:

```text
 1.1575514078140259   0.07681363821029663  -0.1292986124753952  -19.75442886352539
-0.060859519988298416 1.2991782426834106   0.3140552043914795  -52.99283218383789
 0.09060932695865631 -0.2865518629550934   1.019007921218872    4.7753071784973145
 0                    0                      0                    1
```

## Search timing on control-host

| Search implementation | Translation | Nine linear iterations | Native pre-EM matrix |
| --- | ---: | ---: | --- |
| Original NumPy full Python stage | 13.62 s | 418.33 s | 1.1444e-5 maximum error |
| Isolated JIT scorer with original matrix construction | 13.97 s | 206.81 s | Same 1.1444e-5 error |
| Correct float32 grid and source-order matrices, isolated | 10.14 s | 215.12 s | 16/16 exact |
| Corrected full Python stage | 8.29 s | 205.25 s | 16/16 exact |

The final search is about 1.94 times as fast as the previous NumPy nine-round
implementation on this fixed subject. The corrected full Python stage took
230.20 s wall time; the native full `mri_em_register` command took 236.52 s
in a separate reference run. These are entire-command times on the same host,
but not a controlled paired benchmark. Numba compilation and system load
vary between isolated runs. The fixed search uses CPU NumPy/Numba, not
PyTorch GPU.

Run `validate_mri_em_translation_source.py ATLAS NU MASK` for the exact
translated matrix and `validate_mri_em_search_jit.py ATLAS NU MASK
--source-order --first-iteration` for the exact first iteration. The same
search validator without `--first-iteration` checks the final pre-EM matrix
against the native float32 fixture. All three gates exited 0 on control-host.

The full Python stage command and independent native LTA comparison passed:
maximum matrix error `7.45e-9`, equal metadata and geometry, and zero source
voxel-mapping differences for all 315,638 atlas samples. The stage remains
isolated from the recon-all runner and is unvalidated on other inputs.
