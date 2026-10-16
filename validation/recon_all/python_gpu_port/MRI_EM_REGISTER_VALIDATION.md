# `mri_em_register` fixed-T1 stage validation

Status: **isolated full-stage fixed-input gate passes**. The input search, gradient, optimizer,
and writer are separate modules under `src/fnit/recon_all/`.
`mri_em_register_python.py` combines them but has not been connected to the
recon-all entry point. This implementation uses NumPy/SciPy/Numba on CPU;
GPU acceleration has not yet been validated.

## Reference

- FreeSurfer source: pinned `v8.2.0` commit `d932c45b7941662ea380a05efef580568b98d41a`;
  `mri_em_register/{mri_em_register.cpp,emregisterutils.cpp,findtranslation.cpp}`,
  `utils/gca.cpp`, `utils/mri.cpp`, and `utils/mrifilter.cpp`.
- Command: `mri_em_register -uns 3 -mask brainmask.mgz nu.mgz
  RB_all_2020-01-02.gca transforms/talairach.lta`.
- Fixed subject: `work/reconall_main_20260924/single_subjects/fs_sub01/mri` on
  control-host. Atlas SHA-256:
  `2fcd276a39800f01f93a4c8828ae6d0a8cea3d8b8b9fe1599d4ee54e806be93e`.
- Native full command from `work/reference_recon_all.log`: 236.52 s, 4 OpenMP
  threads. Its final voxel-to-voxel matrix is in the reference
  `transforms/talairach.lta`. The corrected Python command independently
  produced a final matrix within one float32 ULP (`7.45e-9` maximum error)
  and mapped all 315,638 atlas samples to the same source voxels.

## Implemented and checked

| Stage | Python result | Native reference |
| --- | --- | --- |
| GCA v5 read | 71,651,552-byte atlas; 64³ density nodes; 128³ priors; 324,992 classifiers; 2,450,897 prior labels; 256³ target geometry | Atlas header and `talairach.lta` destination geometry agree |
| Covariance regularization | Mean standard deviation 7.2200497; minimum determinant 5.2129118; 0 singular, 884 ill-conditioned | Native log rounds to 7.2, 5.2, 0, 884 |
| Atlas white/gray peaks | 107/61, calculated from node means and priors | Native log 107/61 |
| Five mask passes for labels 0–4 | 0/16,777,216 differing uint8 voxels | Native `init_before_intensity.mgz` |
| Input white-matter peak | 112 from a 5-voxel mean image, histogram threshold `15.96281` and sampling box `(92,80,104; 32,31,40)` | Native log: 112, threshold `16.0` rounded to one decimal, same box |
| uint8 intensity multiply | 0/16,777,216 differing uint8 voxels, using independently computed peaks 107 and 112 | Native `init000.mgz` |
| Stable 8-mm samples | 2,841 total, 1,017 Unknown; sample label and mean images each have 0/16,777,216 differing voxels | Native `init000_fsamples.mgz` and `init000_means.mgz` |
| Final `-uns 3` samples | 315,638 prior cells, 118,962 Unknown; all have direct node means and regularized variances | Native log reports 315,638 total |
| GCA mean volume and rotation center | 0/16,777,216 differing voxels; center `[126.8020892492, 119.2902549647, 105.4208947332]` | Native `-write_mean` diagnostic volume and its intensity-weighted center |
| Initial log probability | −4.38200235 | Native log −4.382 |
| Full-sample pre-EM log probability | −3.917819738 at the exact native pre-EM matrix | Native debugger return −3.917819738; exact float32 match |
| First EM gradient | All 12 components compared at the exact native pre-EM matrix; maximum absolute error `1.86e-9`, including three zero translation components | Native debugger capture at the first `computeEMAlignmentGradient` return |
| First EM line search | Final 4×4 matrix maximum absolute difference `7.45e-9` when initialized at the native pre-EM matrix; 20 Python trial calls | Native final `talairach.lta` matrix; the first native pass stops with `INFO=3` |
| LTA writer | All 27 canonical numeric/geometry lines exact when supplied the native final matrix as a validation fixture | Native `talairach.lta`; the writer does not read it during inference |
| Eight translation grids | All 16 float32 elements of the translated matrix match the native debugger capture exactly; translation `[-4.934213161468506, 11.513154983520508, -21.381580352783203]` | Native debugger capture before linear search |
| Nine scale, rotation, translation iterations | Per-iteration scores `[-3.540358, -3.511400, -3.505222, -3.505222, -3.359223, -3.343485, -3.343485, -3.309276, -3.309276]`; final pre-EM matrix 16/16 float32 exact | Native log scores and debugger capture before EM |

The Python and native pre-EM matrix (voxel to voxel, float32) is:

```text
 1.1575514078140259   0.07681363821029663  -0.1292986124753952  -19.75442886352539
-0.060859519988298416 1.2991782426834106   0.3140552043914795  -52.99283218383789
 0.09060932695865631 -0.2865518629550934   1.019007921218872    4.7753071784973145
 0           0           0            1
```

The pre-EM matrix was captured directly from the installed native binary at
its `MatrixPrint` call. The native final LTA differs from this matrix after
the 315,638-sample EM optimization. The Python full-stage command now reaches
the same pre-EM float32 matrix without using a native matrix as input. The
current optimizer implements the first native L-BFGS line search; additional
passes required by other T1 inputs have not been validated.

## Combined command from the same subject input

The experimental `mri_em_register_python.py` independently read the fixed
subject's `nu.mgz`, `brainmask.mgz`, and atlas and wrote
`work/reconall_python_gpu_20260925/em_stage_exact_source_fs_sub01/talairach.lta`
on control-host. `validate_mri_em_stage.py` exited 0 against the native LTA:
geometry and 23 nonmatrix metadata lines agree, the maximum matrix error is
`7.450580596923828e-9` (within the `2e-5` gate), and **0 of 315,638** atlas
samples map to a different rounded source voxel. The LTA is numerically and
voxel-mapping equivalent on this subject; it is not byte-identical because
the matrix has one float32-ULP difference and generated header comments differ.

The corrected Python command took **230.20 s** wall time on control-host (Python
reported 229.51 s in-stage): input preparation 10.14 s, translation 8.29 s,
nine linear-search iterations 205.25 s, EM 5.75 s, and writing 0.09 s.
The available native full-command reference took **236.52 s** on the same
host, so this one fixed-subject result is about 2.7% faster. These are
separate runs, not a controlled paired speed benchmark; no multi-subject
performance claim follows. The nine-iteration Python search is now about
2.0 times faster than its prior NumPy path while preserving the native
pre-EM float32 matrix.

All 31 captured native EM objective evaluations now match exactly. Both
implementations give `3.9178197383880615` at the pre-EM matrix,
`77444.578125` at the first large trial (step ≈34.36), and
`3.9178192615509033` at the next tiny trial (step ≈`2.15e-6`). The tiny
improvement is one float32 ULP. Native More–Thuente line search then jumps to
step ≈17.18 and eventually saves the reference final LTA at its 13th objective
evaluation. The isolated optimizer and LTA writer each passed their own
fixed-input checks; the corrected complete same-input command also passed its
LTA and atlas voxel-mapping gates.

The first native gradient was captured directly at the installed binary's
first `computeEMAlignmentGradient` return, after the pre-EM matrix. Reusing
the source-order VNL inverse and prior-to-source coordinate calculation in
the Python gradient reduced its largest component error from about `1e-6`
to `9.26e-8`. A source-order float32 accumulation of the matrix products and
gradient sum reduced this further to `1.86e-9`. The fixed-input validator
checks the 12 components against this independent native fixture with a
`2e-7` maximum-error gate. The optimizer check uses the reference pre-EM
matrix only as an isolated input fixture.

The former objective discrepancy came from three sample-to-voxel rounding
differences among 315,638 sample sites. Native debugger capture showed that
the sites, labels, and priors were all identical, while one differing voxel
changed its clamped log density by 3.62136. Replacing NumPy's affine inverse
with [VNL's float32 cofactor order](https://vxl.github.io/doc/release/core/vnl/html/vnl__inverse_8h_source.html), then multiplying prior coordinates
in source order, eliminated all three voxel differences. Casting the summed
log density to float32 *before* dividing by the sample count reproduced the
native return value. The complete first-call native/Python per-sample log
density comparison had no absolute difference over `1e-4`; its largest
float32-storage discrepancy was `4.28e-7`. This is a fixed-subject numerical
diagnostic, not an optimizer result.

Earlier bounded native diagnostics captured translation, the first linear
matrix, and the final pre-EM matrix without modifying the reference LTA.
The prior NumPy Python command took 448.82 s and failed the final LTA gate;
its nine linear iterations took 418.33 s. Source-order float32 matrix
arithmetic and Numba scoring now match the translated and pre-EM matrices
exactly; the corrected full command's nine iterations took 205.25 s.
See [`MRI_EM_SEARCH_SOURCE_VALIDATION.md`](MRI_EM_SEARCH_SOURCE_VALIDATION.md)
for the search-specific comparison. The GCA reader, sample search, and EM
candidate use CPU NumPy/SciPy/Numba; this module calls no PyTorch tensor
operation and has no GPU benchmark.

Run `test_mri_em_register_inputs.py` for the six focused tests. Run
`validate_mri_em_register_inputs.py ATLAS NU MASK DIAGNOSTICS` for the fixed
subject comparison; both passed on control-host with Python 3, NumPy, nibabel, and
SciPy/PyTorch available in the existing environment.

`validate_mri_em_first_divergence.py ATLAS NU MASK` is the minimal numerical
gate. It holds 31 native float32 trial matrices and objective returns
as validation fixtures; they are never used by the implementation. On control-host
it checked 315,638 samples in 7.601 s and exited `0` with all 31 costs
equal and the gradient error under `2e-7`. The six focused tests also passed, including exact comparison of the
native 4×4 prior-to-source matrix. The native values came from the installed stripped
`mri_em_register` binary (BuildID `2f7de2059bf6773b3feebab0d2a5bf8801d58c4f`)
via bounded debugger inspection of the matrix at the final EM call and the
captured objective function returns. No reference LTA or volume was modified.

## Remaining matching boundary

The image peak is independently computed for the fixed input. The histogram
mode, skull box, and intensity multiplication match the native log and
snapshot; the rare fallback branches of the native skull-box and robust-fit
logic are not yet implemented. The stable samples and translation search use the deterministic
`vnl_random` sequence; the generator follows the [VXL implementation](https://github.com/vxl/vxl/blob/master/core/vnl/vnl_random.cxx)
called by FreeSurfer's `OpenRan1`. The nine search iterations and final-sample
site selection and classifier attachment are ported. All 31 captured EM
objective calls match, and the first EM gradient matches the native
captured vector within `1.86e-9`. The isolated line search matches the final
matrix within `7.45e-9`; the LTA writer matches all nonmatrix numeric and
geometry lines using an independent reference matrix fixture. The combined
same-input command now passes its fixed-subject LTA and atlas voxel-mapping
gates. Other T1 inputs, EM paths beyond the first native line search, and
integration into the recon-all entry point remain unvalidated.
