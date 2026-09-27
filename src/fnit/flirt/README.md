# PyTorch FLIRT

This directory contains the public FLIRT implementation:

- `core.py`: source-derived 12-DOF correlation-ratio and 6-DOF normmi registration;
- `coordinates.py`: FSL scaled-mm and world-RAS conversion;
- `types.py`: shared result and image-input validation;
- `standalone.py`: FSL-style Python file API with atomic outputs;
- `cli.py` and `__main__.py`: `fnit-flirt` and module commands.

```python
from fnit.flirt import run_flirt

result = run_flirt(
    "subject_GM.nii.gz",
    "template_GM.nii.gz",
    output="subject_GM_to_template.nii.gz",
    omat="subject_GM_to_template.mat",
    device="cuda:0",
)
```

```bash
fnit-flirt \
  -in subject_GM.nii.gz \
  -ref template_GM.nii.gz \
  -out subject_GM_to_template.nii.gz \
  -omat subject_GM_to_template.mat \
  -dof 12 -cost corratio --device cuda:0
```

The input is the moving image. The reference defines the output grid. The
matrix maps input to reference in FSL scaled-mm coordinates; it is not a
world-RAS affine.

The implementation reports a passed FLIRT matrix functional gate for the 0.9
reference suite: 10/10 cases (`rmsdiff <= 0.05 mm`; median 0.008544 mm,
maximum 0.028984 mm). Runtime QC still reports
`validated_fsl_equivalent=false` and `current_input_compared_with_fsl=false`.
`reference_validation_matrix_gate_passed=true` describes that fixed suite; it
does not compare the current input or claim bitwise/complete numerical
equivalence.

The QC value `reference_validation_report="validation/flirt/report.public.json"`
is a source-repository artifact identifier. The wheel does not contain the
root `validation/` directory; use the
[public GitHub report](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/blob/main/validation/flirt/report.public.json) for an installed package.

See `docs/flirt/README.md` in the source
repository for the full input/output contract, argument-by-argument examples,
coordinate conversion, validation table, and timing context.

The modified port and its vendored upstream sources are covered by the
non-commercial FSL Software Licence, Release 6.0.
