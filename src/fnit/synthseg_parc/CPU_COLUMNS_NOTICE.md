# FNIT CPU columns glue provenance

`_columns_reuse.cpp` is FNIT-owned copy and public-SGEMM glue. It was written
for the bounded SynthSeg CPU experiment and does not embed upstream
Unfold3d/CPUBlas implementation bodies, FreeSurfer/FSL code, model weights,
or compiled vendor programs. The production source is byte-identical to the
4,385-byte experimental source, SHA-256
`2440abe802f1da2c14bd192f8b0f84c46b6f1f518de1dc9f0dca8429b05a2527`.

At build time it includes headers from the user's installed PyTorch, whose
BSD-style license is available in that installation and at
<https://github.com/pytorch/pytorch/blob/v2.5.1/LICENSE>. At run time the
optional CPU path uses the user's already-loaded PyTorch and MKL LP64
provider. MKL's Intel license and libgomp's GPL with GCC runtime exception
remain the terms of those installed Conda packages. FNIT does not distribute
those libraries or copy their implementation into this source.

The SynthSeg workflow and external resources retain the FreeSurfer
attribution and terms described in the repository's `THIRD_PARTY_NOTICES.md`
and `licenses/FreeSurfer.txt`. This notice does not assign a new license to
FNIT as a whole or grant additional rights to model files.

The current narrow production candidate has passed real compile/load in the
existing Conda/GCC environment, six fixed numerical contracts, copy/fallback
guards, four complete CPU processes and two complete GPU processes. A new
independent Conda installation has not been tested. Full CPU timing remains
slower than the frozen same-node official result;
see `docs/synthseg/CPU_COLUMNS.md` and
`validation/smri_cpu/seg_columns_integration_20261006/README.md`.
