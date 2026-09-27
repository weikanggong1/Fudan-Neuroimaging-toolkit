# Talairach affine SynthMorph: frozen `fs_sub01` replay

This is an isolated replay of the published hybrid `rca-talairach` stage, not a
full recon-all run. Its archived FreeSurfer 8.2.0 (`d932c45`) wrapper invoked
`fs-synthmorph-reg --affine-only` and the native `lta_convert` program. The
archived `mri_synthmorph` line was intercepted by this package's PyTorch
neural tool (`FS_TORCH_NEURAL_TOOL mri_synthmorph device=cuda:1`). The reference
therefore establishes reproducibility of that **hybrid** chain. It is not an
independent comparison with the unmodified FreeSurfer TensorFlow model.

The actual network input is `mri/synthstrip.mgz`, not `orig.mgz`. The later
`fs-synthmorph-reg --test` call uses `orig.mgz` for a separate resampling check.
The frozen affine chain is:

1. `mri_synthmorph -m affine -t aff.lta synthstrip.mgz mni305.cor.stripped.mgz -j 4`;
2. `lta_convert --ltavox2vox --inlta aff.lta --outlta talairach.xfm.lta`;
3. `lta_convert --inlta talairach.xfm.lta --outmni talairach.xfm`.

The [Python stage](../../../src/fnit/recon_all/talairach_synthmorph.py)
uses the existing PyTorch SynthMorph affine model and a float32 source-order
RAS→voxel→RAS conversion, then writes the MNI XFM. It invokes no FreeSurfer
binary. The fixed profile needs the external `synthmorph.affine.2.h5` weight
and `mni305.cor.stripped.mgz` template. An isolated call is:

```bash
python -m fnit.recon_all.talairach_synthmorph \
  /subject/mri/synthstrip.mgz /assets/average/mni305.cor.stripped.mgz \
  --weights /external/weights --xfm /output/talairach.xfm \
  --lta /output/aff.lta --device cpu --threads 4
```

`register_talairach(moving, template, weights, output_xfm,
output_lta=None, device="cpu", threads=4)` is the Python API. The output LTA
is optional. The XFM comment names the Python writer; byte identity with
`lta_convert` is not a goal of this numerical check.

## Frozen input and numerical result

| File | SHA-256 |
| --- | --- |
| `mri/synthstrip.mgz` | `3152962e99b326e960bce57c728c3f249e51821ffd7895b711746e3e98005ad9` |
| `mni305.cor.stripped.mgz` | `fff93f13255a8d393c0e787fbfcfaf5eb379e88e955bda7e04e31552568878a4` |
| `synthmorph.affine.2.h5` | `1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6` |
| Archived `aff.lta` | `718808364d8c915e0d5bfda41645ee3b2a78e2d7eef7509ce87ec9a8ea611e2a` |
| Archived `talairach.xfm` | `1e509f0e544614555ef77ba4df6c8f6ebfa4c73c4771d7464589250dfc607e59` |

First, the archived `aff.lta` was passed through only the new Python
conversion. Its XFM matrix differed from the archived final XFM by at most
`1.526e-5` per element. This isolates the conversion roundoff from network
inference. A generic Surfa world→voxel→world roundtrip differed by
`2.670e-5`; the FreeSurfer-style float32 product order is closer.

Next, the same frozen images and weight were fed to the PyTorch affine model
on headcw CPU. The source hashes of `synthmorph/pipeline.py`, `models.py`, and
`spatial.py` exactly match the published hybrid checkout. Against the archived
GPU hybrid result, the newly computed LTA differed by at most `1.898e-5` per
matrix element. The final XFM differed by at most `3.052e-5` per element.
After applying both XFMs to the eight voxel-grid corners of the moving image,
the maximum and RMS position differences were `0.000136` and `0.0000935 mm`.
The complete [comparison JSON](talairach_synthmorph_sub01_cpu.json) records
all matrices, file hashes, errors, and runtime. The generated files remain in
an isolated validation directory; the archived subject was read only. As a
format check, the installed FreeSurfer 8.2 `lta_convert --inmni` parsed the
new XFM and wrote an LTA successfully; its isolated readback log has SHA-256
`454cf4cf5a05425b55806eadd5f4603e63992b037541485cc61afd3e62917d6d`.

The isolated Python process took 16.27 s with four CPU threads and peaked at
5,210,460 KiB RSS. The archived hybrid wrapper reported 10.65 s on gpucw1
with an H100 neural call. These are different hosts and devices, so they do
not establish a stage speed ratio. The focused conversion/API tests passed
(`2 passed`) in the archived Python environment.

## Remaining boundary

The CPU replay above does not determine CUDA behavior. Subsequent CUDA same-input replays on gpucw1 showed that the SynthMorph XFM depends on process/cuDNN state and still differs from the archived official XFM after the SynthStrip image is corrected. The [current upstream report](SYNTHSTRIP_TF32_UPSTREAM_FIX_20260927.md) distinguishes that remaining transform gap from the now exact SynthStrip, brainmask, GCA LTA and CA-normalization outputs. Independent unmodified TensorFlow registration parity remains unverified.
