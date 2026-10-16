# Connected SynthSeg CUDA comparison against FreeSurfer 8.2

The input was `orig.mgz` generated from the public `sub-01_T1w.nii.gz` by the
native-free Python input chain (SHA-256
`caf74b3a2a7c9929d280ac4db5bd80cc69a59ddbeca54af2884906cd8acf5de6`).
Its 16,777,216 input voxels match the unmodified official subject's `orig.mgz`.
The independently configured SynthSeg 2.0 weight SHA-256 was
`f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e`.
The candidate ran on `accelerator-a-1` H100 GPU 1 with four CPU threads. A fresh unmodified FreeSurfer 8.2 CPU replay on this exact `orig.mgz`
matched the archived official subject at every hard-label voxel and matched
its volume CSV byte for byte; see the [official replay report](official_synthseg_replay_20260926.json).

The initial candidate used cuDNN TF32. It differed at **166/16,777,216**
segmentation voxels and at up to **89.3775 mm³** among the 33 soft-volume
columns. The other SynthSeg preprocessing, model weights, and postprocessing
were identical to the next trial. This establishes that TF32 cannot be used
for the strict hard-segmentation gate on this fixed input.

With cuDNN TF32 disabled only for SynthSeg, all **16,777,216 hard labels**
matched the official result. The output dtype is `float32`; the affine,
284-byte MGH header, and complete header-plus-voxel payload match. The
source-order posterior reduction alone left up to 0.185 mm³ error among the
33 soft-volume columns. Posterior comparison traced its largest error to
one voxel whose PyTorch foreground probability is approximately 0.25000009,
at the official 0.25 component-mask boundary. Applying a small internal
threshold margin (`0.2500001`) reduced the largest posterior error from
0.2401 to 9.90 × 10⁻⁶, with no entries above 10⁻⁴ among 553,648,128
probabilities. All hard labels remain exact. In the saved candidate CSV,
the maximum printed difference is **0.05 mm³**; 26/33 columns fall within
0.005 mm³. The official writer renders NumPy `float32` scalars, whereas the
candidate had rendered Python `float` scalars. Re-rendering the same saved
candidate values with the official `float32` string form reduces the maximum
printed difference to **0.04 mm³**, with 27/33 columns within 0.005 mm³.
The two `float32` values match bitwise for 19/33 columns; the maximum genuine
`float32` difference is 0.03125 mm³ (CSF). This deterministic serialization
check is in the [CSV representation report](synthseg_csv_repr_20260926.json)
and its [replay script](experimental/audit_synthseg_csv_repr.py); the writer
code now uses that format, with a focused API regression test. GPU inference
was not repeated for a formatting-only change. **Six CSV columns still fail
the 0.005 mm³ gate.** The threshold margin is an empirical same-input FP32
parity correction and has not been validated on other subjects.

The official script sums float32 foreground posteriors with NumPy after
restoring the input orientation, then multiplies by the input header's voxel
resolution. The candidate now follows that order. A single additional
FP32 CUDA inference compared four reductions: GPU aligned-grid sum
(464.8 mm³ maximum difference), NumPy aligned-grid sum (41.12125 mm³),
NumPy restored-grid sum with affine determinant (0.09125 mm³), and the
source-correct NumPy restored-grid sum with header resolution (0.185 mm³),
before the component-mask threshold correction. The determinant is
0.99999976158 mm³ while header voxel volume is 1.0 mm³ on this input.
Selecting the smaller error from the determinant would change the official
algorithm. The [baseline posterior comparison](synthseg_posterior_baseline_20260926.json),
[threshold diagnosis](synthseg_foreground_threshold_20260926.json), and
[corrected posterior comparison](synthseg_posterior_tie_20260926.json) isolate
the one threshold discontinuity and the remaining small posterior differences.

The candidate calls took 7.35 s with TF32, 4.98 s with full float32 before
the threshold adjustment, and 5.05 s after it. The isolated official CPU
replay took approximately 154 s from timestamped logs. These were unpaired
runs on different devices with shared-host load, so they establish no stage
speed ratio. No whole reconstruction was rerun. The [TF32 report](connected_synthseg_gpu_tf32_official_20260926.json),
[pre-adjustment FP32 report](connected_synthseg_gpu_fp32_official_20260926.json),
[adjusted FP32 report](connected_synthseg_gpu_fp32_tie_20260926.json),
[reduction diagnostic](synthseg_reduction_diagnostic_fp32_20260926.json),
[candidate replay script](experimental/run_connected_synthseg_gpu_official.sh),
and [official replay script](experimental/run_official_synthseg_posterior.sh)
preserve the comparisons. Other subjects remain untested.
