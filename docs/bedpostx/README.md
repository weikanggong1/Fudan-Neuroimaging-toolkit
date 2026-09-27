# TorchBEDPOSTX

`TorchBEDPOSTX` estimates voxelwise posterior distributions for up to three crossing-fibre orientations from diffusion MRI. It implements the ball-and-stick signal model with Gaussian residuals, a sparse prior on subsidiary fibre fractions, and Metropolis sampling. The default `model=2` uses a Gamma distribution of diffusivities for multi-shell data. The code runs in float32 on CPU or CUDA; CUDA matmul uses TF32 where supported. It does not require an installed FSL runtime.

## Input and Python use

Provide one subject directory with `data.nii.gz` (4D diffusion signal), `nodif_brain_mask.nii.gz` (3D diffusion-space mask with matching affine), `bvals`, and `bvecs`. `bvecs` may be 3 × N or N × 3; the package normalizes nonzero vectors. The diffusion signal should already have undergone the desired motion, eddy-current, and susceptibility corrections, with matching rotated `bvecs`.

```python
from fnit.bedpostx import TorchBEDPOSTX

result = TorchBEDPOSTX(device="cuda:0")('/absolute/path/to/subject')
print(result.output_dir, result.nvoxels, result.nsamples)
```

The default output is `/absolute/path/to/subject.bedpostX`; pass `output_dir=` to change it. Existing nonempty output directories require `overwrite=True` (or CLI `--overwrite`). Default parameters mirror the FSL `bedpostx` wrapper: 3 fibres, model 2, 1000 burn-in iterations, 1250 subsequent iterations, and one saved draw every 25 iterations (50 draws per voxel). The class also accepts `nfibres`, `model` (1 or 2), `burnin`, `njumps`, `sample_every`, `ard_weight`, `chunk_size`, `seed`, and `threads`. For example, `TorchBEDPOSTX(device="cpu", threads=8, nfibres=2)`.

## Command line

```bash
fnit bedpostx --subject-dir /absolute/path/to/subject --device cuda:0
# The standalone `fnit-bedpostx` command accepts the same options.
```

Use `--output-dir` to choose another destination. `--nfibres`, `--model`, `--burnin`, `--njumps`, `--sample-every`, `--ard-weight`, `--chunk-size`, `--seed`, and `--threads` correspond to the Python options.

## Output and interpretation

`merged_th<i>samples.nii.gz`, `merged_ph<i>samples.nii.gz`, and `merged_f<i>samples.nii.gz` are 4D posterior draws for fibre *i*. Fibres are sorted by posterior mean fraction at each voxel. `mean_f<i>samples.nii.gz`, `dyads<i>.nii.gz`, mean diffusivity/baseline images, and `nodif_brain_mask.nii.gz` support quality checks. The orientation sample and mask filenames match the files read by FSL `probtrackx2`. `run.json` records the parameters, voxel count, draw count, and wall time for reproducible comparisons.

This is an independent implementation of the published model and algorithm, not a copy of FSL's `xfibres` C++. Its tensor initialization, random number generator, floating-point order, and MCMC proposal adaptation differ from FSL. Thus posterior volumes are not expected to be byte-identical. Compare posterior fraction maps, orientation axes with sign-invariant angular error, and downstream streamline density and connectivity to an original FSL run at matching settings. Do not infer anatomical connectivity probability directly from raw streamline counts; seed voxel count and number of samples affect them.

## Validation against original FSL

The [machine-readable public report](../../validation/bedpostx/report.public.json) records exact aggregate metrics, wall times, artifact checks, and source hashes.

The matched small-ROI check used 14 masked voxels from a 7 × 13 × 7 × 105 UK Biobank DWI crop on gpucw1. Both implementations used model 2, three fibres, ARD weight 1, 1000 burn-in jumps, 1250 subsequent jumps, one draw every 25 jumps (50 posterior draws), and seed 8665904. FSL 6.0.7.22 ran the original CPU `xfibres` with `--cnonlinear`; this implementation ran on one H100 GPU in float32/TF32. The DWI crop and subject-level maps stay on the research server. The following are summaries across those 14 voxels, not a whole-brain speed or accuracy estimate.

| Output | Mean absolute difference | Pearson *r* | Other check |
| --- | ---: | ---: | --- |
| Mean first-fibre fraction | 0.0112 | 0.9992 | Sign-invariant median principal-axis difference: 0.66° |
| Mean second-fibre fraction | 0.0145 | 0.4386 | Mean FSL fraction 0.0236; no voxel reached fraction ≥ 0.1 in both runs |
| Mean third-fibre fraction | 0.00353 | 0.7371 | Mean FSL fraction 0.00597; no voxel reached fraction ≥ 0.1 in both runs |
| Mean diffusivity | 0.000120 mm²/s | 0.9607 | |
| Mean diffusivity standard deviation | 0.000216 mm²/s | 0.5160 | The posterior differs substantially |

On the same gpucw1 host, original FSL CPU `xfibres` took 10.99 seconds and TorchBEDPOSTX CPU with one thread took 22.12 seconds. TorchBEDPOSTX on GPU 1 took 41.04 seconds; the FSL CPU/GPU comparison is across devices. For this 14-voxel case, FSL CPU was 2.01 times faster than Torch CPU. The original FSL process returned status 255, but reported 14/14 voxels complete and wrote valid 50-draw posterior volumes; both independent runs returned status 0. The timing includes process startup and file I/O, so it does not predict whole-brain throughput. The table above compares FSL CPU against Torch GPU. Against Torch CPU, the first-fibre fraction had MAE 0.0139 and *r* = 0.9977, while the diffusivity-standard-deviation map had MAE 0.000239 mm²/s and *r* = 0.325; the same weak secondary-fibre and diffusivity-standard-deviation agreement remains.

As a file-contract check, original FSL `probtrackx2` loaded the TorchBEDPOSTX output as three fibres with 50 draws per voxel and tracked 20/20 streamlines from one same-grid voxel. It wrote a 7 × 13 × 7 `fdt_paths.nii.gz` with nonzero density and a `waytotal` of 20. Its log ended with `finished` despite the same status-255 behavior.


## Why the low-fraction and diffusivity-spread maps differ

A controlled rerun on the same 14-voxel ROI showed that the original FSL result itself changes at least as much as the cross-implementation comparison. The FSL tensor-initialized run used `--nospat` instead of `--cnonlinear`; all other short-run settings and the seed matched. The extended runs used 5000 burn-in and 5000 sampling jumps, saving every 100th jump, so every run still saved 50 draws per voxel. Values below are ROI means in mm²/s for `d_std`.

| Run | Initialization | Burn-in / sampling jumps | Mean f2 | Mean f3 | Mean `d_std` |
| --- | --- | ---: | ---: | ---: | ---: |
| FSL CPU, reference | nonlinear | 1000 / 1250 | 0.02361 | 0.005968 | 0.0002243 |
| FSL CPU, tensor control | tensor | 1000 / 1250 | 0.01902 | 0.000199 | 0.0002782 |
| FSL CPU, extended | nonlinear | 5000 / 5000 | 0.01586 | 0.001475 | 0.0000000101 |
| Torch CPU, original | tensor | 1000 / 1250 | 0.02049 | 0.004256 | 0.0001446 |
| Torch CPU, extended original | tensor | 5000 / 5000 | 0.01598 | 0.001580 | 0.000000755 |
| Torch CPU, extended corrected | tensor | 5000 / 5000 | 0.01305 | 0.001572 | 0.000000000354 |

[FSL's model-2 source](../../src/fnit/_vendor_fsl/sources/fdt-2604.0/fibre.h) adds `log(d_std)` to the energy and, with ARD enabled, `log(f)` for each subsidiary fibre. The fitted signal remains finite as these values approach zero. Consequently, those terms pull finite-chain estimates toward zero; with no positive lower bound, the corresponding continuous density is not normalizable at zero. The original FSL `d_std` mean dropped by more than four orders of magnitude in the extended run, and 100% of its saved extended-run draws were below `1e-5`. The short-run FSL result is therefore not a stable reference for this parameter. The very small second and third fractions have the same issue; neither short run had a voxel with fraction ≥ 0.1 in both methods. The first-fibre fraction and axis are much more stable in this ROI.

The implementations also start in different states: FSL's `--cnonlinear` fits a multi-fibre nonlinear model before sampling, while TorchBEDPOSTX starts from a diffusion tensor. FSL's tensor-control and nonlinear runs had MAEs of 0.0108 for mean f1, 0.0148 for mean f2, and 0.000198 mm²/s for mean `d_std`; these are comparable to the short-run cross-implementation differences. Random draws, proposal adaptation, and floating-point precision can then produce different finite chains. A separate implementation error was found in TorchBEDPOSTX: it previously flattened the two logarithmic prior terms below `1e-8`. That floor has been removed to match FSL's source. The corrected short CPU run changed the ROI mean `d_std` from 0.000144647 to 0.000144597 mm²/s; it does not resolve the finite-chain discrepancy. Against the corresponding FSL extended run, the corrected Torch extended run reduced `d_std` MAE from 7.55×10⁻⁷ to 1.05×10⁻⁸ mm²/s and mean-f2 MAE from 0.00574 to 0.00400. Both extended `d_std` means are near zero and their voxelwise correlations are not informative. The corrected H100 short run completed with finite outputs; its FSL `d_std` MAE was 0.0002166 mm²/s versus 0.0002158 before correction. See the [diagnostic report](../../validation/bedpostx/diagnosis.public.json) for the exact run settings and aggregate results.

## Shareable synthetic example

The [synthetic example script](synthetic_example.py) creates an 8 × 8 × 1 multi-shell DWI with two known crossing-fibre fraction maps and no human imaging data. Generate the subject, fit both implementations at the settings above with `--nf=2`, then render the [comparison image](synthetic_example.png):

```bash
python docs/bedpostx/synthetic_example.py /tmp/bedpostx-synthetic
export FSLOUTPUTTYPE=NIFTI_GZ
xfibres --data=/tmp/bedpostx-synthetic/subject/data.nii.gz \
  --mask=/tmp/bedpostx-synthetic/subject/nodif_brain_mask.nii.gz \
  --bvals=/tmp/bedpostx-synthetic/subject/bvals \
  --bvecs=/tmp/bedpostx-synthetic/subject/bvecs \
  --nf=2 --model=2 --fudge=1 --bi=1000 --nj=1250 --se=25 \
  --cnonlinear --seed=8665904 --forcedir \
  --logdir=/tmp/bedpostx-synthetic/fsl_cpu
fnit bedpostx --subject-dir /tmp/bedpostx-synthetic/subject \
  --output-dir /tmp/bedpostx-synthetic/fnit_gpu --device cuda:0 --nfibres 2
python docs/bedpostx/synthetic_example.py /tmp/bedpostx-synthetic --plot
```

![Synthetic crossing-fibre fraction comparison](synthetic_example.png)

Each absolute-difference panel has its own color bar, capped at its 99th-percentile value shown in the title; its colors should be read against that bar rather than the fraction-map bars. In this 64-voxel synthetic example, the original FSL and Torch mean-fraction maps had mean absolute differences of 0.000935 (first fibre) and 0.00104 (second fibre), with Pearson correlations of 0.9999 and 0.9995. Against known input fractions, both methods had mean absolute errors near 0.0042. On the same gpucw1 host, FSL CPU `xfibres` took 27.83 seconds and TorchBEDPOSTX CPU with one thread took 19.97 seconds; TorchBEDPOSTX on GPU 1 took 34.05 seconds. The synthetic Torch CPU fraction maps had FSL MAEs of 0.000892 (first fibre) and 0.00120 (second fibre). The diffusivity-standard-deviation posterior is less consistent even here: FSL versus Torch GPU MAE was 0.000108 mm²/s and *r* = 0.658. These synthetic data were generated from the same mathematical signal model used for fitting, so the fraction-map result does not establish accuracy on real anatomy.

## References

- [FSL BEDPOSTX documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/bedpostx.html)
- [Behrens et al., NeuroImage 2007](https://users.fmrib.ox.ac.uk/~behrens/behrens_xfibres.pdf)
- [Jbabdi et al., Magnetic Resonance in Medicine 2012](https://pubmed.ncbi.nlm.nih.gov/22334356/)
- [FSL software license](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html)
