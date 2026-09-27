# Changelog
All notable changes to MMORF releases will be documented in this file.
Please be sure to check it thoroughly when updating versions.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] 2023-04-13
- This release deals better with registrations where one T1 image has a much brighter scalp signal than the other. This relise on providing a reasonable brain mask for at least one of the two images.
- Note that the masks supplied with the `mask_ref_scalar` and `mask_mov_scalar` flags will now also be applied to the bias field estimation.
- This release reduces some CPU load due to unnecessary image smoothing and oversampling of images during bias field estimation.
### Added
- Variance scaled version of biasfield compound cost function.
- Add masked version of biasfield SSD cost function.
### Changed
- Reduce identical smoothing steps when estimating both warp fields and bias fields.
- Reduce unnecessarily high image sampling when calculating bias field for smoothed images.
- Take masking into account when normalising scalar image intensities.

## [0.3.1] 2022-12-21
- This release reduces GPU memory requirements.
- A "standard" MMORF run which changes from the full Gauss-Newton optimisation to the diagonal approximation method for warps < 4mm will now require under 10GB of RAM to run (down from 12GB).
### Changed
- Implemented *= and += operators for SparseDiagonalMatrixTiled to remove unnecessary copy operations.

## [0.3.0] 2021-09-20
- This release focuses on performance improvements without changing functionality, but as some core calculations have been moved from the CPU to the GPU the results may be very slightly different to previous versions (e.g. Jacobian determinants which differ in the fourth significant digit).
- Performance improvements depend on the number, type and resolution of images being registered, with typical speedups in the range of ~30%.
### Changed
- Conversion from real-world mm to native voxel coordinates when sampling image volumes moved to the GPU.
- Calculation of Jacobian determinants moved to the GPU.
- A single iteration only is performed for bias field estimation (rather than a maximum of five) as the cost function is exactly quadratic and therefore converges to its minimum in one step.

## [0.2.4] 2021-07-19
### Added
- Filenames of images passed to MMORF will be printed to the command line.

## [0.2.3] 2021-05-27
### Changed
- Bias field boundary condition changed from 0 to 1 to better reflect the multiplicative bias field model. This removes the "dark" edge effect in the bias field. The impact on results is minor, particularly when the images do not extend all the way to the volume edges.

### Fixed
- For very small voxel images (e.g. rodents at 0.1mm) regularisation sampling frequency was too low. Performance should now be consistent regardless of the scale of the images being registered.

## [0.2.2] 2021-05-20
- Internal development release only.

## [0.2.1] 2021-05-13
### Added
- Forward compatibility for Ampere architecture GPUs (PTX code for compute_70 is now included in the binary which allows JIT compilation on sm_80 cards such as A100).

## [0.2.0] 2021-03-18
### Added
- This file to keep track of changes in releases.
- `version` flag for keeping track of executable version.

### Changed
- `use_mask_mov_scalar` and `use_mask_mov_tensor` are now **required** options to match `use_mask_ref_scalar` and `use_mask_ref_tensor` functionality.
- config/multimodal_config.ini to reflect new `use_mask_mov_scalar` and `use_mask_mov_tensor` requirements.
- README.md to mention this file.
- README.md to include `use_mask_ref_tensor` and `use_mask_mov_tensor`.
- New version of mmorf.sif singularity image file.

### Fixed
- Corrected a bug where masks were being applied incorrectly, causing registration to stall. The issue was visible in MMORF logs as an apparent increase in the cost function for any proposed parameter update.

## [0.1.0] 2021-01-26
### Added
- mmorf.sif singularity image file for running MMORF.
- README.md file to explain parameter usage.
- config/multimodal_config.ini file as an example configuration file.