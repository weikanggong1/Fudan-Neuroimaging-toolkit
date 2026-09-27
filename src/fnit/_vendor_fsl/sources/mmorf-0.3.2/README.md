# Multi-Modal Registration Framework (MMORF)
MMORF is the new nonlinear registration tool from FSL.
What sets it apart from the previous tool, FNIRT, is the ability to simultaneously align multiple modalities, as well as a unique regularisation method.

If you have used FNIRT in the past, then many of MMORF's options may be familiar to you, however the difference are significant enough that you will need to read this user guide before using it.

# Table of Contents
[[_TOC_]]

# Usable Modalities
[[back to top]](#table-of-contents)

MMORF is capable of registering both scalar and tensor image pairs.
Tensor images _must_ be in the FSL `FDT` format, i.e.:

- Upper-triangular volumes ares stored
- Diffusivity in the x-direction is defined _radiologically_

If you have used FSL to preprocess your data then you shouldn't need to worry about this.

# Parameter Options
[[back to top]](#table-of-contents)

Here I will describe the parameters for which arguments must be provided when running MMORF.
Parameters described as "scalar" accept a single input argument.
Parameters described as "vector" accept multiple input arguments separated by spaces.
Parameters are either defined once per registration, or once per image pair.

All arguments passed via the command line (i.e., not via a config file) must follow the format:

`--parameter_name <argument>`

## **Warp Options**
[[back to top]](#table-of-contents)

### `warp_res_init`
- Initial warp resolution in mm (isotropic)
- `float`
- scalar
- defined once per registration

This defines the coarsest warp resolution in a multi-iteration registration.
For human data, `32` is a reasonable choice.

### `warp_scaling`
- Warp resolution scaling at each iteration
- `int`
- vector
- defined for all iterations once per registration

This defines how the warp resolution should be increased at each iteration.
For example, if:

`warp_res_init = 32`
`warp_scaling = 1 2 2`

then the registration will consist of 3 iterations with warp resolutions of 32mm, 16mm and 8mm respectively.

### `img_warp_space`
- 3D NIfTI volume defining the space in which the warp field will be calculated
- `string`
- scalar
- defined once per registration

Can be provided with or without a suffix (i.e., both `T1.nii.gz` and simply `T1` are acceptable).
The final warp field will be a 4D volume with the same extents and voxel dimensions as this image.

## **Scalar Image Options**
[[back to top]](#table-of-contents)

### `img_ref_scalar`
- 3D NIfTI volume for a scalar reference image
- `string`
- scalar
- defined once per image pair

A scalar modality (T1, T2, etc.) belonging to the reference subject.
Can be provided with or without a suffix (i.e., both `T1.nii.gz` and simply `T1` are acceptable).

### `img_mov_scalar`
- 3D NIfTI volume for a scalar reference image
- `string`
- scalar
- defined once per image pair

A scalar modality (T1, T2, etc.) belonging to the moving subject.
Can be provided with or without a suffix (i.e., both `T1.nii.gz` and simply `T1` are acceptable).

### `aff_ref_scalar`
- Affine transform in FSL `FLIRT` .mat format for a scalar reference image
- `string`
- scalar
- defined once per image pair

Must be provided with the file extension.
Should be the output of registering `img_ref_scalar` to `img_warp_space`.

### `aff_mov_scalar`
- Affine transform in FSL `FLIRT` .mat format for a scalar moving image
- `string`
- scalar
- defined once per image pair

Must be provided with the file extension.
Should be the output of registering `img_mov_scalar` to `img_warp_space`.

### `lambda_scalar`
- Lambda for weighting scalar image cost function at each iteration.
- `float`
- vector
- defined for all iterations once per image pair

Can be set to 1 as a good default.
If using _N_ highly-correlated modalities (e.g., T1 & T2), consider setting to 1/_N_ for each modality (e.g., 0.5, 0.5).

### `fwhm_ref_scalar`
- FWHM for Gaussian smoothing of scalar reference image at each iteration
- `float`
- vector
- defined for all iterations once per image pair

A good default is a quarter of the current warp resolution.
For example:

`warp_res_init = 32`
`warp_scaling = 1 2 2`
`fwhm_ref_scalar = 8 4 2`

### `fwhm_mov_scalar`
- FWHM for Gaussian smoothing of scalar moving image at each iteration
- `float`
- vector
- defined for all iterations once per image pair

A good default is a quarter of the current warp resolution.

## **Tensor Image Options**
[[back to top]](#table-of-contents)

### `img_ref_tensor`
- 3D NIfTI volume for a tensor reference image
- `string`
- scalar
- defined once per image pair

A tensor modality (specifically a DTI) belonging to the reference subject.
Can be provided with or without a suffix (i.e., both `T1.nii.gz` and simply `T1` are acceptable).

### `img_mov_tensor`
- 3D NIfTI volume for a tensor reference image
- `string`
- scalar
- defined once per image pair

A tensor modality (specifically a DTI) belonging to the moving subject.
Can be provided with or without a suffix (i.e., both `T1.nii.gz` and simply `T1` are acceptable).

### `aff_ref_tensor`
- Affine transform in FSL `FLIRT` .mat format for a tensor reference image
- `string`
- scalar
- defined once per image pair

Must be provided with the file extension.
Should be the output of registering `img_ref_tensor` to `img_warp_space`.

### `aff_mov_tensor`
- Affine transform in FSL `FLIRT` .mat format for a tensor moving image
- `string`
- scalar
- defined once per image pair

Must be provided with the file extension.
Should be the output of registering `img_mov_tensor` to `img_warp_space`.

### `lambda_tensor`
- Lambda for weighting tensor image cost function at each iteration.
- `float`
- vector
- defined for all iterations once per image pair

Can be set to 1 as a good default.
If using _N_ highly-correlated modalities (e.g., T1 & T2), consider setting to 1/_N_ for each modality (e.g., 0.5, 0.5).

### `fwhm_ref_tensor`
- FWHM for Gaussian smoothing of tensor reference image at each iteration
- `float`
- vector
- defined for all iterations once per image pair

A good default is a quarter of the current warp resolution.
For example:

`warp_res_init = 32`
`warp_scaling = 1 2 2`
`fwhm_ref_tensor = 8 4 2`

### `fwhm_mov_tensor`
- FWHM for Gaussian smoothing of tensor moving image at each iteration
- `float`
- vector
- defined for all iterations once per image pair

A good default is a quarter of the current warp resolution.

## **Mask Options**
[[back to top]](#table-of-contents)

### `use_implicit_mask`
- Treat zeros in images as missing data during smoothing
- `int`: `0` = False, `1` = True
- scalar
- defined once per registration

Prevents darkening of the edges of images.
My experience is that in most cases this should just be set to `0`.

### `mask_ref_scalar`
- 3D NIfTI volume for a soft mask in scalar reference image space
- `string`
- scalar
- defined once per image pair

The cost function for the associated image pair is multiplied voxelwise by this mask.
Can be provided with or without a suffix (i.e., both `mask.nii.gz` and simply `mask` are acceptable).
If you do not with to supply a mask image, use the keyword `NULL`

### `mask_mov_scalar`
- 3D NIfTI volume for a soft mask in scalar moving image space
- `string`
- scalar
- defined once per image pair

The cost function for the associated image pair is multiplied voxelwise by this mask.
Can be provided with or without a suffix (i.e., both `mask.nii.gz` and simply `mask` are acceptable).
If you do not with to supply a mask image, use the keyword `NULL`

### `mask_ref_tensor`
- 3D NIfTI volume for a soft mask in tensor reference image space
- `string`
- scalar
- defined once per image pair

The cost function for the associated image pair is multiplied voxelwise by this mask.
Can be provided with or without a suffix (i.e., both `mask.nii.gz` and simply `mask` are acceptable).
If you do not with to supply a mask image, use the keyword `NULL`

### `mask_mov_tensor`
- 3D NIfTI volume for a soft mask in tensor moving image space
- `string`
- scalar
- defined once per image pair

The cost function for the associated image pair is multiplied voxelwise by this mask.
Can be provided with or without a suffix (i.e., both `mask.nii.gz` and simply `mask` are acceptable).
If you do not with to supply a mask image, use the keyword `NULL`

### `use_mask_ref_scalar`
- Apply `mask_ref_scalar` at each iteration
- `int`: `0` = False, `1` = True
- vector
- defined for all iterations once per image pair

Whether or not the `mask_ref_scalar` should be used at a particular iteration.

### `use_mask_mov_scalar`
- Apply `mask_mov_scalar` at each iteration
- `int`: `0` = False, `1` = True
- vector
- defined for all iterations once per registration

Whether or not the `mask_ref_scalar` should be used at a particular iteration.

## **Bias Field Options**
[[back to top]](#table-of-contents)

### `estimate_bias`
- Estimate a multiplicative bias field for a scalar image pair
- `int`: `0` = False, `1` = True
- vector
- defined for all iterations once per image pair

Whether or not to update the bias field estimate at the start of a particular iteration.
Note that this is only applicable to scalar image pairs.

### `bias_res_init`
- Initial bias field resolution in mm (isotropic)
- `float`
- scalar
- defined once per registration

This defines the resolution of the estimated multiplicative bias field.
For human data, `32` is a reasonable choice.

### `lambda_bias_reg`
- Lambda for weighting the Bending Energy regularisation.
- `float`
- vector
- defined for all iterations once per image pair

Control how smoothly the estimated bias field varies.
A sensible default for this is a constant `1e9` for all iterations.

## **Regularisation Options**
[[back to top]](#table-of-contents)

### `lambda_reg`
- Lambda for weighting the warp regularisation.
- `float`
- vector
- defined for all iterations once per registration

This sets the regularisation at each iteration, and is independent of the number of modalities used.
The regularisation method is almost always the [SPRED](https://doi.org/10.1016/j.neuroimage.2020.116962) penalty, which penalises the log of the singular values of the local Jacobian field.
I say _almost_ always because of a slight quirk - for a transformation of _exactly_ zero, the Hessian of the SPRED penalty is undefined.
Therefore, for the first iteration of the registration, MMORF is always run with Bending Energy regularisation.

**NB!!!** The upshot of this is that the recommendation is to always run two iterations at your coarsest resolution, which will result in one iteration with Bending Energy that acts as an initialisation for one iteration with SPRED regularisation.

For example:

`warp_res_init = 32`
`warp_scaling = 1 1 2`
`lambda_reg = 4.0e5 3.7e-1 3.1e-1`

would run one iteration at 32mm warp resolution with a Bending Energy regularisation of `4.0e5`, one iteration at 32mm warp resolution with a SPRED regularisation of `3.7e-1`, and one iteration at 16mm warp resolution with a SPRED regularisation of `3.1e-1`.
Note that the regularisation scale is very different between the two methods and they are therefore not directly comparable.
This behaviour is likely to change in the future, and such changes will be indicated prominently.

## **Output Options**
[[back to top]](#table-of-contents)

### `warp_out`
- basename of final warp field
- `string`
- scalar
- defined once per registration

Must be supplied _without_ a suffix.
The final warp field will be a 4D volume with the same extents and voxel dimensions as `img_warp_space`.

### `jac_det_out`
- basename of Jacobian determinant of final warp field
- `string`
- scalar
- defined once per registration

Must be supplied _without_ a suffix.
Will output a 3D volume with the same extents and voxel dimensions as `img_warp_space`.

### `bias_out`
- basename of bias field for scalar image pairs
- `string`
- scalar
- defined once per registration

Must be supplied _without_ a suffix.
Will output a 3D volume with the same extents and voxel dimensions as `img_warp_space`.
One volume will be output for each scalar image pair.
Argument may be `NULL` if not required.

## **Optimiser Options**
[[back to top]](#table-of-contents)

At each iteration of the registration, a number of optimisation iterations are performed.

### `hires`
- warp resolution (in mm isotropic) below which a low-memory optimisation strategy will be used
- `float`
- scalar
- defined once per registration

As warp resolution increases, memory requirements for Levenberg Marquardt (LM) optimisation increases.
Beyond a certain resolution, the memory requirements may exceed the GPU's maximum available RAM.
At this point, MMORF switches to an optimisation strategy which requires more iterations for convergence, but has much lower memory requirements, such as Majorise Minimisation (MM) or Scaled Conjugate Gradient (SCG).

For example, with:

`warp_res_init = 32`
`warp_scaling = 1 1 2 2 2`
`hires = 6`

the warp resolution at each iteration is 32, 32, 16, 8 and 4mm respectively.
With this `hires` setting, only the 4mm iteration will use the low memory optimisation method.

What setting to use will depend on the field of view (FOV) of `img_warp_space`, and the amount of RAM on your GPU.
For GPUs with 12GB or more of RAM, a good default to use for human data is 4mm for an FOV similar to the MNI-152 template, or 8mm for larger FOVs.

### `optimiser_lowres`
- which optimiser to use for coarser warp resolutions (> `hires`)
- `enum`: in {`LM`, `MM`, `SCG`} -> {`0`, `1`, `2`}
- scalar
- defined once per registration

Choose between Levenberg Marquardt (LM), Majorise Minimisation (MM) or Scaled Conjugate Gradient for coarse resolution warps with resolution larger than the `hires` setting.
LM is recommended.

### `optimiser_hires`
- which optimiser to use for finer warp resolutions (< `hires`)
- `enum`: in {`LM`, `MM`, `SCG`} -> {`0`, `1`, `2`}
- scalar
- defined once per registration

Choose between Levenberg Marquardt (LM), Majorise Minimisation (MM) or Scaled Conjugate Gradient for fine resolution warps with resolution smaller than the `hires` setting.
MM is recommended.

### `optimiser_max_it_lowres`
- number of optimisation iterations to run for each coarse resolution warp iteration
- `int`
- scalar
- defined once per registration

Applies to warp resolutions larger than `hires`.
If using the recommended `optimiser_lowres = LM` is used, then a good default is between `5` and `10`.

### `optimiser_max_it_hires`
- number of optimisation iterations to run for each fine resolution warp iteration
- `int`
- scalar
- defined once per registration

Applies to warp resolutions smaller than `hires`.
If using the recommended `optimiser_hires = MM` is used, then a good default is between `5` and `10`.
If using `optimiser_hires = SCG` then a value between `10` and `20` is advised.

### `optimiser_rel_tol_lowres`
- relative tolerance for testing convergence of the optimiser for each coarse resolution warp iteration
- `float`
- scalar
- defined once per registration

Applies to warp resolutions larger than `hires`.
Allows early stopping of optimisation before reaching `optimiser_max_it_lowres` if the decrease in the cost function penalty is less than this value relative to the previous cost.
A good default is `1e-3`.

### `optimiser_rel_tol_hires`
- relative tolerance for testing convergence of the optimiser for each fine resolution warp iteration
- `float`
- scalar
- defined once per registration

Applies to warp resolutions smaller than `hires`.
Allows early stopping of optimisation before reaching `optimiser_max_it_hires` if the decrease in the cost function penalty is less than this value relative to the previous cost.
A good default is `1e-3`.

## **Solver Options**
[[back to top]](#table-of-contents)

Each iteration of the optimisation makes use of an iterative linear solver to calculate the update to the warp field.

### `solver_max_it_lowres`
- number of solver iterations to run for each coarse resolution optimiser iteration
- `int`
- scalar
- defined once per registration

Applies to warp resolutions larger than `hires`.
A good default is `100`.

### `solver_max_it_hires`
- number of solver iterations to run for each fine resolution optimiser iteration
- `int`
- scalar
- defined once per registration

Applies to warp resolutions smaller than `hires`.
A good default is `100`

### `solver_rel_tol_lowres`
- relative tolerance for testing convergence of the solver for each coarse resolution optimiser iteration
- `float`
- scalar
- defined once per registration

Applies to warp resolutions larger than `hires`.
Allows early stopping of the solver before reaching `solver_max_it_lowres` if the decrease in the solver error is less than this value relative to the previous error.
A good default is `1e-3`.

### `solver_rel_tol_hires`
- relative tolerance for testing convergence of the solver for each fine resolution optimiser iteration
- `float`
- scalar
- defined once per registration

Applies to warp resolutions smaller than `hires`.
Allows early stopping of the solver before reaching `solver_max_it_hires` if the decrease in the solver error is less than this value relative to the previous error.
A good default is `1e-3`.

## **Miscellaneous Options**
[[back to top]](#table-of-contents)

### `config`
- config file in `.ini` format containing some or all of the options required to run MMORF
- `string`
- scalar
- defined once per registration

Must be the full filename, and must have the `.ini` suffix.
A handy way of configuring MMORF without having to specify all arguments on the command line.

# Config Files
[[back to top]](#table-of-contents)

By far the most convenient and reproducible method of running MMORF is through the use of `config` files.
Config files use the `.ini` suffix and each line is either a command line option or a comment.
Comment lines begin with a semicolon (`;`) symbol.
Command lines are of the form:

`option_name = option_value_1 option_value_2 option_value_3`

The order in which options are entered is important, and should correspond to the order in which the volumes for each modality is entered.
The easiest way of understanding this is probably with a few simple examples.
In each case the warp space is the T1 image for the reference subject.
These config files assume that affine matrices (in FSL `FLIRT` format) are available for each reference and moving image.

### **Unimodal Scalar Config Example**
[[back to top]](#table-of-contents)

```
; This is a comment at the start of the config file
; Parameters defining the overall registration:

warp_res_init           = 32
warp_scaling            = 1 1 2 2 2
img_warp_space          = ../data/vols/ref/t1/t1
lambda_reg              = 4.0e5 3.7e-1 3.1e-1 2.6e-1 2.2e-1
hires                   = 6
optimiser_max_it_lowres = 5
optimiser_max_it_hires  = 5

; Parameters relating to first scalar image pair

img_ref_scalar      = ../data/vols/ref/t1/t1
img_mov_scalar      = ../data/vols/mov/t1/t1
aff_ref_scalar      = ../data/mats/identity.mat
aff_mov_scalar      = ../data/mats/mov_to_ref_t1.mat
use_implicit_mask   = 0
use_mask_ref_scalar = 0 0 0 0 0
mask_ref_scalar     = NULL
mask_mov_scalar     = NULL
fwhm_ref_scalar     = 8.0 8.0 4.0 2.0 1.0
fwhm_mov_scalar     = 8.0 8.0 4.0 2.0 1.0
lambda_scalar       = 1 1 1 1 1
estimate_bias       = 0
bias_res_init       = 32
lambda_bias_reg     = 1e9 1e9 1e9 1e9 1e9
```

### **Multimodal Scalar Only Config Example**
[[back to top]](#table-of-contents)

```
; This is a comment at the start of the config file
; Parameters defining the overall registration:

warp_res_init           = 32
warp_scaling            = 1 1 2 2 2
img_warp_space          = ../data/vols/ref/t1/t1
lambda_reg              = 4.0e5 3.7e-1 3.1e-1 2.6e-1 2.2e-1
hires                   = 6
optimiser_max_it_lowres = 5
optimiser_max_it_hires  = 5

; Parameters relating to first scalar image pair

img_ref_scalar      = ../data/vols/ref/t1/t1
img_mov_scalar      = ../data/vols/mov/t1/t1
aff_ref_scalar      = ../data/mats/identity.mat
aff_mov_scalar      = ../data/mats/mov_t1_to_ref_t1.mat
use_implicit_mask   = 0
use_mask_ref_scalar = 0 0 0 0 0
mask_ref_scalar     = NULL
mask_mov_scalar     = NULL
fwhm_ref_scalar     = 8.0 8.0 4.0 2.0 1.0
fwhm_mov_scalar     = 8.0 8.0 4.0 2.0 1.0
lambda_scalar       = 1 1 1 1 1
estimate_bias       = 0
bias_res_init       = 32
lambda_bias_reg     = 1e9 1e9 1e9 1e9 1e9

; Parameters relating to second scalar image pair

img_ref_scalar      = ../data/vols/ref/t2/t2
img_mov_scalar      = ../data/vols/mov/t2/t2
aff_ref_scalar      = ../data/mats/ref_t2_to_t1.mat
aff_mov_scalar      = ../data/mats/mov_t2_ref_t1.mat
use_implicit_mask   = 0
use_mask_ref_scalar = 0 0 0 0 0
mask_ref_scalar     = NULL
mask_mov_scalar     = NULL
fwhm_ref_scalar     = 8.0 8.0 4.0 2.0 1.0
fwhm_mov_scalar     = 8.0 8.0 4.0 2.0 1.0
lambda_scalar       = 1 1 1 1 1
estimate_bias       = 0
bias_res_init       = 32
lambda_bias_reg     = 1e9 1e9 1e9 1e9 1e9
```
### **Multimodal Scalar & Tensor Config Example**
[[back to top]](#table-of-contents)

```
; This is a comment at the start of the config file
; Parameters defining the overall registration:

warp_res_init           = 32
warp_scaling            = 1 1 2 2 2
img_warp_space          = ../data/vols/ref/t1/t1
lambda_reg              = 4.0e5 3.7e-1 3.1e-1 2.6e-1 2.2e-1
hires                   = 6
optimiser_max_it_lowres = 5
optimiser_max_it_hires  = 5

; Parameters relating to first scalar image pair

img_ref_scalar      = ../data/vols/ref/t1/t1
img_mov_scalar      = ../data/vols/mov/t1/t1
aff_ref_scalar      = ../data/mats/identity.mat
aff_mov_scalar      = ../data/mats/mov_t1_to_ref_t1.mat
use_implicit_mask   = 0
use_mask_ref_scalar = 0 0 0 0 0
mask_ref_scalar     = NULL
mask_mov_scalar     = NULL
fwhm_ref_scalar     = 8.0 8.0 4.0 2.0 1.0
fwhm_mov_scalar     = 8.0 8.0 4.0 2.0 1.0
lambda_scalar       = 1 1 1 1 1
estimate_bias       = 0
bias_res_init       = 32
lambda_bias_reg     = 1e9 1e9 1e9 1e9 1e9

; Parameters relating to first tensor image pair

img_ref_tensor      = ../data/vols/ref/dti/dti
img_mov_tensor      = ../data/vols/mov/dti/dti
aff_ref_tensor      = ../data/mats/ref_dti_to t1.mat
aff_mov_tensor      = ../data/mats/mov_dti_to_ref_t1.mat
use_mask_ref_tensor = 0 0 0 0 0
mask_ref_tensor     = NULL
mask_mov_tensor     = NULL
fwhm_ref_tensor     = 8.0 8.0 4.0 2.0 1.0
fwhm_mov_tensor     = 8.0 8.0 4.0 2.0 1.0
lambda_tensor       = 1 1 1 1 1
```


# Installing MMORF
[[back to top]](#table-of-contents)

MMORF is installed as part of [FSL 6.1.0](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/). You can also install MMORF independently of FSL using [`conda`](https://docs.conda.io/en/latest/). Assuming that you have [miniconda](https://docs.conda.io/en/latest/miniconda.html) or equivalent installed, you can create a conda environment with MMORF like so:

```
conda create -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ -c conda-forge -n mmorf fsl-mmorf-cuda-10.2
```

Note that the MMORF conda package is named `fsl-mmorf-cuda-10.2`.


# Compiling MMORF
[[back to top]](#table-of-contents)

To compile MMORF, you will need:

 - FSL 6.1.0 or newer (or a conda environment with `fsl-mmorf` installed, as outlined above).
 - A CUDA Toolkit installation

1. Ensure that your environment is configured correctly:
   - the `nvcc` executable from your CUDA toolkit is available on your `$PATH` variable, e.g.:
      ```
      export PATH=<cuda-toolkit>/bin/:$PATH
      ```

   - FSL is configured in your environment. e.g.:
      ```
      export FSLDIR=<fsl-installation>
      source $FSLDIR/etc/fslconf/fsl-devel.sh
      ```

2. Install make and GCC. Note that the required GCC version differs depending on which CUDA Toolkit version you are compiling against. More information on this is available at https://git.fmrib.ox.ac.uk/fsl/conda/docs/-/blob/master/local_development.md. A convenient method of installing and maintaining separate GCC versions is to use separate conda environments, e.g.:
    ```
    $FSLDIR/bin/conda create -n gcc8 -y make "gxx_linux-64 8"
    source $FSLDIR/bin/activate gcc8
    ```

3. Clone the MMORF git repository. MMORF uses git submodules for some dependencies - [cusplibrary](https://github.com/cusplibrary/cusplibrary) and [CubicInterpolationCUDA](https://github.com/DannyRuijters/CubicInterpolationCUDA/):
    ```
    git clone https://git.fmrib.ox.ac.uk/flange/MMORF
    cd MMORF
    git submodule update --init --recursive
    ```

4. Compile MMORF. You can set the `GENCODEFLAGS` variable to override the compute capability options, and you can set the `CUDA_STATIC` variable to statically link against the CUDA runtime:
    ```
    # dynamically linked, wide range of compute capabilities
    make
    # statically linked, wide range of compute capabilities
    make CUDA_STATIC=1
    # dynamically linked, specific compute capability
    make GENCODEFLAS="-gencode arch=compute_72,code=sm_72"
    ```


# Singularity and MMORF
[[back to top]](#table-of-contents)

In order to use MMORF you will require [Singularity](https://sylabs.io/) to be installed on your system.
Singularity is a containerised software solution, similar to Docker but with stronger security.
It is therefore ideal for use on HPC clusters and personal machines.
The reason why Singularity is required is as follows:

MMORF requires a CUDA-capable Nvidia GPU to run.
A constant headache with using CUDA-accelerated software is that the version of CUDA against which the software was compiled must match the version installed on the machine where it runs.
As such, you will often find that there are multiple versions of CUDA compiled software available, and that a small upgrade to your system can break compatibility with software that worked perfectly well yesterday.

Singularity allows _all_ of the CUDA-related dependencies to be encapsulated within the container.
The only requirement is that the correct driver and a basic set of libraries for your GPU are installed on the host machine (see [here](https://sylabs.io/guides/3.7/user-guide/gpu.html) for more details).

The main MMORF application is packaged in a file called `mmorf.sif` -  where the `.sif` extension refers to the Singularity Image Format.

## Running MMORF
[[back to top]](#table-of-contents)

If all of the files you are working with exist somewhere in your user's home directory tree, then running MMORF is as simple as:

`singularity run --nv mmorf.sif --option_1 <argument_1> --option_2 <argument_2.1> <argument_2.2>`

etc.
If your are making use of a config file called `my_config.ini`, then your call may look something like:

`singularity run --nv mmorf.sif --config my_config.ini`

The `--nv` is important as it lets Singularity know that you want to pass GPU access through to the container

## Data in Shared Storage
[[back to top]](#table-of-contents)

A minor complication arises when the data on which MMORF will operate do not exist within your user's home directory tree.
This is because, by default, Singularity only makes the user who executes the image's home directory visible to the application (known as binding).

This can be overcome using [user-defined bind paths](https://sylabs.io/guides/3.7/user-guide/bind_paths_and_mounts.html#user-defined-bind-paths).
These can be defined in two ways.

1. Passing the `--bind` flag when executing `singularity run`, e.g.:
`singularity run --nv --bind host_dir_1,host_dir_2,host_dir_3 mmorf.sif --config my_config.ini`
1. Setting the `SINGULARITY_BIND` environment variable before executing `singularity run`, e.g.:
`export SINGULARITY_BIND="host_dir_1,host_dir_2,host_dir_3"`
`singularity run --nv mmorf.sif --config my_config.ini`

In both cases, multiple bind paths are separated by _commas without spaces_.
Importantly, make certain that both your input and your output directories are correctly bound.

# "Gotchas"
[[back to top]](#table-of-contents)

There are a few aspects of using MMORF which might be slightly unintuitive, and I will attempt to highlight them specifically here.
Some of these aspects are due to moving from single to multiple modalities, and are therefore unavoidable design decisions.
Others are due to MMORF still being a beta release, and are therefore idiosyncrasies which I will endeavour to remove in future versions.

## **Unlikely to Change**
1. An affine matrix must be supplied for every image, for both the reference and the moving subject.
This is because the reference image and warp space are not _necessarily_ the same.
This is a consequence of the fact that different modality images from the same subject might not be in the same space.
If the reference image _is_ in the same space as the warp, then an identity matrix must be used as the input parameter.

## **Likely to Change in the Future**
For a given configuration to be valid, a number of parameters must have the correct dimensionality (e.g., if you want to run 6 iterations then you need 6 smoothing values per volume).
MMORF assumes that the `warp_scaling` parameter is always correct, and therefore if other parameters have a different length, then MMORF will not run.
Unfortunately, this is not currently tested when parsing the command line inputs, but rather when the "register images" function is called.
What this means for the user is that some parameters become compulsory _even if they will have no effect on the registration_.
I intend to amend this behaviour in the future, but for now the following rules apply:

1. Every `lambda_bias_reg` argument length must match `warp_scaling` argument length _even if_ `estimate_bias` is set to `0`.
1. Every `use_mask_ref_scalar` argument length must match `warp_scaling` argument length _even if_ `mask_ref_scalar` is set to `NULL`.
1. Every `use_mask_ref_tensor` argument length must match `warp_scaling` argument length _even if_ `mask_ref_tensor` is set to `NULL`.
