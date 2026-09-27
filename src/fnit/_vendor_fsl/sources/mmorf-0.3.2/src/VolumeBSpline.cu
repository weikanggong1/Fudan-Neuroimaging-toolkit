///
/// \file
/// \brief Implementation of MMORF::Volume based on the CubicInterpolationCUDA code by Danny
///        Ruijters (see: https://github.com/DannyRuijters/CubicInterpolationCUDA)
/// \details This volume resides exclusively on the GPU and allows for rapid cubic
///          interpolation of values within the volume. It uses regular cubic B-splines, and
///          stores the spline coefficients as a 3D device texture, but this shouldn't affect
///          the interface.
/// \author Frederik Lange
/// \date April 2018
/// \copyright FMRIB Copyright Licence

#include "MmorfMemory.h"
#include "VolumeBSpline.cuh"

#include "helper_cuda.h"

#include "newimage/newimageall.h"

#include <cutil.h>
#include <memcpy.cu>
#include <cubicPrefilter3D.cu>
#include <cubicTex3D.cu>

#include <armadillo>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <cmath>

// VERY ANNOYING GLOBAL TEXTURE VARIABLE
// NB!NB!NB! FIND A WAY TO USE TEXTURE REFERENCES RATHER NB!NB!NB!
texture<float, 3, cudaReadModeElementType> coeffs;  //3D texture

/// Multi-Modal Registration Framework
namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// __global__ functions
////////////////////////////////////////////////////////////////////////////////
  __global__ void kernel_cubic_bspline_sample(
      // Input
      const unsigned int n_samples,
      const float* __restrict__ x_coords,
      const float* __restrict__ y_coords,
      const float* __restrict__ z_coords,
      // Output
      float* __restrict__ samples)
  {
    auto sample_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample_id < n_samples){
      auto coord = make_float3(
          x_coords[sample_id] + 0.5,
          y_coords[sample_id] + 0.5,
          z_coords[sample_id] + 0.5);
      samples[sample_id] = cubicTex3D(coeffs, coord);
    }
  }

  __global__ void kernel_cubic_bspline_sample_with_affine(
      // Input
      const unsigned int n_samples,
      const float* __restrict__ x_coords,
      const float* __restrict__ y_coords,
      const float* __restrict__ z_coords,
      const float* __restrict__ affine,
      // Output
      float* __restrict__ samples)
  {
    auto sample_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample_id < n_samples){
      auto x_transformed = x_coords[sample_id]*affine[0]
                         + y_coords[sample_id]*affine[1]
                         + z_coords[sample_id]*affine[2]
                         + affine[3];
      auto y_transformed = x_coords[sample_id]*affine[4]
                         + y_coords[sample_id]*affine[5]
                         + z_coords[sample_id]*affine[6]
                         + affine[7];
      auto z_transformed = x_coords[sample_id]*affine[8]
                         + y_coords[sample_id]*affine[9]
                         + z_coords[sample_id]*affine[10]
                         + affine[11];
      auto coord = make_float3(
          x_transformed + 0.5,
          y_transformed + 0.5,
          z_transformed + 0.5);
      samples[sample_id] = cubicTex3D(coeffs, coord);
    }
  }

  __global__ void kernel_cubic_bspline_sample_derivative(
      // Input
      const int dimension,
      const unsigned int n_samples,
      const float* __restrict__ x_coords,
      const float* __restrict__ y_coords,
      const float* __restrict__ z_coords,
      // Output
      float* __restrict__ samples)
  {
    auto sample_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample_id < n_samples){
      auto coord = make_float3(
          x_coords[sample_id] + 0.5,
          y_coords[sample_id] + 0.5,
          z_coords[sample_id] + 0.5);
      switch(dimension){
        case 0 : samples[sample_id] = cubicTex3D_1st_derivative_x(coeffs, coord);
                 break;
        case 1 : samples[sample_id] = cubicTex3D_1st_derivative_y(coeffs, coord);
                 break;
        case 2 : samples[sample_id] = cubicTex3D_1st_derivative_z(coeffs, coord);
                 break;
      }
    }
  }

  __global__ void kernel_cubic_bspline_sample_derivative_with_affine(
      // Input
      const unsigned int n_samples,
      const float* __restrict__ x_coords,
      const float* __restrict__ y_coords,
      const float* __restrict__ z_coords,
      const float* __restrict__ affine,
      // Output
      float* __restrict__ dx_samples,
      float* __restrict__ dy_samples,
      float* __restrict__ dz_samples)
  {
    auto sample_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample_id < n_samples){
      auto x_transformed = x_coords[sample_id]*affine[0]
                         + y_coords[sample_id]*affine[1]
                         + z_coords[sample_id]*affine[2]
                         + affine[3];
      auto y_transformed = x_coords[sample_id]*affine[4]
                         + y_coords[sample_id]*affine[5]
                         + z_coords[sample_id]*affine[6]
                         + affine[7];
      auto z_transformed = x_coords[sample_id]*affine[8]
                         + y_coords[sample_id]*affine[9]
                         + z_coords[sample_id]*affine[10]
                         + affine[11];
      auto coord = make_float3(
          x_transformed + 0.5,
          y_transformed + 0.5,
          z_transformed + 0.5);
      auto dx = cubicTex3D_1st_derivative_x(coeffs, coord);
      auto dy = cubicTex3D_1st_derivative_y(coeffs, coord);
      auto dz = cubicTex3D_1st_derivative_z(coeffs, coord);

      dx_samples[sample_id] = dx*affine[0]
                            + dy*affine[4]
                            + dz*affine[8];
      dy_samples[sample_id] = dx*affine[1]
                            + dy*affine[5]
                            + dz*affine[9];
      dz_samples[sample_id] = dx*affine[2]
                            + dy*affine[6]
                            + dz*affine[10];
    }
  }
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  /// Implementation class for VolumeBSpline
  class VolumeBSpline::Impl
  {
    public:
      //////////////////////////////////////////////////
      // Public functions
      //////////////////////////////////////////////////
      Impl(
          const int                 id,
          const std::vector<int>&   dimensions,
          const std::vector<float>& resolution,
          const std::vector<float>& origin,
          const bool                clamp_border)
        : id_(id)
        , resolution_(3)
        , dimensions_(dimensions)
        , mm_to_vox_(dimensions.size() + 1, dimensions.size() + 1, arma::fill::zeros)
        , mm_to_vox_vec_(16, 0.0f)
        , vox_to_mm_vec_(16, 0.0f)
        , clamp_border_(clamp_border)
      {
        resolution_ = resolution;
        // Initialise mm_to_vox_ to correct values
        // Note that we first create a "vox_to_mm" matrix, and then invert it. It may be
        // better to just construct with 1/resolution values though for speed.
        for (auto i_dim = size_t(0);
            i_dim < resolution.size();
            ++i_dim){
          //mm_to_vox_(i_dim,i_dim) = resolution[i_dim];
          mm_to_vox_(i_dim,i_dim) = 1.0f/resolution[i_dim];
        }
        mm_to_vox_(dimensions.size(), dimensions.size()) = 1.0;
        for (auto i_dim = size_t(0);
            i_dim < origin.size();
            ++i_dim){
          mm_to_vox_(i_dim,origin.size()) += origin[i_dim];
        }
        arma::fmat vox_to_mm = mm_to_vox_.i();
        for (auto i = 0; i < 4; ++i){
          for (auto j = 0; j < 4; ++j){
            mm_to_vox_vec_[i*4 + j] = mm_to_vox_(i,j);
            vox_to_mm_vec_[i*4 + j] = vox_to_mm(i,j);
          }
        }
        // Initialise coefficients to 0
        auto coef_sz = 1;
        for (auto dim : dimensions_){
          coef_sz *= dim;
        }
        coefficients_ = std::vector<float>(coef_sz,0.0);
      }

      /// Construct by passing in a NEWIMAGE::volume object
      Impl(
          const int                       id,
          const NEWIMAGE::volume<float>&  vol,
          const bool                      clamp_border)
        : id_(id)
        , resolution_(3)
        , dimensions_(3)
        , clamp_border_(clamp_border)
        , mm_to_vox_vec_(16, 0.0f)
        , vox_to_mm_vec_(16, 0.0f)
      {
        // Set resolution
        resolution_[0] = vol.xdim();
        resolution_[1] = vol.ydim();
        resolution_[2] = vol.zdim();
        // Initialise dimensions
        dimensions_[0] = vol.xsize();
        dimensions_[1] = vol.ysize();
        dimensions_[2] = vol.zsize();
        // Initialise mm_to_vox_
        mm_to_vox_ = extract_newimage_mm_to_vox_(vol);
        arma::fmat vox_to_mm = mm_to_vox_.i();
        for (auto i = 0; i < 4; ++i){
          for (auto j = 0; j < 4; ++j){
            mm_to_vox_vec_[i*4 + j] = mm_to_vox_(i,j);
            vox_to_mm_vec_[i*4 + j] = vox_to_mm(i,j);
          }
        }
        // Copy data from NEWIMAGE::volume to host vector
        coefficients_ = extract_newimage_data_(vol);
#pragma omp critical
{
        values_to_coefficients_(dimensions_, coefficients_);
}
      }

      /// Construct by passing in a filename
      Impl(
          const int   id,
          std::string vol_name,
          const bool  clamp_border)
        : id_(id)
        , resolution_(3)
        , dimensions_(3)
        , clamp_border_(clamp_border)
        , mm_to_vox_vec_(16, 0.0f)
        , vox_to_mm_vec_(16, 0.0f)
      {
        // Read in nifti file
        NEWIMAGE::volume<float> vol;
        NEWIMAGE::read_volume(vol,vol_name);
        // Set resolution
        resolution_[0] = vol.xdim();
        resolution_[1] = vol.ydim();
        resolution_[2] = vol.zdim();
        // Set dimensions
        dimensions_[0] = vol.xsize();
        dimensions_[1] = vol.ysize();
        dimensions_[2] = vol.zsize();
        // Initialise mm_to_vox
        mm_to_vox_ = extract_newimage_mm_to_vox_(vol);
        arma::fmat vox_to_mm = mm_to_vox_.i();
        for (auto i = 0; i < 4; ++i){
          for (auto j = 0; j < 4; ++j){
            mm_to_vox_vec_[i*4 + j] = mm_to_vox_(i,j);
            vox_to_mm_vec_[i*4 + j] = vox_to_mm(i,j);
          }
        }
        // Copy data from NEWIMAGE::volume to host vector
        coefficients_ = extract_newimage_data_(vol);
#pragma omp critical
{
        values_to_coefficients_(dimensions_, coefficients_);
}
      }

      /// Construct from NEWIMAGE::volume and smooth with Gaussian (defined by FWHM mm)
      Impl(
          const int                       id,
          const NEWIMAGE::volume<float>&  vol,
          const float                     smooth_fwhm_mm,
          const bool                      implicit_mask,
          const bool                      clamp_border,
          const bool                      debug)
        : id_(id)
        , resolution_(3)
        , dimensions_(3)
        , clamp_border_(clamp_border)
        , mm_to_vox_vec_(16, 0.0f)
        , vox_to_mm_vec_(16, 0.0f)
      {
        /// \todo Replace assert with exception
        assert (smooth_fwhm_mm >= 0.0f);
        // Convert fwhm to sigma
        auto smooth_sigma_mm = smooth_fwhm_mm/2.355f;
        // Smooth volume
        auto mask = NEWIMAGE::volume<float>(vol.xsize(), vol.ysize(), vol.zsize());
        mask.setdims(vol.xdim(), vol.ydim(), vol.zdim());
        if (implicit_mask){
          for (auto k = 0; k < vol.zsize(); ++k){
            for (auto j = 0; j < vol.ysize(); ++j){
              for (auto i = 0; i < vol.xsize(); ++i){
                mask(i,j,k) = fabsf(vol(i,j,k)) <= std::numeric_limits<float>::epsilon() ?
                  0.0f : 1.0f;
              }
            }
          }
        }
        else{
          mask = 1.0f;
        }
        auto vol_smoothed = NEWIMAGE::smooth(vol, smooth_sigma_mm);
        auto mask_smoothed = NEWIMAGE::smooth(mask, smooth_sigma_mm);
        for (auto k = 0; k < vol_smoothed.zsize(); ++k){
          for (auto j = 0; j < vol_smoothed.ysize(); ++j){
            for (auto i = 0; i < vol_smoothed.xsize(); ++i){
              vol_smoothed(i,j,k) = mask(i,j,k) > 0.5f ?
                vol_smoothed(i,j,k)/mask_smoothed(i,j,k) : 0.0f;
            }
          }
        }
        // Set resolution
        resolution_[0] = vol_smoothed.xdim();
        resolution_[1] = vol_smoothed.ydim();
        resolution_[2] = vol_smoothed.zdim();
        // Initialise dimensions
        dimensions_[0] = vol_smoothed.xsize();
        dimensions_[1] = vol_smoothed.ysize();
        dimensions_[2] = vol_smoothed.zsize();
        // Initialise mm_to_vox_
        mm_to_vox_ = extract_newimage_mm_to_vox_(vol_smoothed);
        arma::fmat vox_to_mm = mm_to_vox_.i();
        for (auto i = 0; i < 4; ++i){
          for (auto j = 0; j < 4; ++j){
            mm_to_vox_vec_[i*4 + j] = mm_to_vox_(i,j);
            vox_to_mm_vec_[i*4 + j] = vox_to_mm(i,j);
          }
        }
        // Copy data from NEWIMAGE::volume to host vector
        coefficients_ = extract_newimage_data_(vol_smoothed);
#pragma omp critical
{
        values_to_coefficients_(dimensions_, coefficients_);
}
        if (debug){
          // Create folder
          auto mkdir_command = "mkdir -p " + debug_dir_;
          auto tmp_sysval = std::system(mkdir_command.c_str());
          // Save grad
          NEWIMAGE::write_volume(vol_smoothed, debug_dir_ + "vol_" + std::to_string(id));
        }
      }

      /// Construct from nifti image file and smooth with Gaussian (defined by FWHM mm)
      Impl(
          const int   id,
          std::string vol_name,
          const float smooth_fwhm_mm,
          const bool  implicit_mask,
          const bool  clamp_border,
          const bool  debug)
        : id_(id)
        , resolution_(3)
        , dimensions_(3)
        , clamp_border_(clamp_border)
        , mm_to_vox_vec_(16, 0.0f)
        , vox_to_mm_vec_(16, 0.0f)
      {
        /// \todo Replace assert with exception
        assert (smooth_fwhm_mm >= 0.0f);
        // Read in nifti file
        NEWIMAGE::volume<float> vol;
        NEWIMAGE::read_volume(vol,vol_name);
        // Convert fwhm to sigma
        auto smooth_sigma_mm = smooth_fwhm_mm/2.355f;
        // Smooth volume
        auto mask = NEWIMAGE::volume<float>(vol.xsize(), vol.ysize(), vol.zsize());
        mask.setdims(vol.xdim(), vol.ydim(), vol.zdim());
        if (implicit_mask){
          for (auto k = 0; k < vol.zsize(); ++k){
            for (auto j = 0; j < vol.ysize(); ++j){
              for (auto i = 0; i < vol.xsize(); ++i){
                mask(i,j,k) = fabsf(vol(i,j,k)) <= std::numeric_limits<float>::epsilon() ?
                  0.0f : 1.0f;
              }
            }
          }
        }
        else{
          mask = 1.0f;
        }
        auto vol_smoothed = NEWIMAGE::smooth(vol, smooth_sigma_mm);
        auto mask_smoothed = NEWIMAGE::smooth(mask, smooth_sigma_mm);
        for (auto k = 0; k < vol_smoothed.zsize(); ++k){
          for (auto j = 0; j < vol_smoothed.ysize(); ++j){
            for (auto i = 0; i < vol_smoothed.xsize(); ++i){
              vol_smoothed(i,j,k) = mask(i,j,k) > 0.5f ?
                vol_smoothed(i,j,k)/mask_smoothed(i,j,k) : 0.0f;
            }
          }
        }
        // Set resolution
        resolution_[0] = vol_smoothed.xdim();
        resolution_[1] = vol_smoothed.ydim();
        resolution_[2] = vol_smoothed.zdim();
        // Set dimensions
        dimensions_[0] = vol_smoothed.xsize();
        dimensions_[1] = vol_smoothed.ysize();
        dimensions_[2] = vol_smoothed.zsize();
        // Initialise mm_to_vox
        mm_to_vox_ = extract_newimage_mm_to_vox_(vol_smoothed);
        arma::fmat vox_to_mm = mm_to_vox_.i();
        for (auto i = 0; i < 4; ++i){
          for (auto j = 0; j < 4; ++j){
            mm_to_vox_vec_[i*4 + j] = mm_to_vox_(i,j);
            vox_to_mm_vec_[i*4 + j] = vox_to_mm(i,j);
          }
        }
        // Copy data from NEWIMAGE::volume to host vector
        coefficients_ = extract_newimage_data_(vol_smoothed);
#pragma omp critical
{
        values_to_coefficients_(dimensions_, coefficients_);
}
        if (debug){
          // Create folder
          auto mkdir_command = "mkdir -p " + debug_dir_;
          auto tmp_sysval = std::system(mkdir_command.c_str());
          // Save grad
          NEWIMAGE::write_volume(vol_smoothed, debug_dir_ + "vol_" + std::to_string(id));
        }
      }

      /// Return dimensions of volume
      std::vector<int> get_dimensions() const
      {
        std::vector<int> o_vec(3);
        o_vec[0] = dimensions_[0];
        o_vec[1] = dimensions_[1];
        o_vec[2] = dimensions_[2];

        return o_vec;
      }

      /// Return values at all specified positions
      std::vector<float> sample(
          const std::vector<std::vector<float> >& positions) const
      {
        // Convert to voxel coordinates and sample in one step
        auto vox_samples = sample_mm_(positions);
        return vox_samples;
      }

      /// Return value of volume at all specified positions, differentiated in all dimensions
      std::vector<thrust::host_vector<float> > sample_gradient(
          const std::vector<std::vector<float> >& positions) const
      {
        // Convert to voxel coordinates and sample in one step
        auto vox_gradient_samples = sample_gradient_mm_(positions);
        return vox_gradient_samples;
      }

      /// Get original sample points in mm
      std::vector<std::vector<float> > get_original_sample_positions() const
      {
        /// \todo Repalce 3D volume assumption with more general implementation
        auto sample_positions_vox = std::vector<std::vector<float> >(3);
        // Generate voxel locations
        for (auto z = 0; z < dimensions_[2]; ++z){
          for (auto y = 0; y < dimensions_[1]; ++y){
            for (auto x = 0; x < dimensions_[0]; ++x){
              sample_positions_vox[0].push_back(x);
              sample_positions_vox[1].push_back(y);
              sample_positions_vox[2].push_back(z);
            }
          }
        }
        // Convert voxels to mm
        auto sample_positions_mm = apply_vox_to_mm_(sample_positions_vox);
        return sample_positions_mm;
      }

      /// Set b-spline coefficients
      void set_coefficients(const std::vector<float>& coefficients)
      {
        /// \todo Add exception test
        assert (coefficients.size() == coefficients_.size());
        coefficients_ = coefficients;
      }

      /// Get b-spline coefficients
      std::vector<float> get_coefficients() const
      {
        auto coefficients_out = std::vector<float>(coefficients_.size());
        thrust::copy(coefficients_.begin(), coefficients_.end(), coefficients_out.begin());
        return coefficients_out;
      }

      /// Get valid extents of volume in real coordinates (mm)
      /// \details Each vector in the pair represents one vertex of the volume, and the
      ///          difference between them gives the valid range for each volume. NOTE: These
      ///          values ARE ordered, i.e. it is guaranteed that:
      ///
      ///                             pair.first =< pair.second
      ///
      std::pair<std::vector<float>, std::vector<float> > get_extents() const
      {
        auto corners_vox = std::vector<std::vector<float> >(3, std::vector<float>(8));
        { // Scope limiting
          auto corner_i = 0;
          for (auto z = 0; z < 2; ++z){
            for (auto y = 0; y < 2; ++y){
              for (auto x = 0; x < 2; ++x){
                 corners_vox[0][corner_i] =
                   (x == 0)
                   ? 0.0f
                   : (static_cast<int>(dimensions_[0]) - 1);
                 corners_vox[1][corner_i] =
                   (y == 0)
                   ? 0.0f
                   : (static_cast<int>(dimensions_[1]) - 1);
                 corners_vox[2][corner_i] =
                   (z == 0)
                   ? 0.0f
                   : (static_cast<int>(dimensions_[2]) - 1);
                 ++corner_i;
              }
            }
          }
        } // Scope limiting
        auto corners_mm = apply_vox_to_mm_(corners_vox);
        auto corner_bottom = std::vector<float>(3);
        auto corner_top = std::vector<float>(3);
        { // Scope limiting
          auto i_dim = 0;
          for (const auto& dim : corners_mm){
            corner_bottom[i_dim] = *std::min_element(dim.begin(), dim.end());
            corner_top[i_dim] = *std::max_element(dim.begin(), dim.end());
            ++ i_dim;
          }
        } // Scope limiting
        return std::make_pair(corner_bottom,corner_top);
      }

      /// Get a reparameterised copy of this volume
      /// \details This increases the number of parameters used in the b-spline representation
      ///          by an integer factor in all directions. This should maintain a one-to-one
      ///          mapping between interpolated intensities in both parameterisations
      void reparameterise(const int scaling_factor)
      {
        // Get indices of the new parameterisation in VOXELS
        // To do this we place a sample point at and between each original voxel, including
        // to the left and the right. We should end up therefor with:
        //      n_voxels_new = (n_voxels_old x scaling_factor) + (scaling_factor - 1)
        // This keeps the total support of the field constant, but may introduce some splines
        // which do not directly overlap with the original voxels
        auto resolution_new = resolution_;
        auto dimensions_new = dimensions_;
        for (auto i = 0; i < dimensions_.size(); ++i){
          resolution_new[i] = resolution_[i]/static_cast<float>(scaling_factor);
          dimensions_new[i] = dimensions_[i]*scaling_factor + (scaling_factor - 1);
        }
        auto position_step = 1.0f/static_cast<float>(scaling_factor);
        auto position_offset = static_cast<float>(scaling_factor - 1) * position_step;
        auto positions_vox = std::vector<std::vector<float> >(dimensions_.size());
        for (auto z = (int)0; z < dimensions_new[2]; ++z){
          for (auto y = (int)0; y < dimensions_new[1]; ++y){
            for (auto x = (int)0; x < dimensions_new[0]; ++x){
              positions_vox[0].push_back(
                  static_cast<float>(x) * position_step - position_offset);
              positions_vox[1].push_back(
                  static_cast<float>(y) * position_step - position_offset);
              positions_vox[2].push_back(
                  static_cast<float>(z) * position_step - position_offset);
            }
          }
        }
        // Sample the original values at the new positions
        auto samples_new = thrust::host_vector<float>(sample_voxels_(positions_vox));
        // Convert the new samples to b-spline coefficients
#pragma omp critical
{
        values_to_coefficients_(dimensions_new, samples_new);
}
        // Update the mm_to_vox matrix to reflect the change in parameterisation
        /// \todo This won't work if off-diagonals are non-zero
        auto mm_to_vox_new = mm_to_vox_;
        for (auto i = 0; i < dimensions_.size(); ++i){
          mm_to_vox_new(i,i) *= static_cast<float>(scaling_factor);
          mm_to_vox_new(i,dimensions_.size())
            = (mm_to_vox_new(i,dimensions_.size()) * static_cast<float>(scaling_factor))
            + static_cast<float>(scaling_factor - 1);
        }
        // Replace the private datamembers with the new versions
        resolution_ = resolution_new;
        dimensions_ = dimensions_new;
        mm_to_vox_ = mm_to_vox_new;
        arma::fmat vox_to_mm = mm_to_vox_.i();
        for (auto i = 0; i < 4; ++i){
          for (auto j = 0; j < 4; ++j){
            mm_to_vox_vec_[i*4 + j] = mm_to_vox_(i,j);
            vox_to_mm_vec_[i*4 + j] = vox_to_mm(i,j);
          }
        }
        coefficients_ = samples_new;
      }

      /// Crop the volume to the defined extents
      /// \details The underlying samples will be reduced such that the given extents still
      ///          have the maximum number of splines with support at that position
      void crop(const std::pair<std::vector<float>, std::vector<float> > extents)
      {
        /// \todo Replace assert with exception
        assert (extents.first.size() == extents.second.size()
            && extents.first.size() == dimensions_.size());

        auto n_dims = dimensions_.size();
        auto extents_mm = std::vector<std::vector<float> >(n_dims, std::vector<float>(2));
        { // Scope limiting
          auto i_dim = 0;
          for (const auto& dim : dimensions_){
            extents_mm[i_dim][0] = extents.first[i_dim];
            extents_mm[i_dim][1] = extents.second[i_dim];
            ++i_dim;
          }
        } // Scope limiting
        auto extents_vox = apply_mm_to_vox_(extents_mm);
        auto extents_bottom_vox = std::vector<float>(n_dims);
        auto extents_top_vox = std::vector<float>(n_dims);
        { // Scope limiting
          auto i_dim = 0;
          for (const auto& dim : extents_vox){
            extents_bottom_vox[i_dim] = std::min(dim[0],dim[1]);
            extents_top_vox[i_dim] = std::max(dim[0],dim[1]);
            ++ i_dim;
          }
        } // Scope limiting

        // Calculate the voxels corresponding to the extents
        //auto extent_vox_first = apply_mm_to_vox_(extents.first);
        //auto extent_vox_second = apply_mm_to_vox_(extents.second);
        // Calculate the voxel extents producing full support at the original mm extents
        auto corner_bottom_vox = std::vector<int>(n_dims);
        auto corner_top_vox = std::vector<int>(n_dims);

        for (auto i = 0; i < n_dims; ++i){
          corner_bottom_vox[i] = static_cast<int>(std::floor(extents_bottom_vox[i] - 1));
          corner_bottom_vox[i] = std::max(corner_bottom_vox[i], 0);
          corner_top_vox[i] = static_cast<int>(std::ceil(extents_top_vox[i] + 1));
          corner_top_vox[i] = std::min(corner_top_vox[i], dimensions_[i] - 1);
        }
        // "Crop" data
        auto cropped_coefficients = thrust::host_vector<float>();
        for (auto z = corner_bottom_vox[2]; z <= corner_top_vox[2]; ++z){
          for (auto y = corner_bottom_vox[1]; y <= corner_top_vox[1]; ++y){
            for (auto x = corner_bottom_vox[0]; x <= corner_top_vox[0]; ++x){
              cropped_coefficients.push_back(
                  coefficients_[z*dimensions_[1]*dimensions_[0] + y*dimensions_[0] + x]);
            }
          }
        }
        coefficients_ = cropped_coefficients;
        // Update dimensions
        for (auto i = 0; i < n_dims; ++i){
          dimensions_[i] = corner_top_vox[i] - corner_bottom_vox[i] + 1;
        }
        // Update mm_to_vox matrix
        for (auto i = 0; i < n_dims; ++i){
          mm_to_vox_(i, n_dims) -= corner_bottom_vox[i];
        }
        arma::fmat vox_to_mm = mm_to_vox_.i();
        for (auto i = 0; i < 4; ++i){
          for (auto j = 0; j < 4; ++j){
            mm_to_vox_vec_[i*4 + j] = mm_to_vox_(i,j);
            vox_to_mm_vec_[i*4 + j] = vox_to_mm(i,j);
          }
        }
      }

      /// Get the resolution of the underlying parameterisation
      std::vector<float> get_resolution() const
      {
        /// \todo Make this more general, i.e. not limited to 3 dimensions
        return resolution_;
      }

      /// Get the mm_to_vox rotation. NOTE: This may include a flip of the x-axis, and as
      /// such is not a true rotation.
      arma::fmat33 get_mm_to_vox_rotation() const
      {
        auto rotation_mat = arma::fmat33();
        auto mm_to_vox_mat = arma::fmat33(mm_to_vox_.submat(0,0,2,2));

        rotation_mat = arma::sqrtmat_sympd(
            (mm_to_vox_mat * (mm_to_vox_mat.t())).i()) * mm_to_vox_mat;
        return rotation_mat;
      }
    private:
      //////////////////////////////////////////////////
      // Private functions
      //////////////////////////////////////////////////
      /// Wrap texture binding to conform to RAII and ensure the texture is
      /// always released
      class TextureBinder
      {
        public:
          /// All the work is done in the constructor & destructor
          TextureBinder(
              const thrust::host_vector<int>&   dimensions,
              const thrust::host_vector<float>& values,
              const bool                        clamp_border)
            : coeffs_array(0)
          {
            // Copy data to device
            auto coeffs_pitched_ptr = CopyVolumeHostToDevice(
                thrust::raw_pointer_cast(values.data()),
                static_cast<unsigned int>(dimensions[0]),
                static_cast<unsigned int>(dimensions[1]),
                static_cast<unsigned int>(dimensions[2]));
            // Create a texture to hold the coefficients
            auto volume_extent = make_cudaExtent(
                static_cast<unsigned int>(dimensions[0]),
                static_cast<unsigned int>(dimensions[1]),
                static_cast<unsigned int>(dimensions[2]));
            // Note: coeffs is GLOBAL in scope
            CreateTextureFromVolume(
                &coeffs,
                &coeffs_array,
                coeffs_pitched_ptr,
                volume_extent,
                true);
            coeffs.normalized = false;
            if (clamp_border){
              coeffs.addressMode[0] = cudaAddressModeClamp;
              coeffs.addressMode[1] = cudaAddressModeClamp;
              coeffs.addressMode[2] = cudaAddressModeClamp;
            }
            else{
              coeffs.addressMode[0] = cudaAddressModeBorder;
              coeffs.addressMode[1] = cudaAddressModeBorder;
              coeffs.addressMode[2] = cudaAddressModeBorder;
            }
            // Now release pitched pointer as data is in texture
            CUDA_SAFE_CALL(cudaFree(coeffs_pitched_ptr.ptr));
          }
          /// Release the texture to be used again later
          ~TextureBinder()
          {
            // Unbind texture
            CUDA_SAFE_CALL(cudaUnbindTexture(coeffs));
            CUDA_SAFE_CALL(cudaFreeArray(coeffs_array));
          }
        private:
          cudaArray *coeffs_array;
      };

      /// Extract mm_to_vox from NEWIMAGE::volume
      arma::Mat<float> extract_newimage_mm_to_vox_(const NEWIMAGE::volume<float>& vol)
      {
        auto vox_to_mm_newimage = vol.newimagevox2mm_mat();
        //auto vox_to_mm_newimage = vol.sampling_mat();
        auto vox_to_mm_arma = arma::Mat<float>(4, 4, arma::fill::zeros);
        for (auto j = 0; j < 4; ++j){
          for (auto i = 0; i < 4; ++i){
            vox_to_mm_arma(i,j) = vox_to_mm_newimage.element(i,j);
          }
        }
        auto mm_to_vox_arma = affine_invert(vox_to_mm_arma);
        return mm_to_vox_arma;
      }

      /// Invert a 4x4 affine transform
      arma::fmat affine_invert(const arma::fmat& affine_mat) const
      {
        auto affine_mat_inv = affine_mat;
        affine_mat_inv.submat(0,0,2,2) = affine_mat_inv.submat(0,0,2,2).i();
        affine_mat_inv.submat(0,3,2,3) =
            -affine_mat_inv.submat(0,0,2,2)*affine_mat_inv.submat(0,3,2,3);
        return affine_mat_inv;
      }

      /// Extract data from NEWIMAGE::volume
      thrust::host_vector<float> extract_newimage_data_(const NEWIMAGE::volume<float>& vol)
      {
        thrust::host_vector<float> newimage_data(vol.xsize()*vol.ysize()*vol.zsize());
        unsigned int i = 0;
        for (NEWIMAGE::volume<float>::fast_const_iterator it = vol.fbegin();
            it != vol.fend();
            ++it, ++i)
        {
          newimage_data[i] = *it;
        }
        return newimage_data;
      }

      /// Calculate spline coefficients
      void values_to_coefficients_(
          const thrust::host_vector<int>& dimensions,
          thrust::host_vector<float>& values)
      {
        // Calculate the b-spline coefficients for the volume
        auto coeffs_pitched_ptr = CopyVolumeHostToDevice(
            values.data(),
            static_cast<unsigned int>(dimensions[0]),
            static_cast<unsigned int>(dimensions[1]),
            static_cast<unsigned int>(dimensions[2]));
        CubicBSplinePrefilter3D(
            static_cast<float*>(coeffs_pitched_ptr.ptr),
            static_cast<unsigned int>(coeffs_pitched_ptr.pitch),
            static_cast<unsigned int>(dimensions[0]),
            static_cast<unsigned int>(dimensions[1]),
            static_cast<unsigned int>(dimensions[2]));
        CopyVolumeDeviceToHost(
            values.data(),
            coeffs_pitched_ptr,
            static_cast<unsigned int>(dimensions[0]),
            static_cast<unsigned int>(dimensions[1]),
            static_cast<unsigned int>(dimensions[2]));
      }

      /// Convert from mm to voxel coordinates using internal mm_to_vox_ matrix
      std::vector<std::vector<float> > apply_mm_to_vox_(
          const std::vector<std::vector<float> >& mm_positions) const
      {
        auto vox_positions = std::vector<std::vector<float> >(mm_positions.size());
        // Apply mm_to_vox transform
        // NB!!! Because armadillo is column major, we use a column vector for each vector of
        // positions for a particular dimension, and therefore the transpose of the mm_to_vox_
        // matrix
        arma::Mat<float> augmented_positions(mm_positions[0].size(), mm_positions.size()+1);
        { // Scope limiting
          auto i_dim = 0;
          for (const auto& mm_vec : mm_positions){
            augmented_positions.col(i_dim) = arma::Col<float>(mm_vec);
            ++i_dim;
          }
          augmented_positions.col(i_dim) = arma::Col<float>(
              mm_positions[0].size(),
              arma::fill::ones);
        } // Scope limiting

        augmented_positions = augmented_positions * arma::trans(mm_to_vox_);

        { // Scope limiting
          auto i_dim = 0;
          for (auto& vox_vec : vox_positions){
            vox_vec = arma::conv_to<std::vector<float> >::from(
                augmented_positions.col(i_dim));
            ++i_dim;
          }
        } // Scope limiting
        return vox_positions;
      }

      /// Convert from voxel to mm coordinates using internal mm_to_vox_ matrix
      std::vector<std::vector<float> > apply_vox_to_mm_(
          const std::vector<std::vector<float> >& vox_positions) const
      {
        auto mm_positions = std::vector<std::vector<float> >(vox_positions.size());
        // Apply vox_to_mm transform
        // NB!!! Because armadillo is column major, we use a column vector for each vector of
        // positions for a particular dimension, and therefore the transpose of the mm_to_vox_
        // matrix
        arma::Mat<float> augmented_positions(vox_positions[0].size(), vox_positions.size()+1);
        { // Scope limiting
          auto i_dim = 0;
          for (const auto& vox_vec : vox_positions){
            augmented_positions.col(i_dim) = arma::Col<float>(vox_vec);
            ++i_dim;
          }
          augmented_positions.col(i_dim) = arma::Col<float>(
              vox_positions[0].size(),
              arma::fill::ones);
        } // Scope limiting
        augmented_positions = augmented_positions * arma::trans(affine_invert(mm_to_vox_));
        { // Scope limiting
          auto i_dim = 0;
          for (auto& mm_vec : mm_positions){
            mm_vec = arma::conv_to<std::vector<float> >::from(
                augmented_positions.col(i_dim));
            ++i_dim;
          }
        } // Scope limiting
        return mm_positions;
      }

      /// Sample volume using voxel coordinates
      std::vector<float> sample_voxels_(
          const std::vector<std::vector<float> >& vox_positions) const
      {
        auto n_samples = vox_positions[0].size();
        /// \todo Check dimensionality of the samples is sensible
        // Copy volume data into device texture
        TextureBinder volume_texture(dimensions_, coefficients_, clamp_border_);
        // Perform interpolation
        // Copy data to device
        auto x_coords_dev = thrust::device_vector<float>(vox_positions[0]);
        auto y_coords_dev = thrust::device_vector<float>(vox_positions[1]);
        auto z_coords_dev = thrust::device_vector<float>(vox_positions[2]);
        auto samples_dev = thrust::device_vector<float>(n_samples);
        // Get raw pointers to device data
        float *x_coords_raw = thrust::raw_pointer_cast(x_coords_dev.data());
        float *y_coords_raw = thrust::raw_pointer_cast(y_coords_dev.data());
        float *z_coords_raw = thrust::raw_pointer_cast(z_coords_dev.data());
        float *samples_raw = thrust::raw_pointer_cast(samples_dev.data());
        // Calculate kernel parameters
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_cubic_bspline_sample,
            0,
            0);
        unsigned int grid_size = (n_samples + block_size - 1)/block_size;
        // Call CUDA Kernel
        MMORF::kernel_cubic_bspline_sample<<<grid_size, block_size>>>(
            // Input
            n_samples,
            x_coords_raw,
            y_coords_raw,
            z_coords_raw,
            // Output
            samples_raw);
        checkCudaErrors(cudaDeviceSynchronize());
        // Copy interpolated data back to host
        auto samples_host = thrust::host_vector<float>(samples_dev);
        // Return interpolated values
        auto samples_return = std::vector<float>(samples_host.size());
        thrust::copy(samples_host.begin(), samples_host.end(), samples_return.begin());
        return samples_return;
      }

      /// Sample volume using mm coordinates
      std::vector<float> sample_mm_(
          const std::vector<std::vector<float> >& mm_positions) const
      {
        auto n_samples = mm_positions[0].size();
        /// \todo Check dimensionality of the samples is sensible
        // Copy volume data into device texture
        TextureBinder volume_texture(dimensions_, coefficients_, clamp_border_);
        // Perform interpolation
        // Copy data to device
        auto x_coords_dev = thrust::device_vector<float>(mm_positions[0]);
        auto y_coords_dev = thrust::device_vector<float>(mm_positions[1]);
        auto z_coords_dev = thrust::device_vector<float>(mm_positions[2]);
        auto samples_dev = thrust::device_vector<float>(n_samples);
        // Get raw pointers to device data
        float *x_coords_raw = thrust::raw_pointer_cast(x_coords_dev.data());
        float *y_coords_raw = thrust::raw_pointer_cast(y_coords_dev.data());
        float *z_coords_raw = thrust::raw_pointer_cast(z_coords_dev.data());
        const float *affine_raw = thrust::raw_pointer_cast(mm_to_vox_vec_.data());
        float *samples_raw = thrust::raw_pointer_cast(samples_dev.data());
        // Calculate kernel parameters
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_cubic_bspline_sample_with_affine,
            0,
            0);
        unsigned int grid_size = (n_samples + block_size - 1)/block_size;
        // Call CUDA Kernel
        MMORF::kernel_cubic_bspline_sample_with_affine<<<grid_size, block_size>>>(
            // Input
            n_samples,
            x_coords_raw,
            y_coords_raw,
            z_coords_raw,
            affine_raw,
            // Output
            samples_raw);
        checkCudaErrors(cudaDeviceSynchronize());
        // Copy interpolated data back to host
        auto samples_host = thrust::host_vector<float>(samples_dev);
        // Return interpolated values
        auto samples_return = std::vector<float>(samples_host.size());
        thrust::copy(samples_host.begin(), samples_host.end(), samples_return.begin());
        return samples_return;
      }

      /// Return value of volume at all specified positions, differentiated in all dimensions
      std::vector<thrust::host_vector<float> > sample_gradient_voxels_(
          const std::vector<std::vector<float> >& vox_positions) const
      {
        /// \todo Exception testing
        auto n_samples = vox_positions[0].size();
        auto samples_host = std::vector<thrust::host_vector<float> >(3);
        /// \todo Check dimensionality of the samples is sensible
        // Copy volume data into device texture
        TextureBinder volume_texture(dimensions_, coefficients_, clamp_border_);
        // Perform interpolation
        // Copy data to device
        auto x_coords_dev = thrust::device_vector<float>(vox_positions[0]);
        auto y_coords_dev = thrust::device_vector<float>(vox_positions[1]);
        auto z_coords_dev = thrust::device_vector<float>(vox_positions[2]);
        // Get raw pointers to device data
        float *x_coords_raw = thrust::raw_pointer_cast(x_coords_dev.data());
        float *y_coords_raw = thrust::raw_pointer_cast(y_coords_dev.data());
        float *z_coords_raw = thrust::raw_pointer_cast(z_coords_dev.data());
        // We need to sample dervatives in all 3 dimension due to affine transform
        for (auto dim = 0; dim < 3; ++dim){
          auto samples_dev = thrust::device_vector<float>(n_samples,0.0f);
          float *samples_raw = thrust::raw_pointer_cast(samples_dev.data());
          // Calculate kernel parameters
          int min_grid_size;
          int block_size;
          cudaOccupancyMaxPotentialBlockSize(
              &min_grid_size,
              &block_size,
              MMORF::kernel_cubic_bspline_sample_derivative,
              0,
              0);
          unsigned int grid_size = (n_samples + block_size - 1)/block_size;
          // Call CUDA Kernel
          MMORF::kernel_cubic_bspline_sample_derivative<<<grid_size, block_size>>>(
              // Input
              dim,
              n_samples,
              x_coords_raw,
              y_coords_raw,
              z_coords_raw,
              // Output
              samples_raw);
          checkCudaErrors(cudaDeviceSynchronize());
          // Copy interpolated data back to host
          samples_host[dim] = thrust::host_vector<float>(samples_dev);
        }
        // Correct from derivative in per_vox to per_mm
        auto samples_arma = arma::fmat(3, n_samples);
#pragma omp parallel for
        for (auto dim = 0; dim < 3; ++dim){
          for (auto samp = 0; samp < n_samples; ++samp){
            samples_arma.at(dim,samp) = samples_host[dim][samp];
          }
        }
        samples_arma = mm_to_vox_.submat(0,0,2,2).t() * samples_arma;
#pragma omp parallel for
        for (auto dim = 0; dim < 3; ++dim){
          for (auto samp = 0; samp < n_samples; ++samp){
            samples_host[dim][samp] = samples_arma.at(dim,samp);
          }
        }
        return samples_host;
      }

      /// Return value of volume at all specified positions, differentiated in all dimensions
      std::vector<thrust::host_vector<float> > sample_gradient_mm_(
          const std::vector<std::vector<float> >& vox_positions) const
      {
        /// \todo Exception testing
        auto n_samples = vox_positions[0].size();
        auto samples_host = std::vector<thrust::host_vector<float> >(3);
        /// \todo Check dimensionality of the samples is sensible
        // Copy volume data into device texture
        TextureBinder volume_texture(dimensions_, coefficients_, clamp_border_);
        // Perform interpolation
        // Copy data to device
        auto x_coords_dev = thrust::device_vector<float>(vox_positions[0]);
        auto y_coords_dev = thrust::device_vector<float>(vox_positions[1]);
        auto z_coords_dev = thrust::device_vector<float>(vox_positions[2]);
        auto dx_samples_dev = thrust::device_vector<float>(n_samples);
        auto dy_samples_dev = thrust::device_vector<float>(n_samples);
        auto dz_samples_dev = thrust::device_vector<float>(n_samples);
        // Get raw pointers to device data
        float *x_coords_raw = thrust::raw_pointer_cast(x_coords_dev.data());
        float *y_coords_raw = thrust::raw_pointer_cast(y_coords_dev.data());
        float *z_coords_raw = thrust::raw_pointer_cast(z_coords_dev.data());
        const float *affine_raw = thrust::raw_pointer_cast(mm_to_vox_vec_.data());
        float *dx_samples_raw = thrust::raw_pointer_cast(dx_samples_dev.data());
        float *dy_samples_raw = thrust::raw_pointer_cast(dy_samples_dev.data());
        float *dz_samples_raw = thrust::raw_pointer_cast(dz_samples_dev.data());

        // We need to sample dervatives in all 3 dimension due to affine transform
        // Calculate kernel parameters
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_cubic_bspline_sample_derivative_with_affine,
            0,
            0);
        unsigned int grid_size = (n_samples + block_size - 1)/block_size;
        // Call CUDA Kernel
        MMORF::kernel_cubic_bspline_sample_derivative_with_affine<<<grid_size, block_size>>>(
            // Input
            n_samples,
            x_coords_raw,
            y_coords_raw,
            z_coords_raw,
            affine_raw,
            // Output
            dx_samples_raw,
            dy_samples_raw,
            dz_samples_raw);
        checkCudaErrors(cudaDeviceSynchronize());
        // Copy interpolated data back to host
        samples_host[0] = thrust::host_vector<float>(dx_samples_dev);
        samples_host[1] = thrust::host_vector<float>(dy_samples_dev);
        samples_host[2] = thrust::host_vector<float>(dz_samples_dev);

        return samples_host;
      }
      //////////////////////////////////////////////////
      // Private datamembers
      //////////////////////////////////////////////////
      /// Resolution of volume in mm/voxel
      std::vector<float>          resolution_;
      /// Dimensions of volume in VOXELS
      thrust::host_vector<int>    dimensions_;
      /// Matrix for converting real-world (mm) coordinates to voxel coordinates
      /// \details This takes the form of an affine transform matrix, which scales mm values
      ///          to voxel values. Additionally, this ensures that the position of
      ///          the origin is conserved.
      arma::Mat<float>            mm_to_vox_;
      /// Unwrapped row major mm_to_vox_matrix
      thrust::device_vector<float>  mm_to_vox_vec_;
      /// Unwrapped row major vox_to_mm_matrix
      thrust::device_vector<float>  vox_to_mm_vec_;
      /// Unwrapped coefficients
      thrust::host_vector<float>  coefficients_;
      /// How to handle sampling outside of border
      bool                        clamp_border_;
      // Debugging options
      int                         id_;
      const static std::string    debug_dir_;
  };

  /// Static debug directory initialisation
  const std::string VolumeBSpline::Impl::debug_dir_ =
    "debug/VolumeBSpline/";

////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  VolumeBSpline::~VolumeBSpline() = default;
  /// Move ctor
  VolumeBSpline::VolumeBSpline(VolumeBSpline&& rhs) = default;
  /// Move assignment operator
  VolumeBSpline& VolumeBSpline::operator=(VolumeBSpline&& rhs) = default;
  /// Copy ctor
  VolumeBSpline::VolumeBSpline(const VolumeBSpline& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  VolumeBSpline& VolumeBSpline::operator=(const VolumeBSpline& rhs)
  {
    if (!rhs.pimpl_){
      pimpl_.reset();
    }
    else if (!pimpl_){
      // pimpl_ = std::unique_ptr<Impl>(new Impl(*rhs.pimpl_));
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
    else{
      *pimpl_ = *rhs.pimpl_;
    }
    return *this;
  }
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
  /// Empty volume of given dimensions and resolution
  VolumeBSpline::VolumeBSpline(
      const std::vector<int>&   dimensions,
      const std::vector<float>& resolution,
      const std::vector<float>& origin,
      const bool                clamp_border)
    : pimpl_(MMORF::make_unique<Impl>(
          next_id_++,
          dimensions,
          resolution,
          origin,
          clamp_border))
  {}
  /// Construct from NEWIMAGE::volume
  VolumeBSpline::VolumeBSpline(
      const NEWIMAGE::volume<float>&  vol,
      const bool                      clamp_border)
    : pimpl_(MMORF::make_unique<Impl>(
          next_id_++,
          vol,
          clamp_border))
  {}
  /// Construct from nifti image file
  VolumeBSpline::VolumeBSpline(
      std::string vol_name,
      const bool  clamp_border)
    : pimpl_(MMORF::make_unique<Impl>(
          next_id_++,
          vol_name,
          clamp_border))
  {}
  /// Construct from NEWIMAGE::volume and smooth with Gaussian (defined by FWHM mm)
  VolumeBSpline::VolumeBSpline(
      const NEWIMAGE::volume<float>&  vol,
      const float                     smooth_fwhm_mm,
      const bool                      implicit_mask,
      const bool                      clamp_border,
      const bool                      debug)
    : pimpl_(MMORF::make_unique<Impl>(
          next_id_++,
          vol,
          smooth_fwhm_mm,
          implicit_mask,
          clamp_border,
          debug))
  {}
  /// Construct from nifti image file and smooth with Gaussian (defined by FWHM mm)
  VolumeBSpline::VolumeBSpline(
      std::string vol_name,
      const float smooth_fwhm_mm,
      const bool  implicit_mask,
      const bool  clamp_border,
      bool        debug)
    : pimpl_(MMORF::make_unique<Impl>(
          next_id_++,
          vol_name,
          smooth_fwhm_mm,
          implicit_mask,
          clamp_border,
          debug))
  {}
  /// Return dimensions of volume
  std::vector<int> VolumeBSpline::get_dimensions() const
  {
    return pimpl_->get_dimensions();
  }
  /// Return values at all specified positions
  std::vector<float> VolumeBSpline::sample(
      const std::vector<std::vector<float>>& positions) const
  {
    return pimpl_->sample(positions);
  }
  /// Return value of volume at all specified positions, differentiated in one dimension
  std::vector<thrust::host_vector<float> > VolumeBSpline::sample_gradient(
      const std::vector<std::vector<float> >& positions) const
  {
    return pimpl_->sample_gradient(positions);
  }
  /// Get original sample points in mm
  std::vector<std::vector<float> > VolumeBSpline::get_original_sample_positions() const
  {
    return pimpl_->get_original_sample_positions();
  }
  /// Set b-spline coefficients
  void VolumeBSpline::set_coefficients(const std::vector<float>& coefficients)
  {
    pimpl_->set_coefficients(coefficients);
  }
  /// Get b-spline coefficients
  std::vector<float> VolumeBSpline::get_coefficients() const
  {
    return pimpl_->get_coefficients();
  }
  /// Get valid extents of volume in real coordinates (mm)
  /// \details Each vector in the pair represents one vertex of the volume, and the
  ///          difference between them gives the valid range for each volume. NOTE: These
  ///          values ARE ordered, i.e. it is guaranteed that:
  ///                             pair.first =< pair.second
  std::pair<std::vector<float>, std::vector<float> > VolumeBSpline::get_extents() const
  {
    return pimpl_->get_extents();
  }
  /// Get a reparameterised copy of this volume
  /// \details This increases the number of parameters used in the b-spline representation
  ///          by an integer factor in all directions. This should maintain a one-to-one
  ///          mapping between interpolated intensities in both parameterisations
  VolumeBSpline VolumeBSpline::reparameterise(const int scaling_factor) const
  {
    /// \todo Replace assert with exception
    assert (scaling_factor > 0);
    auto return_vol = VolumeBSpline(*this);
    return_vol.pimpl_->reparameterise(scaling_factor);
    return return_vol;
  }
  /// Crop the volume to the defined extents
  /// \details The underlying samples will be reduced such that the given extents still
  ///          have the maximum number of splines with support at that position
  VolumeBSpline VolumeBSpline::crop(
      const std::pair<std::vector<float>, std::vector<float> > extents) const
  {
    auto return_vol = VolumeBSpline(*this);
    return_vol.pimpl_->crop(extents);
    return return_vol;
  }
  /// Get the resolution of the underlying parameterisation
  std::vector<float> VolumeBSpline::get_resolution() const
  {
    return pimpl_->get_resolution();
  }
  /// Get the mm_to_vox rotation
  arma::fmat33 VolumeBSpline::get_mm_to_vox_rotation() const
  {
    return pimpl_->get_mm_to_vox_rotation();
  }
  /// Static datamember initialisation
  int VolumeBSpline::next_id_ = 0;
} // MMORF
