//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the sum of squared differences between two volumes, one of
///        which is warped. Additionally, a mask in the ref/mov image domain is used to
///        constrain the area of interest during optimisaton. This is now extended to include
///        a greedy approximation to symmetrisation of the problem by multiplying the cost
///        by (1 + |J|).
/// \details The warp is defined by b-spline field coefficients, one set per dimension in the
///          volume
/// \author Frederik Lange
/// \date March 2021
/// \copyright Copyright (C) 2021 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "CostFxnSSDWarpFieldSymmetricMaskedExcluded.cuh"
#include "MmorfMemory.h"
#include "Volume.h"
#include "WarpFieldBSpline.cuh"
#include "BiasFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"
#include "TextureHandleLinear.cuh"
#include "CostFxnHelpers.cuh"
#include "CostFxnKernels.cuh"
#include "IntensityMapperPolynomial.cuh"
#include "helper_cuda.h"

#include "basisfield/fsl_splines.h"

#include <armadillo>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include <memory>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <functional>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class CostFxnSSDWarpFieldSymmetricMaskedExcluded::Impl
  {
    public:
      //////////////////////////////////////////////////
      // Public functions
      //////////////////////////////////////////////////
      /// Construct by passing in fully constructed volumes, warp and (optional) bias field
      /// \param vol_ref Reference (stationary) volume
      /// \param vol_mov Transformed volume
      /// \param affine_ref Pre-calculated affine transform FROM a common space TO the
      ///                   reference space
      /// \param affine_mov Pre-calculated affine transform FROM a common space TO the
      ///                   moving space
      /// \param mask_ref 3D volume used to mask the reference volume
      /// \param mask_mov 3D volume used to mask the moving volume
      /// \param warp_field Shared pointer to fully formed warp field
      /// \param sampling_frequency How often to sample between B-spline knots. E.g. for:
      ///             knot_spacing        = 10mm
      ///             sampling_frequency  = 5
      ///        warp field will be sampled every 2mm
      /// \param bias_field Shared pointer to fully formed bias field
      Impl(
          std::shared_ptr<MMORF::Volume>           vol_ref,
          std::shared_ptr<MMORF::Volume>           vol_mov,
          const arma::fmat&                        affine_ref,
          const arma::fmat&                        affine_mov,
          std::shared_ptr<MMORF::Volume>           mask_ref,
          std::shared_ptr<MMORF::Volume>           mask_mov,
          std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          int                                      sampling_frequency,
          std::shared_ptr<MMORF::BiasFieldBSpline> bias_field,
          const bool                               debug
          )
        : vol_ref_(vol_ref)
        , vol_mov_(vol_mov)
        , affine_ref_(affine_ref)
        , affine_mov_(affine_mov)
        , mask_ref_(mask_ref)
        , mask_mov_(mask_mov)
        , warp_field_(warp_field)
        , sample_dimensions_(warp_field_->get_robust_sample_dimensions(sampling_frequency))
        , spline_1D_(3,sampling_frequency) // Cubic spline with the correct no. samples
        , bias_field_(bias_field)
        , debug_(debug)
      {
        /// \todo Replace assert with exception
        assert(sampling_frequency > 0);
        sample_positions_warp_ = warp_field_->get_robust_sample_positions(sampling_frequency);
        sample_positions_ref_ = affine_transform_(sample_positions_warp_, affine_ref_);
        // Calculate normalisation values for both volumes
        auto sample_positions_mov = affine_transform_(sample_positions_warp_, affine_mov);
        auto ref_samples = vol_ref_->sample(sample_positions_ref_);
        auto mov_samples = vol_mov_->sample(sample_positions_mov);
        auto ref_mask_samples = std::vector<float>(ref_samples.size(), 1.0f);
        auto mov_mask_samples = std::vector<float>(mov_samples.size(), 1.0f);
        if(mask_ref_){
          ref_mask_samples = mask_ref_->sample(sample_positions_ref_);
        }
        if(mask_mov_){
          mov_mask_samples = mask_mov_->sample(sample_positions_mov);
        }
        norm_factor_ref_ = calculate_robust_norm_factor_(
          ref_samples,
          ref_mask_samples,
          mov_mask_samples
          );
        norm_factor_mov_ = calculate_robust_norm_factor_(
          mov_samples,
          ref_mask_samples,
          mov_mask_samples
          );
        //create_intensity_mapper_();
      }

      /// Get the current value of the parameters
      std::vector<std::vector<float> > get_parameters() const
      {
        return warp_field_->get_parameters();
      }

      /// Set the current value of the parameters
      void set_parameters(
          const std::vector<std::vector<float> >& parameters)
      {
        warp_field_->set_parameters(parameters);
      }

      /// Get cost under current parameterisation
      float cost() const
      {
        // Sample reference and moving volumes
        auto samples_ref = vol_ref_->sample(sample_positions_ref_);
        normalise_samples_(samples_ref,norm_factor_ref_);
        if (bias_field_){
          samples_ref = bias_field_->apply_bias(sample_positions_ref_, samples_ref);
        }
        //samples_ref = intensity_mapper_.map_intensities(samples_ref);
        auto sample_positions_warped = warp_field_->apply_warp(sample_positions_warp_);
        sample_positions_warped = affine_transform_(sample_positions_warped, affine_mov_);
        auto samples_mov = vol_mov_->sample(sample_positions_warped);
        normalise_samples_(samples_mov,norm_factor_mov_);
        // Calculate Jacobian determinant for current parametrisation
        auto samples_jac_det = warp_field_->jacobian_determinants(sample_positions_warp_);
        // Calculate cost from samples
        auto samples_error = std::vector<float>(samples_ref.size());
        std::transform(
            samples_mov.begin(),
            samples_mov.end(),
            samples_ref.begin(),
            samples_error.begin(),
            std::minus<float>()
            );
        std::transform(
            samples_error.begin(),
            samples_error.end(),
            samples_error.begin(),
            samples_error.begin(),
            std::multiplies<float>()
            );
        std::transform(
            samples_error.begin(),
            samples_error.end(),
            samples_jac_det.begin(),
            samples_error.begin(),
            [](float a, float b) -> float { return 0.5f*(1.0f + b)*a; }
            );
        if (mask_ref_){
          auto samples_mask_ref = mask_ref_->sample(sample_positions_ref_);
          std::transform(
              samples_error.begin(),
              samples_error.end(),
              samples_mask_ref.begin(),
              samples_error.begin(),
              std::multiplies<float>()
              );
        }
        if (mask_mov_){
          auto samples_mask_mov = mask_mov_->sample(sample_positions_warped);
          std::transform(
              samples_error.begin(),
              samples_error.end(),
              samples_mask_mov.begin(),
              samples_error.begin(),
              std::multiplies<float>()
              );
        }
        auto cost = std::accumulate(
            samples_error.begin(),
            samples_error.end(),
            0.0f
            );
        // Avergage cot
        cost /= samples_ref.size();
        return cost;
      }
      /// Get Jte under current parameterisation
      arma::fvec grad() const
      {
        // Calculate jte_sz
        auto warp_sz = warp_field_->get_parameter_size();
        auto jte_sz = warp_sz.first * warp_sz.second;
        // Make a device vector for storing the result
        auto jte_dev = thrust::device_vector<float>(jte_sz);
        // Create and sample error volume
        // Sample reference and moving volumes
        auto samples_ref = vol_ref_->sample(sample_positions_ref_);
        normalise_samples_(samples_ref,norm_factor_ref_);
        if (bias_field_){
          samples_ref = bias_field_->apply_bias(sample_positions_ref_, samples_ref);
        }
        //samples_ref = intensity_mapper_.map_intensities(samples_ref);
        auto sample_positions_warped = warp_field_->apply_warp(sample_positions_warp_);
        sample_positions_warped = affine_transform_(sample_positions_warped, affine_mov_);
        auto samples_mov = vol_mov_->sample(sample_positions_warped);
        normalise_samples_(samples_mov,norm_factor_mov_);
        // Calculate Jacobian determinant for current parametrisation
        auto samples_jac_det = warp_field_->jacobian_determinants(sample_positions_warp_);
        // Calculate current sample error
        auto samples_error = std::vector<float>(samples_ref.size());
        std::transform(
            samples_mov.begin(),
            samples_mov.end(),
            samples_ref.begin(),
            samples_error.begin(),
            std::minus<float>()
            );
        std::transform(
            samples_error.begin(),
            samples_error.end(),
            samples_jac_det.begin(),
            samples_error.begin(),
            [](float a, float b) -> float { return 0.5f*(1.0f + b)*a; }
            );
        if (mask_ref_){
          auto samples_mask_ref = mask_ref_->sample(sample_positions_ref_);
          std::transform(
              samples_error.begin(),
              samples_error.end(),
              samples_mask_ref.begin(),
              samples_error.begin(),
              std::multiplies<float>()
              );
        }
        if (mask_mov_){
          auto samples_mask_mov = mask_mov_->sample(sample_positions_warped);
          std::transform(
              samples_error.begin(),
              samples_error.end(),
              samples_mask_mov.begin(),
              samples_error.begin(),
              std::multiplies<float>()
              );
        }
        // Convert spline to vec
        auto spline_vec = spline_as_vec_(spline_1D_);
        // Calculate derivative samples
        auto samples_mov_deriv_vec = calculate_derivatives_mov_(
            sample_positions_warped,
            affine_mov_);
        // Loop over all warp directions
        for (auto warp_dim = 0; warp_dim < warp_sz.first; ++warp_dim){
          // Normalise sampled derivatives
          normalise_samples_(samples_mov_deriv_vec[warp_dim],norm_factor_mov_);
          // Pre-multiply the error and derivative samples
          auto samples_pre_mult = std::vector<float>(samples_error.size());
          std::transform(
              samples_error.begin(),
              samples_error.end(),
              samples_mov_deriv_vec[warp_dim].begin(),
              samples_pre_mult.begin(),
              std::multiplies<float>()
              );
          // Get correct index into jte_dev (x, then y, then z)
          auto jte_dev_ptr = thrust::raw_pointer_cast(
              &jte_dev[warp_dim * warp_sz.second]);
          calculate_sub_jte_(
              // Input
              samples_pre_mult,
              sample_dimensions_,
              spline_vec,
              spline_vec,
              spline_vec,
              warp_field_->get_dimensions(),
              std::vector<int>(3, static_cast<int>(spline_1D_.KnotSpacing())),
              // Output
              jte_dev_ptr);
        }
        // Copy jte back to host
        auto jte_host = thrust::host_vector<float>(jte_dev);
        // Return result
        auto jte_return = arma::fvec(jte_host.data(), jte_host.size());
        jte_return = 2.0f*jte_return/samples_ref.size();
        return jte_return;
      }
      /// Get JtJ under current parameterisation
      MMORF::SparseDiagonalMatrixTiled hess() const
      {
        // Get size of warp field
        auto warp_sz = warp_field_->get_parameter_size();
        // Apply warp field to sample positions
        auto sample_positions_warped = warp_field_->apply_warp(sample_positions_warp_);
        sample_positions_warped = affine_transform_(sample_positions_warped, affine_mov_);
        // Sample derivative of moving volume in all directions and store in vector
        auto samples_mov_deriv_vec = calculate_derivatives_mov_(
            sample_positions_warped,
            affine_mov_);
        for (auto i = 0; i < warp_sz.first; ++i){
          normalise_samples_(samples_mov_deriv_vec[i],norm_factor_mov_);
        }
        // Calculate Jacobian determinant for current parametrisation
        auto samples_jac_det = warp_field_->jacobian_determinants(sample_positions_warp_);
        // Convert spline to vec
        auto spline_vec = spline_as_vec_(spline_1D_);
        // Create sparse tiled matrix to store Hessian
        auto sparse_jtj = create_empty_jtj_(warp_field_->get_dimensions());
        // Loop over all warp directions
        for (auto row = 0; row < warp_sz.first; ++row){
          // Loop over all warp directions again
          for (auto col = 0; col < warp_sz.first; ++col){
            // Check if this is a unique sub-jtj - we will keep the main diagonal and below
            if (col > row) continue;
            // Pre-multiply the derivative samples
            auto samples_pre_mult = std::vector<float>(samples_mov_deriv_vec[0].size());
            std::transform(
                samples_mov_deriv_vec[row].begin(),
                samples_mov_deriv_vec[row].end(),
                samples_mov_deriv_vec[col].begin(),
                samples_pre_mult.begin(),
                std::multiplies<float>()
                );
            // Pre-multiply by the Jacobian modulation
            std::transform(
                samples_pre_mult.begin(),
                samples_pre_mult.end(),
                samples_jac_det.begin(),
                samples_pre_mult.begin(),
                [](float a, float b) -> float { return 0.5f*(1.0f + b)*a; }
                );
            if (mask_ref_){
              auto samples_mask_ref = mask_ref_->sample(sample_positions_ref_);
              std::transform(
                  samples_pre_mult.begin(),
                  samples_pre_mult.end(),
                  samples_mask_ref.begin(),
                  samples_pre_mult.begin(),
                  std::multiplies<float>()
                  );
            }
            if (mask_mov_){
              auto samples_mask_mov = mask_mov_->sample(sample_positions_warped);
              std::transform(
                  samples_pre_mult.begin(),
                  samples_pre_mult.end(),
                  samples_mask_mov.begin(),
                  samples_pre_mult.begin(),
                  std::multiplies<float>()
                  );
            }
            // Call kernel
            calculate_sub_jtj_symmetrical_(
              // Input
              samples_pre_mult,
              sample_dimensions_,
              spline_vec,
              spline_vec,
              spline_vec,
              warp_field_->get_dimensions(),
              std::vector<int>(3, static_cast<int>(spline_1D_.KnotSpacing())),
              row,
              col,
              sparse_jtj);
            // Save above diagonal sub-jtj
            if (row > col) sparse_jtj.copy_submatrix(row, col, col, row);
          }
        }
        // Return completed matrix
        sparse_jtj *= (2.0 / static_cast<float>(samples_mov_deriv_vec[0].size()));
        return sparse_jtj;
      }
    private:
      //////////////////////////////////////////////////
      // Private functions
      //////////////////////////////////////////////////
      /// Affine transform a given set of positions
      std::vector<std::vector<float> > affine_transform_(
          const std::vector<std::vector<float> >& positions,
          const arma::fmat& affine_mat) const
      {
        /// \todo Exception testing
        assert(positions.size() == sample_dimensions_.size());
        auto positions_out = std::vector<std::vector<float> >(positions.size());
        // Apply affine transform
        // NB!!! Because armadillo is column major, we use a column vector for each vector of
        // positions for a particular dimension, and therefore the transpose of the affine_
        // matrix
        arma::fmat augmented_positions(positions[0].size(), positions.size() + 1);
        { // Scope limiting
          auto vec_dim = 0;
          for (const auto& vec : positions){
            augmented_positions.col(vec_dim) = arma::fvec(vec);
            ++vec_dim;
          }
          augmented_positions.col(vec_dim) = arma::fvec(
              positions[0].size(),
              arma::fill::ones);
          augmented_positions = augmented_positions * affine_mat.t();
        } // Scope limiting
        { // Scope limiting
          auto vec_dim = 0;
          for (auto& vec : positions_out){
            vec = arma::conv_to<std::vector<float> >::from(augmented_positions.col(vec_dim));
            ++vec_dim;
          }
        } // Scope limiting
        return positions_out;
      }

      /// Calculate a scaling value that normalises to the robust mean
      /// \details By robust mean we refer to the mean of values which are part of the actual
      ///          object of interest. In order to do this, we first find the global mean of
      ///          the samples, and then find the mean of all samples with values greater
      ///          than 1/6th of the global mean
      float calculate_robust_norm_factor_(
        const std::vector<float>& image_samples,
        const std::vector<float>& ref_mask_samples,
        const std::vector<float>& mov_mask_samples) const
      {
        // First pass through, find global mean
        auto global_mean = 0.0;
        auto global_weighting = 0.0;
        for (auto i = 0; i < image_samples.size(); ++i){
          global_mean += image_samples[i]*ref_mask_samples[i]*mov_mask_samples[i];
          global_weighting += ref_mask_samples[i]*mov_mask_samples[i];
        }
        global_mean /= global_weighting;
        // Second pass through, find robust mean
        auto robust_mean = 0.0;
        auto robust_weighting = 0;
        for (auto i = 0; i < image_samples.size(); ++i){
          if (image_samples[i]*ref_mask_samples[i]*mov_mask_samples[i] > 0.17*global_mean){
            robust_mean += image_samples[i]*ref_mask_samples[i]*mov_mask_samples[i];
            robust_weighting += ref_mask_samples[i]*mov_mask_samples[i];
          }
        }
        robust_mean /= robust_weighting;
        // Set the normalisation factor such that the robust mean scales to 100
        auto robust_norm = static_cast<float>(100.0/robust_mean);
        return robust_norm;
      }

      /// Normalise a set of samples
      void normalise_samples_(std::vector<float>& samples, float norm_factor) const
      {
        for (auto& sample : samples){
          sample *= norm_factor;
        }
      }

      /// Attempt to correct for differences in contrast between the volumes which might
      /// confound our SSD calculations
      /// Create intensity mapper based on the initial alignment of the images
      void create_intensity_mapper_()
      {
        // Sample reference and moving volumes
        auto samples_ref = vol_ref_->sample(sample_positions_ref_);
        normalise_samples_(samples_ref,norm_factor_ref_);
        if (bias_field_){
          samples_ref = bias_field_->apply_bias(sample_positions_ref_, samples_ref);
        }
        auto sample_positions_warped = warp_field_->apply_warp(sample_positions_warp_);
        sample_positions_warped = affine_transform_(sample_positions_warped, affine_mov_);
        auto samples_mov = vol_mov_->sample(sample_positions_warped);
        normalise_samples_(samples_mov,norm_factor_mov_);
        // Create intensity mapper
        auto polynomial_degree = 3;
        intensity_mapper_ = MMORF::IntensityMapperPolynomial(
            polynomial_degree,
            samples_ref,
            samples_mov,
            debug_);
      }

      /// Calculate the derivitive of the moving volume wrt to changes in the x, y and z
      /// displacements in the shared reference space
      std::vector<std::vector<float> > calculate_derivatives_mov_(
          const std::vector<std::vector<float> >& positions,
          const arma::fmat& affine_mat) const
      {
        /// \todo replace assert with exception
        assert(positions.size() == sample_dimensions_.size());
        auto deriv_dim = positions.size();
        auto samples_dim = positions[0].size();
        // Create an armadillo matrix to store the samples as this will make life easier when
        // doing the affine transform
        auto derivatives_vec = vol_mov_->sample_gradient(positions);
        auto derivatives_arma = arma::fmat(positions[0].size(), positions.size());
        for (auto i = 0; i < deriv_dim; ++i){
          for (auto j = 0; j < samples_dim; ++j){
            derivatives_arma(j,i) = derivatives_vec[i][j];
          }
        }
        // Multiply derivative samples by the transposed affine matrix (excluding the offset)
        //
        // This step is a bit confusing for 2 reasons.
        //  1)  I'm post-multiplying by the affine. This is done because of Armadillo's
        //      column-major formatting. This makes the operation somewhat more efficient
        //      (although I am now dubious about this given the confusion it may cause).
        //      IMPORTANTLY this requires introducing a transpose into the affine.
        //
        //  2)  This is the more confusing part. The reason we multiply by the transpose
        //      affine is because of the relationship:
        //
        //            if:   f2(y) = f1(Ty)  <-- Here "T" is the warp-to-moving affine
        //            =>    grad(f2(y)) = transp(T)*grad(f1(Ty))
        //
        //            Where T is any linear transform (e.g. an affine matrix)
        //
        //      But because we've already introduced a transpose in 1), we undo the transpose,
        //      and therefore we simply post-multiply by the original affine.
        //
        derivatives_arma = derivatives_arma * affine_mat.submat(0,0,2,2);
        // Convert back to std::vector
        auto derivatives_return = std::vector<std::vector<float> >(positions.size());
        for (auto i = 0; i < derivatives_return.size(); ++i){
          derivatives_return[i] =
            arma::conv_to<std::vector<float> >::from(derivatives_arma.col(i));
        }
        return derivatives_return;
      }
      /// Add two positions together
      std::vector<std::vector<float> > add_positions_(
          const std::vector<std::vector<float> > positions_first,
          const std::vector<std::vector<float> > positions_second) const
      {
        // Check positions have the same dimensions
        assert(positions_first.size() == positions_second.size()
            && positions_first[0].size() == positions_second[0].size());
        auto positions_out = positions_first;
        for (auto i = 0; i < positions_out.size(); ++i){
          for (auto j = 0; j < positions_out[0].size(); ++j){
            positions_out[i][j] += positions_second[i][j];
          }
        }
        return positions_out;
      }
      // Convert 1D spline to vector of floats
      std::vector<float> spline_as_vec_(
          const BASISFIELD::Spline1D<float>& spline) const
      {
        auto spline_vals = std::vector<float>(spline.KernelSize());
        for (auto i = 0; i < spline.KernelSize(); ++i)
        {
          spline_vals.at(i) = spline(static_cast<unsigned int>(i+1));
        }
        return spline_vals;
      }
      // Create an empty JtJ matrix with the correct sparsity pattern
      MMORF::SparseDiagonalMatrixTiled create_empty_jtj_(
          const std::vector<int>& coef_sz) const
      {
        std::vector<int> offsets(343);
        for (unsigned int i = 0; i < 343; ++i)
        {
            unsigned int first_row, first_col, last_row, last_col;
            MMORF::identify_diagonal(i,7,7,7,coef_sz[0],coef_sz[1],coef_sz[2],
                                       &first_row,&last_row,&first_col,&last_col);
            if (first_col == 0){
              offsets.at(i) = -static_cast<int>(first_row);
            }
            else{
              offsets.at(i) = static_cast<int>(first_col);
            }
        }
        unsigned int max_diagonal = coef_sz[0]*coef_sz[1]*coef_sz[2];
        MMORF::SparseDiagonalMatrixTiled r_matrix(max_diagonal, max_diagonal, 3, offsets);
        return r_matrix;
      }
      // Calculate a sub-vector of Jte
      void calculate_sub_jte_(
          // Input
          const std::vector<float>& prod_ima,
          const std::vector<int>& ima_sz,
          const std::vector<float>& spline_x,
          const std::vector<float>& spline_y,
          const std::vector<float>& spline_z,
          const std::vector<int>& coef_sz,
          const std::vector<int>& ksp,
          // Output
          float* sub_jte_raw) const
      {
        // Make everything a device vector using thrust::
        auto prod_ima_dev = thrust::device_vector<float>(prod_ima);
        auto spline_x_dev = thrust::device_vector<float>(spline_x);
        auto spline_y_dev = thrust::device_vector<float>(spline_y);
        auto spline_z_dev = thrust::device_vector<float>(spline_z);
        // Create texture handle
        auto texture_handle = MMORF::TextureHandleLinear(prod_ima_dev,ima_sz);
        auto tex = texture_handle.get_texture();
        // Get all the raw pointers ready for the Kernel
        float *spline_x_raw = thrust::raw_pointer_cast(spline_x_dev.data());
        float *spline_y_raw = thrust::raw_pointer_cast(spline_y_dev.data());
        float *spline_z_raw = thrust::raw_pointer_cast(spline_z_dev.data());
        // Calculate parameters for running kernel
        int min_grid_size;
        int threads;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &threads,
            MMORF::kernel_make_jte,
            0,
            0);
        unsigned int blocks = static_cast<unsigned int>(
            std::ceil(float(coef_sz.at(0)*coef_sz.at(1)*coef_sz.at(2))/float(threads)));
        dim3 blocks_1d(blocks);
        unsigned int smem = (spline_x.size()+spline_y.size()+spline_z.size())*sizeof(float);
        // Call CUDA Kernel
        MMORF::kernel_make_jte<<<blocks_1d,threads,smem>>>(
            // Input
            ima_sz.at(0),
            ima_sz.at(1),
            ima_sz.at(2),
            tex,
            spline_x_raw,
            spline_y_raw,
            spline_z_raw,
            ksp.at(0),
            ksp.at(1),
            ksp.at(2),
            coef_sz.at(0),
            coef_sz.at(1),
            coef_sz.at(2),
            // Output
            sub_jte_raw);
        /// \todo Add error checking to cudaDeviceSynchronize
        checkCudaErrors(cudaDeviceSynchronize());
      }
      // Calclulate sub-matrix of jtj for symmetrical splines
      void calculate_sub_jtj_symmetrical_(
          const std::vector<float>& prod_ima,
          const std::vector<int>& ima_sz,
          const std::vector<float>& spline_x,
          const std::vector<float>& spline_y,
          const std::vector<float>& spline_z,
          const std::vector<int>& coef_sz,
          const std::vector<int>& ksp,
          const unsigned int sub_jtj_row,
          const unsigned int sub_jtj_col,
          MMORF::SparseDiagonalMatrixTiled& sparse_jtj) const
      {
        // Make everything a device vector using thrust::
        auto sparse_jtj_offsets_dev = thrust::device_vector<int>(sparse_jtj.get_offsets());
        auto prod_ima_dev = thrust::device_vector<float>(prod_ima);
        auto spline_x_dev = thrust::device_vector<float>(spline_x);
        auto spline_y_dev = thrust::device_vector<float>(spline_y);
        auto spline_z_dev = thrust::device_vector<float>(spline_z);
        // Create texture handle
        auto texture_handle = MMORF::TextureHandleLinear(prod_ima_dev,ima_sz);
        auto tex = texture_handle.get_texture();
        // Get all the raw pointers ready for the Kernel
        float *spline_x_raw = thrust::raw_pointer_cast(spline_x_dev.data());
        float *spline_y_raw = thrust::raw_pointer_cast(spline_y_dev.data());
        float *spline_z_raw = thrust::raw_pointer_cast(spline_z_dev.data());
        int *sparse_jtj_offsets_raw = thrust::raw_pointer_cast(
            sparse_jtj_offsets_dev.data());
        float *sparse_jtj_raw = sparse_jtj.get_raw_pointer(sub_jtj_row, sub_jtj_col);
        // Calculate parameters for running kernel
        int min_grid_size;
        int threads;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &threads,
            MMORF::kernel_make_jtj_symmetrical,
            0,
            0);
        unsigned int blocks = 172;
        unsigned int chunks = static_cast<unsigned int>(
            std::ceil(float(coef_sz.at(0)*coef_sz.at(1)*coef_sz.at(2))/float(threads)));
        dim3 blocks_2d(blocks,chunks);
        unsigned int smem = (spline_x.size()+spline_y.size()+spline_z.size())*sizeof(float);
        // Call CUDA Kernel
        MMORF::kernel_make_jtj_symmetrical<<<blocks_2d,threads,smem>>>(
            // Input
            ima_sz.at(0),
            ima_sz.at(1),
            ima_sz.at(2),
            tex,
            spline_x_raw,
            spline_y_raw,
            spline_z_raw,
            ksp.at(0),
            ksp.at(1),
            ksp.at(2),
            coef_sz.at(0),
            coef_sz.at(1),
            coef_sz.at(2),
            sparse_jtj_offsets_raw,
            // Output
            sparse_jtj_raw);
        /// \todo Add error checking to cudaDeviceSynchronize
        checkCudaErrors(cudaDeviceSynchronize());
      }
      //////////////////////////////////////////////////
      // Private datamembers
      //////////////////////////////////////////////////
      // Private datamembers
      std::shared_ptr<MMORF::Volume>           vol_ref_;
      std::shared_ptr<MMORF::Volume>           vol_mov_;
      arma::fmat                               affine_ref_;
      arma::fmat                               affine_mov_;
      std::shared_ptr<MMORF::Volume>           mask_ref_;
      std::shared_ptr<MMORF::Volume>           mask_mov_;
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field_;
      std::vector<int>                         sample_dimensions_;
      BASISFIELD::Spline1D<float>              spline_1D_;
      std::vector<std::vector<float> >         sample_positions_warp_;
      std::vector<std::vector<float> >         sample_positions_ref_;
      float                                    norm_factor_ref_;
      float                                    norm_factor_mov_;
      std::shared_ptr<MMORF::BiasFieldBSpline> bias_field_;
      MMORF::IntensityMapperPolynomial         intensity_mapper_;
      bool                                     debug_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  CostFxnSSDWarpFieldSymmetricMaskedExcluded::~CostFxnSSDWarpFieldSymmetricMaskedExcluded() = default;
  /// Move ctor
  CostFxnSSDWarpFieldSymmetricMaskedExcluded::CostFxnSSDWarpFieldSymmetricMaskedExcluded(CostFxnSSDWarpFieldSymmetricMaskedExcluded&& rhs) = default;
  /// Move assignment operator
  CostFxnSSDWarpFieldSymmetricMaskedExcluded& CostFxnSSDWarpFieldSymmetricMaskedExcluded::operator=(CostFxnSSDWarpFieldSymmetricMaskedExcluded&& rhs) = default;
  /// Copy ctor
  CostFxnSSDWarpFieldSymmetricMaskedExcluded::CostFxnSSDWarpFieldSymmetricMaskedExcluded(const CostFxnSSDWarpFieldSymmetricMaskedExcluded& rhs)
    : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  CostFxnSSDWarpFieldSymmetricMaskedExcluded& CostFxnSSDWarpFieldSymmetricMaskedExcluded::operator=(const CostFxnSSDWarpFieldSymmetricMaskedExcluded& rhs)
  {
    if (!rhs.pimpl_){
      pimpl_.reset();
    }
    else if (!pimpl_){
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
  /// Construct by passing in fully constructed volumes, warp and (optional) bias field
  /// \param vol_ref Reference (stationary) volume
  /// \param vol_mov Transformed volume
  /// \param affine_ref Pre-calculated affine transform FROM a common space TO the
  ///                   reference space
  /// \param affine_mov Pre-calculated affine transform FROM a common space TO the
  ///                   moving space
  /// \param mask_ref 3D volume used to mask the reference volume
  /// \param mask_mov 3D volume used to mask the moving volume
  /// \param warp_field Shared pointer to fully formed warp field
  /// \param sampling_frequency How often to sample between B-spline knots. E.g. for:
  ///             knot_spacing        = 10mm
  ///             sampling_frequency  = 5
  ///        warp field will be sampled every 2mm
  /// \param bias_field Shared pointer to fully formed bias field
  CostFxnSSDWarpFieldSymmetricMaskedExcluded::CostFxnSSDWarpFieldSymmetricMaskedExcluded(
          std::shared_ptr<MMORF::Volume>           vol_ref,
          std::shared_ptr<MMORF::Volume>           vol_mov,
          const arma::fmat&                        affine_ref,
          const arma::fmat&                        affine_mov,
          std::shared_ptr<MMORF::Volume>           mask_ref,
          std::shared_ptr<MMORF::Volume>           mask_mov,
          std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          int                                      sampling_frequency,
          std::shared_ptr<MMORF::BiasFieldBSpline> bias_field,
          const bool                               debug
          )
    : pimpl_(MMORF::make_unique<Impl>(
          vol_ref,
          vol_mov,
          affine_ref,
          affine_mov,
          mask_ref,
          mask_mov,
          warp_field,
          sampling_frequency,
          bias_field,
          debug)
        )
  {}
  /// Get the current value of the parameters
  std::vector<std::vector<float> > CostFxnSSDWarpFieldSymmetricMaskedExcluded::get_parameters() const
  {
    return pimpl_->get_parameters();
  }
  /// Set the current value of the parameters
  void CostFxnSSDWarpFieldSymmetricMaskedExcluded::set_parameters(
      const std::vector<std::vector<float> >& parameters)
  {
    pimpl_->set_parameters(parameters);
  }
  /// Get cost under current parameterisation
  float CostFxnSSDWarpFieldSymmetricMaskedExcluded::cost() const
  {
    return pimpl_->cost();
  }
  /// Get Jte under current parameterisation
  arma::fvec CostFxnSSDWarpFieldSymmetricMaskedExcluded::grad() const
  {
    return pimpl_->grad();
  }
  /// Get JtJ under current parameterisation
  MMORF::SparseDiagonalMatrixTiled CostFxnSSDWarpFieldSymmetricMaskedExcluded::hess() const
  {
    return pimpl_->hess();
  }
} // MMORF
