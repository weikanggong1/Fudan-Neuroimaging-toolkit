//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the sum of squared differences between two volumes, one of
///        which is warped. Additionally, a mask in the reference image domain is used to
///        constrain the area of interest during optimisaton
/// \details The warp is defined by b-spline field coefficients, one set per dimension in the
///          volume
/// \author Frederik Lange
/// \date August 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "CostFxnSSDBiasFieldMaskedExcluded.cuh"
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
  class CostFxnSSDBiasFieldMaskedExcluded::Impl
  {
    public:
      //////////////////////////////////////////////////
      // Public functions
      //////////////////////////////////////////////////
      /// Construct by passing in fully constructed volumes, and defining warp field
      /// characteristics
      /// \param vol_ref Reference (stationary) volume
      /// \param vol_mov Transformed volume
      /// \param affine_ref Pre-calculated affine transform FROM a common space TO the
      ///                   reference space
      /// \param affine_mov Pre-calculated affine transform FROM a common space TO the
      ///                   moving space
      /// \param mask_ref 3D volume used to mask the reference volume
      /// \param mask_mov 3D volume used to mask the moving volume
      /// \param warp_field Warp field to resample vol_mov
      /// \param bias_field Bias field to apply to vol_ref
      /// \param sampling_frequency How often to sample between B-spline knots. E.g. for:
      ///             knot_spacing        = 10mm
      ///             sampling_frequency  = 5
      ///        bias field will be sampled every 2mm
      Impl(
          std::shared_ptr<MMORF::Volume>                 vol_ref,
          std::shared_ptr<MMORF::Volume>                 vol_mov,
          const arma::fmat&                              affine_ref,
          const arma::fmat&                              affine_mov,
          std::shared_ptr<MMORF::Volume>                 mask_ref,
          std::shared_ptr<MMORF::Volume>                 mask_mov,
          const std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          std::shared_ptr<MMORF::BiasFieldBSpline>       bias_field,
          int                                            sampling_frequency,
          const bool                                     debug,
          const int                                      id
          )
        : vol_ref_(vol_ref)
        , vol_mov_(vol_mov)
        , affine_ref_(affine_ref)
        , affine_mov_(affine_mov)
        , mask_ref_(mask_ref)
        , mask_mov_(mask_mov)
        , warp_field_(warp_field)
        , bias_field_(bias_field)
        , sample_dimensions_(bias_field_->get_robust_sample_dimensions(sampling_frequency))
        , spline_1D_(3,sampling_frequency) // Cubic spline with the correct no. samples
        , debug_(debug)
        , id_(id)
        , count_cost_(0)
        , count_grad_(0)
        , count_hess_(0)
      {
        if (debug_){
          auto tmp_debug_dir = "mkdir -p " + debug_dir_ + std::to_string(id_);
          auto tmp_sysval = std::system(tmp_debug_dir.c_str());
        }
        /// \todo Replace assert with exception
        assert(sampling_frequency > 0);
        sample_positions_bias_ = bias_field_->get_robust_sample_positions(sampling_frequency);
        sample_positions_ref_ = affine_transform_(sample_positions_bias_, affine_ref_);
        // Calculate normalisation values for both volumes
        auto sample_positions_mov = affine_transform_(sample_positions_bias_, affine_mov);
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
      }

      /// Get the current value of the parameters
      std::vector<std::vector<float> > get_parameters() const
      {
        return bias_field_->get_parameters();
      }

      /// Set the current value of the parameters
      void set_parameters(
          const std::vector<std::vector<float> >& parameters)
      {
        bias_field_->set_parameters(parameters);
      }

      /// Get cost under current parameterisation
      float cost() const
      {
        // Sample reference and moving volumes
        auto samples_ref = vol_ref_->sample(sample_positions_ref_);
        normalise_samples_(samples_ref,norm_factor_ref_);
        samples_ref = bias_field_->apply_bias(sample_positions_bias_, samples_ref);
        auto sample_positions_warped = warp_field_->apply_warp(sample_positions_bias_);
        sample_positions_warped = affine_transform_(sample_positions_warped, affine_mov_);
        auto samples_mov = vol_mov_->sample(sample_positions_warped);
        normalise_samples_(samples_mov,norm_factor_mov_);
        // Calculate cost from samples
        auto samples_error = std::vector<float>(samples_ref.size());
        std::transform(
            samples_mov.begin(),
            samples_mov.end(),
            samples_ref.begin(),
            samples_error.begin(),
            std::minus<float>()
            );
        auto cost = std::inner_product(
            samples_error.begin(),
            samples_error.end(),
            samples_error.begin(),
            0.0f);
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
        // Avergage cost
        cost /= samples_ref.size();
        return cost;
      }

      /// Get Jte under current parameterisation
      arma::fvec grad() const
      {
        // Calculate jte_sz
        auto bias_field_sz = bias_field_->get_parameter_size();
        auto jte_sz = bias_field_sz.first * bias_field_sz.second;
        // Make a device vector for storing the result
        auto jte_dev = thrust::device_vector<float>(jte_sz);
        // Create and sample error volume
        // Sample reference and moving volumes
        auto samples_ref = vol_ref_->sample(sample_positions_ref_);
        normalise_samples_(samples_ref,norm_factor_ref_);
        auto samples_ref_biased = bias_field_->apply_bias(sample_positions_bias_, samples_ref);
        auto sample_positions_warped = warp_field_->apply_warp(sample_positions_bias_);
        sample_positions_warped = affine_transform_(sample_positions_warped, affine_mov_);
        auto samples_mov = vol_mov_->sample(sample_positions_warped);
        normalise_samples_(samples_mov,norm_factor_mov_);
        // Calculate current sample error
        auto samples_error = std::vector<float>(samples_ref.size());
        std::transform(
            samples_ref_biased.begin(),
            samples_ref_biased.end(),
            samples_mov.begin(),
            samples_error.begin(),
            std::minus<float>()
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
        // Pre-multiply the error and reference samples
        auto samples_pre_mult = std::vector<float>(samples_error.size());
        std::transform(
            samples_error.begin(),
            samples_error.end(),
            samples_ref.begin(),
            samples_pre_mult.begin(),
            std::multiplies<float>()
            );
        /// \todo Can I replace this with jte_dev.begin() ?
        auto jte_dev_ptr = thrust::raw_pointer_cast(&jte_dev[0]);
        calculate_sub_jte_(
            // Input
            samples_pre_mult,
            sample_dimensions_,
            spline_vec,
            spline_vec,
            spline_vec,
            bias_field_->get_dimensions(),
            std::vector<int>(3, static_cast<int>(spline_1D_.KnotSpacing())),
            // Output
            jte_dev_ptr);
        // Copy jte back to host
        auto jte_host = thrust::host_vector<float>(jte_dev);
        // Return result
        auto jte_return = arma::fvec(jte_host.data(), jte_host.size());
        jte_return = (2.0f / static_cast<float>(samples_ref.size())) * jte_return;
        if (debug_){
          // Create folder
          auto debug_dir =
            debug_dir_ +
            std::to_string(id_) + "/grad/" +
            std::to_string(count_grad_++);
          auto mkdir_command =
            "mkdir -p " +
            debug_dir;
          auto tmp_sysval = std::system(mkdir_command.c_str());
          // Save grad
          jte_return.save(debug_dir + "/grad", arma::raw_ascii);
        }
        return jte_return;
      }

      /// Get JtJ under current parameterisation
      MMORF::SparseDiagonalMatrixTiled hess() const
      {
        // Get size of bias field
        auto bias_field_sz = bias_field_->get_parameter_size();
        auto samples_ref = vol_ref_->sample(sample_positions_ref_);
        normalise_samples_(samples_ref,norm_factor_ref_);
        // Convert spline to vec
        auto spline_vec = spline_as_vec_(spline_1D_);
        // Create sparse tiled matrix to store Hessian
        auto sparse_jtj = create_empty_jtj_(bias_field_->get_dimensions());
        // Square the ref image
        std::transform(
            samples_ref.begin(),
            samples_ref.end(),
            samples_ref.begin(),
            samples_ref.begin(),
            std::multiplies<float>()
            );
        if (mask_ref_){
            auto samples_mask_ref = mask_ref_->sample(sample_positions_ref_);
            std::transform(
                samples_ref.begin(),
                samples_ref.end(),
                samples_mask_ref.begin(),
                samples_ref.begin(),
                std::multiplies<float>()
                );
        }
        if (mask_mov_){
            auto sample_positions_warped = warp_field_->apply_warp(sample_positions_bias_);
            sample_positions_warped = affine_transform_(sample_positions_warped, affine_mov_);
            auto samples_mask_mov = mask_mov_->sample(sample_positions_warped);
            std::transform(
                samples_ref.begin(),
                samples_ref.end(),
                samples_mask_mov.begin(),
                samples_ref.begin(),
                std::multiplies<float>()
                );
        }

        // Call kernel
        calculate_sub_jtj_symmetrical_(
          // Input
          samples_ref,
          sample_dimensions_,
          spline_vec,
          spline_vec,
          spline_vec,
          bias_field_->get_dimensions(),
          std::vector<int>(3, static_cast<int>(spline_1D_.KnotSpacing())),
          sparse_jtj);
        // Return completed matrix
        sparse_jtj *= (2.0f / static_cast<float>(samples_ref.size()));
        // Debugging
        if (debug_){
          // Create folder
          auto debug_dir =
            debug_dir_ +
            std::to_string(id_) + "/hess/" +
            std::to_string(count_hess_++);
          auto mkdir_command =
            "mkdir -p " +
            debug_dir;
          auto tmp_sysval = std::system(mkdir_command.c_str());
          // Save hess
          sparse_jtj.convert_to_csc().save(debug_dir + "/hess", arma::coord_ascii);
        }
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
      MMORF::IntensityMapperPolynomial create_intensity_mapper_(
          const std::vector<float>& samples_domain,
          const std::vector<float>& samples_range,
          const std::vector<float>& samples_mask) const
      {
        // Extract non-masked-out values
        auto samples_domain_masked = std::vector<float>();
        auto samples_range_masked = std::vector<float>();
        for (auto i = 0; i < samples_mask.size(); ++i){
          if (samples_mask[i] > 0.1f){ // Using 0.1 here just to limit what is valid
            samples_domain_masked.push_back(samples_domain[i]);
            samples_range_masked.push_back(samples_range[i]);
          }
        }
        // Create intensity mapper
        auto polynomial_degree = 3;
        auto intensity_mapper = MMORF::IntensityMapperPolynomial(
            polynomial_degree,
            samples_domain_masked,
            samples_range_masked);
        // Resample domain
        return intensity_mapper;
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
        MMORF::SparseDiagonalMatrixTiled r_matrix(max_diagonal, max_diagonal, 1, offsets);
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
        float *sparse_jtj_raw = sparse_jtj.get_raw_pointer(0,0);
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
      std::shared_ptr<MMORF::BiasFieldBSpline> bias_field_;
      std::vector<int>                         sample_dimensions_;
      BASISFIELD::Spline1D<float>              spline_1D_;
      std::vector<std::vector<float> >         sample_positions_bias_;
      std::vector<std::vector<float> >         sample_positions_ref_;
      float                                    norm_factor_ref_;
      float                                    norm_factor_mov_;
      // Debug only datamembers
      bool                                     debug_;
      int                                      id_;
      mutable int                              count_cost_;
      mutable int                              count_grad_;
      mutable int                              count_hess_;
      const static std::string                 debug_dir_;
  };

  /// Static debug directory initialisation
  const std::string CostFxnSSDBiasFieldMaskedExcluded::Impl::debug_dir_ =
    "debug/CostFxnSSDBiasFieldMaskedExcluded/";

////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  CostFxnSSDBiasFieldMaskedExcluded::~CostFxnSSDBiasFieldMaskedExcluded() = default;
  /// Move ctor
  CostFxnSSDBiasFieldMaskedExcluded::CostFxnSSDBiasFieldMaskedExcluded(CostFxnSSDBiasFieldMaskedExcluded&& rhs) = default;
  /// Move assignment operator
  CostFxnSSDBiasFieldMaskedExcluded& CostFxnSSDBiasFieldMaskedExcluded::operator=(CostFxnSSDBiasFieldMaskedExcluded&& rhs) = default;
  /// Copy ctor
  CostFxnSSDBiasFieldMaskedExcluded::CostFxnSSDBiasFieldMaskedExcluded(const CostFxnSSDBiasFieldMaskedExcluded& rhs)
    : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  CostFxnSSDBiasFieldMaskedExcluded& CostFxnSSDBiasFieldMaskedExcluded::operator=(const CostFxnSSDBiasFieldMaskedExcluded& rhs)
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
  /// Construct by passing in fully constructed volumes, and defining warp field
  /// characteristics
  /// \param vol_ref Reference (stationary) volume
  /// \param vol_mov Transformed volume
  /// \param affine_ref Pre-calculated affine transform FROM a common space TO the
  ///                   reference space
  /// \param affine_mov Pre-calculated affine transform FROM a common space TO the
  ///                   moving space
  /// \param mask_ref 3D volume used to mask the reference volume
  /// \param mask_mov 3D volume used to mask the moving volume
  /// \param knot_spacing Bias field B-spline knot spacing (mm) in reference space
  /// \param sampling_frequency How often to sample between B-spline knots. E.g. for:
  ///             knot_spacing        = 10mm
  ///             sampling_frequency  = 5
  ///        warp field will be sampled every 2mm
  CostFxnSSDBiasFieldMaskedExcluded::CostFxnSSDBiasFieldMaskedExcluded(
          std::shared_ptr<MMORF::Volume>                 vol_ref,
          std::shared_ptr<MMORF::Volume>                 vol_mov,
          const arma::fmat&                              affine_ref,
          const arma::fmat&                              affine_mov,
          std::shared_ptr<MMORF::Volume>                 mask_ref,
          std::shared_ptr<MMORF::Volume>                 mask_mov,
          const std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          std::shared_ptr<MMORF::BiasFieldBSpline>       bias_field,
          int                                            sampling_frequency,
          const bool                                     debug
          )
    : pimpl_(MMORF::make_unique<Impl>(
          vol_ref,
          vol_mov,
          affine_ref,
          affine_mov,
          mask_ref,
          mask_mov,
          warp_field,
          bias_field,
          sampling_frequency,
          debug,
          next_id_++)
        )
  {}

  /// Get the current value of the parameters
  std::vector<std::vector<float> > CostFxnSSDBiasFieldMaskedExcluded::get_parameters() const
  {
    return pimpl_->get_parameters();
  }

  /// Set the current value of the parameters
  void CostFxnSSDBiasFieldMaskedExcluded::set_parameters(
      const std::vector<std::vector<float> >& parameters)
  {
    pimpl_->set_parameters(parameters);
  }

  /// Get cost under current parameterisation
  float CostFxnSSDBiasFieldMaskedExcluded::cost() const
  {
    return pimpl_->cost();
  }

  /// Get Jte under current parameterisation
  arma::fvec CostFxnSSDBiasFieldMaskedExcluded::grad() const
  {
    return pimpl_->grad();
  }

  /// Get JtJ under current parameterisation
  MMORF::SparseDiagonalMatrixTiled CostFxnSSDBiasFieldMaskedExcluded::hess() const
  {
    return pimpl_->hess();
  }

  /// Static datamember initialisation
  int CostFxnSSDBiasFieldMaskedExcluded::next_id_ = 0;
} // MMORF
