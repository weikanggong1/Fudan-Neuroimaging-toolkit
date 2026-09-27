//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the sum of squared differences between two volumes, one of
///        which is warped
/// \details The warp is defined by b-spline field coefficients, one set per dimension in the
///          volume
/// \author Frederik Lange
/// \date April 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "CostFxnBendingEnergy.cuh"
#include "MmorfMemory.h"
#include "VolumeBSpline.cuh"
#include "WarpFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"
#include "TextureHandleLinear.cuh"
#include "CostFxnHelpers.cuh"
#include "CostFxnKernels.cuh"
#include "helper_cuda.h"

#include "basisfield/fsl_splines.h"

#include <armadillo>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include <memory>
#include <vector>
#include <utility>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class CostFxnBendingEnergy::Impl
  {
    public:
      /// Construct by passing in a fully constructed warpfield
      Impl(
          std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          int sampling_frequency)
        : warp_field_(warp_field)
        , sampling_frequency_(sampling_frequency)
        , spline_1D_(3,sampling_frequency)
        , bkk_(warp_field_->get_parameter_size().second,
            warp_field_->get_parameter_size().second)
        , sparse_bkk_(0,0,0,std::vector<int>(0))
        , norm_factor_(1.0f)
      {
        // Calculate sample resolution
        auto warp_dims = warp_field_->get_dimensions();
        auto warp_extents = warp_field_->get_extents();
        for (auto i = 0; i < warp_dims.size(); ++i){
          auto dim_res =
            warp_field_->get_knot_spacing() / static_cast<float>(sampling_frequency);
          sample_resolution_.push_back(dim_res);
        }
        // Make bkk matrix
        make_bkk_();
        // Calculate warp sample resolution
        for (auto dim :warp_dims){
          norm_factor_ /= (dim*sampling_frequency);
        }
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
      // Cost is equal to bt*Bkk*b
      float cost() const
      {
        // Calculate the values of the local Jacobian at all sample points
        auto sample_positions = warp_field_->get_robust_sample_positions(
            sampling_frequency_);
        // Check to make sure that the field hasn't gone non-diffeomorphic. If any Jacobian
        // determinant is negative, return the cost as infinity
        auto jacobian_determinants = warp_field_->jacobian_determinants(sample_positions);
        auto min_element_it = std::min_element(
            jacobian_determinants.begin(),
            jacobian_determinants.end());
        std::cout << "Minimum Jacobian is: " << *min_element_it << std::endl;
        if (*min_element_it < 0.0f){
          return std::numeric_limits<float>::infinity();
        }
        // If we're sure everything is still diffeomorphic, then calculate the cost
        else{
          auto col_param = arma::fvec();
          // Concatenate warp parameters
          for (const auto& warp_params : warp_field_->get_parameters()){
            col_param = arma::join_cols(col_param,arma::fvec(warp_params));
          }
          arma::fvec rhs_mult = bkk_*col_param;
          float lhs_mult = arma::dot(col_param,rhs_mult);
          auto cost = lhs_mult*norm_factor_;
          return cost;
        }
      }
      /// Get grad under current parameterisation
      arma::fvec grad() const
      {
        auto col_param = arma::fvec();
        // Concatenate warp parameters
        for (const auto& warp_params : warp_field_->get_parameters()){
          col_param = arma::join_cols(col_param,arma::fvec(warp_params));
        }
        arma::fvec grad = 2*bkk_*col_param*norm_factor_;
        return grad;
      }
      /// Get hess under current parameterisation
      MMORF::SparseDiagonalMatrixTiled hess() const
      {
        auto hess = sparse_bkk_;
        hess *= (2*norm_factor_);
        return hess;
      }
    private:
      /// Calculate the large matrix central to bending energy calculations
      void make_bkk_()
      {
        // Get size of warp field
        auto warp_sz = warp_field_->get_parameter_size();
        // Create sparse tiled matrix for storing Bkk
        sparse_bkk_ = create_empty_bkk_(warp_field_->get_dimensions());
        // Loop over all warp directions, assuming this corresponds to dimensionality of the
        // warped volume
        for (auto dim_0 = 0; dim_0 < warp_sz.first; ++dim_0){
          // Loop over all warp directions again
          for (auto dim_1 = 0; dim_1 < warp_sz.first; ++dim_1){
            // Check if this is a unique combination - we will keep the main diagonal and below
            if (dim_1 > dim_0) continue;
            // Generate differentiated splines
            auto x_deriv_order = 0;
            auto y_deriv_order = 0;
            auto z_deriv_order = 0;
            x_deriv_order = dim_0 == 0 ? x_deriv_order + 1 : x_deriv_order;
            x_deriv_order = dim_1 == 0 ? x_deriv_order + 1 : x_deriv_order;
            y_deriv_order = dim_0 == 1 ? y_deriv_order + 1 : y_deriv_order;
            y_deriv_order = dim_1 == 1 ? y_deriv_order + 1 : y_deriv_order;
            z_deriv_order = dim_0 == 2 ? z_deriv_order + 1 : z_deriv_order;
            z_deriv_order = dim_1 == 2 ? z_deriv_order + 1 : z_deriv_order;
            auto spline_x = BASISFIELD::Spline1D<float>(
                spline_1D_.Order(),
                spline_1D_.KnotSpacing(),
                x_deriv_order);
            auto spline_y = BASISFIELD::Spline1D<float>(
                spline_1D_.Order(),
                spline_1D_.KnotSpacing(),
                y_deriv_order);
            auto spline_z = BASISFIELD::Spline1D<float>(
                spline_1D_.Order(),
                spline_1D_.KnotSpacing(),
                z_deriv_order);
            auto spline_x_vals = spline_as_vec_(
                spline_x,
                sample_resolution_[0],
                x_deriv_order);
            auto spline_y_vals = spline_as_vec_(
                spline_y,
                sample_resolution_[1],
                y_deriv_order);
            auto spline_z_vals = spline_as_vec_(
                spline_z,
                sample_resolution_[2],
                z_deriv_order);
            // Call kernel
            // Is this a unique sub-bkk? (i.e. same dimension differentiated)
            // Otherwise, we need to add this sub-bkk twice (once for dim_0 > dim_1, and once
            // for dim_0 < dim_1 which give the same result)
            calculate_sub_bkk_(
              // Input
              spline_x_vals,
              spline_y_vals,
              spline_z_vals,
              warp_field_->get_dimensions(),
              std::vector<int>(3, static_cast<int>(spline_1D_.KnotSpacing())),
              dim_0 == dim_1,
              sparse_bkk_);
            }
          }
          // Now copy the sub-matrix within the sparse tiled matrix
          sparse_bkk_.copy_submatrix(0,0,1,1);
          sparse_bkk_.copy_submatrix(0,0,2,2);
          // Create armadillo matrix
          bkk_ = sparse_bkk_.convert_to_csc();
      }
      // Convert 1D spline to vector of floats
      // Note that this function takes care of correcting the derivative scaling if the
      // spline_res parameter is used. This parameter is the resolution between each value
      // in the spline
      std::vector<float> spline_as_vec_(
          const BASISFIELD::Spline1D<float>& spline,
          float spline_res = 1.0f,
          int diff_order = 0) const
      {
        auto spline_vals = std::vector<float>(spline.KernelSize());
        for (auto i = 0; i < spline.KernelSize(); ++i)
        {
          spline_vals.at(i) =
            spline(static_cast<unsigned int>(i+1))
            / std::pow(spline_res, diff_order);
        }
        return spline_vals;
      }
      // Create an empty Bkk matrix with the correct sparsity pattern
      MMORF::SparseDiagonalMatrixTiled create_empty_bkk_(
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
      // Calclulate sub-matrix of Bkk for symmetrical splines
      void calculate_sub_bkk_(
          const std::vector<float>& spline_x,
          const std::vector<float>& spline_y,
          const std::vector<float>& spline_z,
          const std::vector<int>& coef_sz,
          const std::vector<int>& ksp,
          const bool dims_equal,
          MMORF::SparseDiagonalMatrixTiled& sparse_bkk) const
      {
        // Make everything a device vector using thrust::
        auto sparse_bkk_offsets_dev = thrust::device_vector<int>(sparse_bkk.get_offsets());
        auto spline_x_dev = thrust::device_vector<float>(spline_x);
        auto spline_y_dev = thrust::device_vector<float>(spline_y);
        auto spline_z_dev = thrust::device_vector<float>(spline_z);
        // Get all the raw pointers ready for the Kernel
        float *spline_x_raw = thrust::raw_pointer_cast(spline_x_dev.data());
        float *spline_y_raw = thrust::raw_pointer_cast(spline_y_dev.data());
        float *spline_z_raw = thrust::raw_pointer_cast(spline_z_dev.data());
        int *sparse_bkk_offsets_raw = thrust::raw_pointer_cast(
            sparse_bkk_offsets_dev.data());
        float *sparse_bkk_raw = sparse_bkk.get_raw_pointer(0,0);
        // Calculate parameters for running kernel
        int min_grid_size;
        int threads;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &threads,
            MMORF::kernel_make_bkk,
            0,
            0);
        unsigned int blocks = 172;
        unsigned int chunks = static_cast<unsigned int>(
            std::ceil(float(coef_sz.at(0)*coef_sz.at(1)*coef_sz.at(2))/float(threads)));
        dim3 blocks_2d(blocks,chunks);
        unsigned int smem = (spline_x.size()+spline_y.size()+spline_z.size())*sizeof(float);
        // Call CUDA Kernel
      MMORF::kernel_make_bkk<<<blocks_2d,threads,smem>>>(
            // Input
            spline_x_raw,
            spline_y_raw,
            spline_z_raw,
            ksp.at(0),
            ksp.at(1),
            ksp.at(2),
            coef_sz.at(0),
            coef_sz.at(1),
            coef_sz.at(2),
            sparse_bkk_offsets_raw,
            dims_equal,

            // Output
            sparse_bkk_raw);
        /// \todo Add error checking to cudaDeviceSynchronize
        checkCudaErrors(cudaDeviceSynchronize());
      }
      // Private datamembers
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field_;
      BASISFIELD::Spline1D<float> spline_1D_;
      int sampling_frequency_;
      arma::sp_fmat bkk_;
      MMORF::SparseDiagonalMatrixTiled sparse_bkk_;
      float norm_factor_;
      std::vector<float> sample_resolution_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  CostFxnBendingEnergy::~CostFxnBendingEnergy() = default;
  /// Move ctor
  CostFxnBendingEnergy::CostFxnBendingEnergy(CostFxnBendingEnergy&& rhs) = default;
  /// Move assignment operator
  CostFxnBendingEnergy& CostFxnBendingEnergy::operator=(CostFxnBendingEnergy&& rhs) = default;
  /// Copy ctor
  CostFxnBendingEnergy::CostFxnBendingEnergy(const CostFxnBendingEnergy& rhs)
    : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  CostFxnBendingEnergy& CostFxnBendingEnergy::operator=(const CostFxnBendingEnergy& rhs)
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
  /// Construct by passing in a fully constructed warpfield
  CostFxnBendingEnergy::CostFxnBendingEnergy(
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
      int sampling_frequency)
    : pimpl_(MMORF::make_unique<Impl>(warp_field, sampling_frequency))
  {}
  /// Get the current value of the parameters
  std::vector<std::vector<float> > CostFxnBendingEnergy::get_parameters() const
  {
    return pimpl_->get_parameters();
  }
  /// Set the current value of the parameters
  void CostFxnBendingEnergy::set_parameters(
      const std::vector<std::vector<float> >& parameters)
  {
    pimpl_->set_parameters(parameters);
  }
  /// Get cost under current parameterisation
  float CostFxnBendingEnergy::cost() const
  {
    return pimpl_->cost();
  }
  /// Get Jte under current parameterisation
  arma::fvec CostFxnBendingEnergy::grad() const
  {
    return pimpl_->grad();
  }
  /// Get JtJ under current parameterisation
  MMORF::SparseDiagonalMatrixTiled CostFxnBendingEnergy::hess() const
  {
    return pimpl_->hess();
  }
} // MMORF
