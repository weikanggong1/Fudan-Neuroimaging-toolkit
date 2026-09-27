//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the bending energy of a bias field
/// \details This cost function is designed primarily for use in regularising bias field
///          defined by B-splines, and not as a stand-alone cost function
/// \author Frederik Lange
/// \date March 2020
/// \copyright Copyright (C) 2020 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "CostFxnBendingEnergyBiasField.cuh"
#include "MmorfMemory.h"
#include "VolumeBSpline.cuh"
#include "BiasFieldBSpline.cuh"
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
  class CostFxnBendingEnergyBiasField::Impl
  {
    public:
      /// Construct by passing in a fully constructed bias field
      Impl(
          std::shared_ptr<MMORF::BiasFieldBSpline> bias_field,
          int sampling_frequency)
        : bias_field_(bias_field)
        , sampling_frequency_(sampling_frequency)
        , spline_1D_(3,sampling_frequency)
        , sparse_bkk_(0,0,0,std::vector<int>(0))
        , bkk_(bias_field_->get_parameter_size().second,
            bias_field_->get_parameter_size().second)
        , norm_factor_(1.0f)
      {
        // Calculate sample resolution
        auto bias_dims = bias_field_->get_dimensions();
        auto bias_extents = bias_field_->get_extents();
        for (auto i = 0; i < bias_dims.size(); ++i){
          auto dim_res =
            bias_field_->get_knot_spacing() / static_cast<float>(sampling_frequency);
          sample_resolution_.push_back(dim_res);
        }
        // Make bkk matrix
        make_bkk_();
        // Calculate bias sample resolution
        for (auto dim : bias_dims){
          norm_factor_ /= (dim*sampling_frequency);
        }
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
      // Cost is equal to bt*Bkk*b
      float cost() const
      {
        auto col_param = arma::fvec(bias_field_->get_parameters()[0]);
        col_param = col_param - 1.0f;
        //arma::fvec rhs_mult = bkk_*col_param;
        //float lhs_mult = arma::dot(col_param,rhs_mult);
        //auto cost = lhs_mult*norm_factor_;
        float cost = norm_factor_*arma::as_scalar(col_param.t()*bkk_*col_param);
        return cost;
      }
      /// Get grad under current parameterisation
      arma::fvec grad() const
      {
        auto col_param = arma::fvec(bias_field_->get_parameters()[0]);
        col_param = col_param - 1.0f;
        arma::fvec grad = 2*norm_factor_*bkk_*col_param;
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
      //////////////////////////////////////////////////
      // Private functions
      //////////////////////////////////////////////////
      /// Calculate the large matrix central to bending energy calculations
      void make_bkk_()
      {
        // Get size of bias field
        auto bias_sz = bias_field_->get_parameter_size();
        // Create sparse tiled matrix for storing Bkk
        sparse_bkk_ = create_empty_bkk_(bias_field_->get_dimensions());
        // Assume this is a 3D only volume
        for (auto dim_0 = 0; dim_0 < 3; ++dim_0){
          // Loop over all bias directions again
          for (auto dim_1 = 0; dim_1 < 3; ++dim_1){
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
              bias_field_->get_dimensions(),
              std::vector<int>(3, static_cast<int>(spline_1D_.KnotSpacing())),
              dim_0 == dim_1,
              sparse_bkk_);
            }
          }
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
        MMORF::SparseDiagonalMatrixTiled r_matrix(max_diagonal, max_diagonal, 1, offsets);
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

      //////////////////////////////////////////////////
      // Private datamembers
      //////////////////////////////////////////////////
      std::shared_ptr<MMORF::BiasFieldBSpline> bias_field_;
      BASISFIELD::Spline1D<float>              spline_1D_;
      int                                      sampling_frequency_;
      MMORF::SparseDiagonalMatrixTiled         sparse_bkk_;
      arma::sp_fmat                            bkk_;
      float                                    norm_factor_;
      std::vector<float>                       sample_resolution_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  CostFxnBendingEnergyBiasField::~CostFxnBendingEnergyBiasField() = default;
  /// Move ctor
  CostFxnBendingEnergyBiasField::CostFxnBendingEnergyBiasField(CostFxnBendingEnergyBiasField&& rhs) = default;
  /// Move assignment operator
  CostFxnBendingEnergyBiasField& CostFxnBendingEnergyBiasField::operator=(CostFxnBendingEnergyBiasField&& rhs) = default;
  /// Copy ctor
  CostFxnBendingEnergyBiasField::CostFxnBendingEnergyBiasField(const CostFxnBendingEnergyBiasField& rhs)
    : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  CostFxnBendingEnergyBiasField& CostFxnBendingEnergyBiasField::operator=(const CostFxnBendingEnergyBiasField& rhs)
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
  /// Construct by passing in a fully constructed biasfield
  CostFxnBendingEnergyBiasField::CostFxnBendingEnergyBiasField(
      std::shared_ptr<MMORF::BiasFieldBSpline> bias_field,
      int sampling_frequency)
    : pimpl_(MMORF::make_unique<Impl>(bias_field, sampling_frequency))
  {}
  /// Get the current value of the parameters
  std::vector<std::vector<float> > CostFxnBendingEnergyBiasField::get_parameters() const
  {
    return pimpl_->get_parameters();
  }
  /// Set the current value of the parameters
  void CostFxnBendingEnergyBiasField::set_parameters(
      const std::vector<std::vector<float> >& parameters)
  {
    pimpl_->set_parameters(parameters);
  }
  /// Get cost under current parameterisation
  float CostFxnBendingEnergyBiasField::cost() const
  {
    return pimpl_->cost();
  }
  /// Get Jte under current parameterisation
  arma::fvec CostFxnBendingEnergyBiasField::grad() const
  {
    return pimpl_->grad();
  }
  /// Get JtJ under current parameterisation
  MMORF::SparseDiagonalMatrixTiled CostFxnBendingEnergyBiasField::hess() const
  {
    return pimpl_->hess();
  }
} // MMORF
