//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the log of the singular values of the local Jacobian of the
///        warp field
/// \details This cost function is designed primarily for use in regularising warps defined
///          by B-splines, and not as a stand-alone cost function. As the analytical forms of
///          the gradient and Hessian are rather complicated, the code to calculate them was
///          formulated with the help of the Matlab Symbolic Toolbox.
///          Note that the Hessian calculation uses an approximation whereby it is represented
///          by a single main diagonal made up of the sum of absolute values of each
///          row/column (which is the same thing as H is symmetrical).
/// \author Frederik Lange
/// \date July 2019
/// \copyright Copyright (C) 2019 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "CostFxnLogJacobianSingularValuesDiagHess.cuh"
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
#include <thrust/transform.h>
#include <thrust/functional.h>

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
  class CostFxnLogJacobianSingularValuesDiagHess::Impl
  {
    public:
      //////////////////////////////////////////////////
      // Public functions
      //////////////////////////////////////////////////
      /// Construct by passing in a fully constructed warpfield
      Impl(
          std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          int sampling_frequency)
        : warp_field_(warp_field)
        , sampling_frequency_(sampling_frequency)
        , spline_1D_(3,sampling_frequency)
        , norm_factor_(1.0e3f)
      {
        // Calculate sample resolution
        auto warp_dims = warp_field_->get_dimensions();
        auto warp_extents = warp_field_->get_extents();
        for (auto i = 0; i < warp_dims.size(); ++i){
          auto dim_res =
            warp_field_->get_knot_spacing() / static_cast<float>(sampling_frequency);
          sample_resolution_.push_back(dim_res);
        }
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
      /// \details The cost in this case is equal to the sum of the square of the log of the
      ///          singular values at each point in the volume
      float cost() const
      {
        auto cost_out = 0.0f;
        // Calculate the values of the local Jacobian at all sample points
        auto sample_positions = warp_field_->get_robust_sample_positions(
            sampling_frequency_);
        // Check to make sure that the field hasn't gone non-diffeomorphic. If any Jacobian
        // determinant is negative, return the cost as infinity
        auto jacobian_determinants = warp_field_->jacobian_determinants(
            sample_positions);
        auto min_element_it = std::min_element(
            jacobian_determinants.begin(),
            jacobian_determinants.end());
        std::cout << "Minimum Jacobian is: " << *min_element_it << std::endl;
        if (*min_element_it < 1e-3f || !std::isfinite(*min_element_it)){
          cost_out= std::numeric_limits<float>::infinity();
        }
        // If we're sure everything is still diffeomorphic, then calculate the cost
        else{
          auto jacobian_elements = warp_field_->get_jacobian_elements(sample_positions);
          auto cost_per_sample = calculate_cost_per_sample_(jacobian_elements);
          auto cost = std::accumulate(cost_per_sample.begin(), cost_per_sample.end(), 0.0f);
          cost_out = norm_factor_*cost;
        }
        return cost_out;
      }
      /// Get grad under current parameterisation
      arma::fvec grad() const
      {
        // Calculate grad per jacobian element
        auto sample_positions = warp_field_->get_robust_sample_positions(
            sampling_frequency_);
        auto jacobian_elements = warp_field_->get_jacobian_elements(sample_positions);
        auto grad_per_jacobian_element = calculate_grad_per_jacobian_element_(
            jacobian_elements);
        // Calculate grad per spline
        auto grad_per_spline = calculate_grad_per_spline_(grad_per_jacobian_element);
        return norm_factor_*grad_per_spline;
      }
      /// Get hess under current parameterisation
      MMORF::SparseDiagonalMatrixTiled hess() const
      {
        // Calculate cost & grad per jacobian element
        auto sample_positions = warp_field_->get_robust_sample_positions(
            sampling_frequency_);
        auto jacobian_elements = warp_field_->get_jacobian_elements(sample_positions);
        auto cost_per_sample = calculate_cost_per_sample_(jacobian_elements);
        auto grad_per_jacobian_element = calculate_grad_per_jacobian_element_(
            jacobian_elements);
        auto hess_per_spline = calculate_hess_per_spline_(
            cost_per_sample,
            grad_per_jacobian_element);
        hess_per_spline *= norm_factor_;
        return hess_per_spline;
      }
    private:
      //////////////////////////////////////////////////
      // Private functions
      //////////////////////////////////////////////////
      /// Calculate the contribution to the cost per sample position
      std::vector<float> calculate_cost_per_sample_(
          const std::vector<std::vector<float> >& jacobian_elements) const
      {
        auto n_samples = jacobian_elements[0].size();
        // Create device vectors using thrust::
        auto j11_dev = thrust::device_vector<float>(jacobian_elements[0]);
        auto j12_dev = thrust::device_vector<float>(jacobian_elements[1]);
        auto j13_dev = thrust::device_vector<float>(jacobian_elements[2]);
        auto j21_dev = thrust::device_vector<float>(jacobian_elements[3]);
        auto j22_dev = thrust::device_vector<float>(jacobian_elements[4]);
        auto j23_dev = thrust::device_vector<float>(jacobian_elements[5]);
        auto j31_dev = thrust::device_vector<float>(jacobian_elements[6]);
        auto j32_dev = thrust::device_vector<float>(jacobian_elements[7]);
        auto j33_dev = thrust::device_vector<float>(jacobian_elements[8]);
        auto cost_per_sample_dev = thrust::device_vector<float>(n_samples);
        // Get raw pointers to device data
        auto j11_raw = thrust::raw_pointer_cast(j11_dev.data());
        auto j12_raw = thrust::raw_pointer_cast(j12_dev.data());
        auto j13_raw = thrust::raw_pointer_cast(j13_dev.data());
        auto j21_raw = thrust::raw_pointer_cast(j21_dev.data());
        auto j22_raw = thrust::raw_pointer_cast(j22_dev.data());
        auto j23_raw = thrust::raw_pointer_cast(j23_dev.data());
        auto j31_raw = thrust::raw_pointer_cast(j31_dev.data());
        auto j32_raw = thrust::raw_pointer_cast(j32_dev.data());
        auto j33_raw = thrust::raw_pointer_cast(j33_dev.data());
        auto cost_per_sample_raw = thrust::raw_pointer_cast(cost_per_sample_dev.data());
        // Calculate kernel parameters
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_log_jacobian_singular_values_cost_3d,
            0,
            0);
        unsigned int grid_size = (n_samples + block_size - 1)/block_size;
        // Call CUDA Kernel
        MMORF::kernel_log_jacobian_singular_values_cost_3d<<<grid_size, block_size>>>(
            // Input
            n_samples,
            j11_raw,
            j12_raw,
            j13_raw,
            j21_raw,
            j22_raw,
            j23_raw,
            j31_raw,
            j32_raw,
            j33_raw,
            // Output
            cost_per_sample_raw);
        checkCudaErrors(cudaDeviceSynchronize());
        // Copy calculated cost data back to host
        auto cost_per_sample_host = thrust::host_vector<float>(cost_per_sample_dev);
        // Return cost data
        auto cost_per_sample_return = std::vector<float>(cost_per_sample_host.size());
        thrust::copy(
            cost_per_sample_host.begin(),
            cost_per_sample_host.end(),
            cost_per_sample_return.begin());
        return cost_per_sample_return;
      }
      /// Calculate the gradient of the cost per jacobian element
      std::vector<std::vector<float> > calculate_grad_per_jacobian_element_(
          const std::vector<std::vector<float> >& jacobian_elements) const
      {
        auto n_samples = jacobian_elements[0].size();
        // Create device vectors using thrust::
        auto j11_dev = thrust::device_vector<float>(jacobian_elements[0]);
        auto j12_dev = thrust::device_vector<float>(jacobian_elements[1]);
        auto j13_dev = thrust::device_vector<float>(jacobian_elements[2]);
        auto j21_dev = thrust::device_vector<float>(jacobian_elements[3]);
        auto j22_dev = thrust::device_vector<float>(jacobian_elements[4]);
        auto j23_dev = thrust::device_vector<float>(jacobian_elements[5]);
        auto j31_dev = thrust::device_vector<float>(jacobian_elements[6]);
        auto j32_dev = thrust::device_vector<float>(jacobian_elements[7]);
        auto j33_dev = thrust::device_vector<float>(jacobian_elements[8]);
        auto d_cost_d_j11_dev = thrust::device_vector<float>(n_samples);
        auto d_cost_d_j12_dev = thrust::device_vector<float>(n_samples);
        auto d_cost_d_j13_dev = thrust::device_vector<float>(n_samples);
        auto d_cost_d_j21_dev = thrust::device_vector<float>(n_samples);
        auto d_cost_d_j22_dev = thrust::device_vector<float>(n_samples);
        auto d_cost_d_j23_dev = thrust::device_vector<float>(n_samples);
        auto d_cost_d_j31_dev = thrust::device_vector<float>(n_samples);
        auto d_cost_d_j32_dev = thrust::device_vector<float>(n_samples);
        auto d_cost_d_j33_dev = thrust::device_vector<float>(n_samples);
        // Get raw pointers to device data
        auto j11_raw = thrust::raw_pointer_cast(j11_dev.data());
        auto j12_raw = thrust::raw_pointer_cast(j12_dev.data());
        auto j13_raw = thrust::raw_pointer_cast(j13_dev.data());
        auto j21_raw = thrust::raw_pointer_cast(j21_dev.data());
        auto j22_raw = thrust::raw_pointer_cast(j22_dev.data());
        auto j23_raw = thrust::raw_pointer_cast(j23_dev.data());
        auto j31_raw = thrust::raw_pointer_cast(j31_dev.data());
        auto j32_raw = thrust::raw_pointer_cast(j32_dev.data());
        auto j33_raw = thrust::raw_pointer_cast(j33_dev.data());
        auto d_cost_d_j11_raw = thrust::raw_pointer_cast(d_cost_d_j11_dev.data());
        auto d_cost_d_j12_raw = thrust::raw_pointer_cast(d_cost_d_j12_dev.data());
        auto d_cost_d_j13_raw = thrust::raw_pointer_cast(d_cost_d_j13_dev.data());
        auto d_cost_d_j21_raw = thrust::raw_pointer_cast(d_cost_d_j21_dev.data());
        auto d_cost_d_j22_raw = thrust::raw_pointer_cast(d_cost_d_j22_dev.data());
        auto d_cost_d_j23_raw = thrust::raw_pointer_cast(d_cost_d_j23_dev.data());
        auto d_cost_d_j31_raw = thrust::raw_pointer_cast(d_cost_d_j31_dev.data());
        auto d_cost_d_j32_raw = thrust::raw_pointer_cast(d_cost_d_j32_dev.data());
        auto d_cost_d_j33_raw = thrust::raw_pointer_cast(d_cost_d_j33_dev.data());
        // Calculate kernel parameters
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_log_jacobian_singular_values_grad_jxx_3d,
            0,
            0);
        unsigned int grid_size = (n_samples + block_size - 1)/block_size;
        // Call CUDA Kernel
        MMORF::kernel_log_jacobian_singular_values_grad_jxx_3d<<<grid_size, block_size>>>(
            // Input
            n_samples,
            j11_raw,
            j12_raw,
            j13_raw,
            j21_raw,
            j22_raw,
            j23_raw,
            j31_raw,
            j32_raw,
            j33_raw,
            // Output
            d_cost_d_j11_raw,
            d_cost_d_j12_raw,
            d_cost_d_j13_raw,
            d_cost_d_j21_raw,
            d_cost_d_j22_raw,
            d_cost_d_j23_raw,
            d_cost_d_j31_raw,
            d_cost_d_j32_raw,
            d_cost_d_j33_raw);
        checkCudaErrors(cudaDeviceSynchronize());
        // Copy calculated grad data back to host
        auto d_cost_d_j11_host = thrust::host_vector<float>(d_cost_d_j11_dev);
        auto d_cost_d_j12_host = thrust::host_vector<float>(d_cost_d_j12_dev);
        auto d_cost_d_j13_host = thrust::host_vector<float>(d_cost_d_j13_dev);
        auto d_cost_d_j21_host = thrust::host_vector<float>(d_cost_d_j21_dev);
        auto d_cost_d_j22_host = thrust::host_vector<float>(d_cost_d_j22_dev);
        auto d_cost_d_j23_host = thrust::host_vector<float>(d_cost_d_j23_dev);
        auto d_cost_d_j31_host = thrust::host_vector<float>(d_cost_d_j31_dev);
        auto d_cost_d_j32_host = thrust::host_vector<float>(d_cost_d_j32_dev);
        auto d_cost_d_j33_host = thrust::host_vector<float>(d_cost_d_j33_dev);
        // Return cost data
        auto grad_per_jacobian_element = std::vector<std::vector<float> >(
            9,
            std::vector<float>(n_samples));
        thrust::copy(
          d_cost_d_j11_host.begin(),
          d_cost_d_j11_host.end(),
          grad_per_jacobian_element[0].begin());
        thrust::copy(
          d_cost_d_j12_host.begin(),
          d_cost_d_j12_host.end(),
          grad_per_jacobian_element[1].begin());
        thrust::copy(
          d_cost_d_j13_host.begin(),
          d_cost_d_j13_host.end(),
          grad_per_jacobian_element[2].begin());
        thrust::copy(
          d_cost_d_j21_host.begin(),
          d_cost_d_j21_host.end(),
          grad_per_jacobian_element[3].begin());
        thrust::copy(
          d_cost_d_j22_host.begin(),
          d_cost_d_j22_host.end(),
          grad_per_jacobian_element[4].begin());
        thrust::copy(
          d_cost_d_j23_host.begin(),
          d_cost_d_j23_host.end(),
          grad_per_jacobian_element[5].begin());
        thrust::copy(
          d_cost_d_j31_host.begin(),
          d_cost_d_j31_host.end(),
          grad_per_jacobian_element[6].begin());
        thrust::copy(
          d_cost_d_j32_host.begin(),
          d_cost_d_j32_host.end(),
          grad_per_jacobian_element[7].begin());
        thrust::copy(
          d_cost_d_j33_host.begin(),
          d_cost_d_j33_host.end(),
          grad_per_jacobian_element[8].begin());
        return grad_per_jacobian_element;
      }
      /// Calculate the gradient of the cost per spline coefficient
      arma::fvec calculate_grad_per_spline_(
          const std::vector<std::vector<float> >& grad_per_jacobian_element) const
      {
        // Calculate jte_sz
        auto warp_sz = warp_field_->get_parameter_size();
        auto jte_sz = warp_sz.first * warp_sz.second;
        auto sample_dimensions = warp_field_->get_robust_sample_dimensions(
            sampling_frequency_);
        // Make a device vector for storing the result
        auto jte_dev = thrust::device_vector<float>(jte_sz,0.0f);
        // Loop over all positions in the local Jacobian
        for (auto row = 0; row < warp_sz.first; ++row){
          for (auto col = 0; col < warp_sz.first; ++col){
            auto element_id = (row * warp_sz.first) + col;
            // Generate the correct differentiated splines
            auto x_deriv_order = col == 0 ? 1 : 0;
            auto y_deriv_order = col == 1 ? 1 : 0;
            auto z_deriv_order = col == 2 ? 1 : 0;
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
            // Get correct index into jte_dev (x, then y, then z)
            auto jte_dev_ptr = thrust::raw_pointer_cast(
                &jte_dev[row * warp_sz.second]);
            auto temp_jte_dev = thrust::device_vector<float>(warp_sz.second);
            calculate_sub_grad_(
                // Input
                grad_per_jacobian_element[element_id],
                sample_dimensions,
                spline_x_vals,
                spline_y_vals,
                spline_z_vals,
                warp_field_->get_dimensions(),
                std::vector<int>(3, static_cast<int>(spline_1D_.KnotSpacing())),
                // Output
                temp_jte_dev);
            // Add subgrad to appropriate position
            auto jte_dev_begin_it = jte_dev.begin() + row * warp_sz.second;
            thrust::transform(
                temp_jte_dev.begin(),
                temp_jte_dev.end(),
                jte_dev_begin_it,
                jte_dev_begin_it,
                thrust::plus<float>());
          }
        }
        // Copy data to host
        auto jte_host = thrust::host_vector<float>(jte_dev);
        // Return result
        auto jte_return = arma::fvec(jte_host.data(), jte_host.size());
        return jte_return;
      }
      MMORF::SparseDiagonalMatrixTiled calculate_hess_per_spline_(
          const std::vector<float>&               cost_per_sample,
          const std::vector<std::vector<float> >& grad_per_jacobian_element) const
      {
        // Determine sizes of splines etc.
        auto spline_sz = spline_1D_.KernelSize();
        auto field_sz = cost_per_sample.size();
        // Create sparse tiled matrix to store sub-Hessian
        auto sparse_jtj = create_empty_jtj_(warp_field_->get_dimensions());
        // Get size of warp field
        auto warp_sz = warp_field_->get_parameter_size();
        // Get dimensions of samples
        auto sample_dimensions = warp_field_->get_robust_sample_dimensions(
            sampling_frequency_);
        // Calculate the possible combinations of both gradient images and splines. There
        // should be 9 combinations for each of the 9 "sub-Hessians" leading to 81 in total.
        // There is however symmetry involved here. 3 of each of the 9s are redundant (or
        // are the transpose of each other) and therefore there should only be 36 unique
        // sub-hessians to calculate (which is still admittedly a lot).
        //
        //auto field_pre_mult = std::vector<std::vector<float> >(
        //    warp_sz.first*warp_sz.first*warp_sz.first*warp_sz.first,
        //    std::vector<float>(cost_per_sample.size(), 0));
        // Loop over all warp directions
        for (auto row_hess = 0; row_hess < warp_sz.first; ++row_hess){
          // Loop over all warp directions again
          for (auto col_hess = 0; col_hess < warp_sz.first; ++col_hess){
            // Check if this is a unique sub-jtj - we will keep the main diagonal and below
            //if (col_hess > row_hess) continue;
            auto field_pre_mult = std::vector<std::vector<float> >(9,
                std::vector<float>(field_sz, 0));

            auto spline_x_row_vals = std::vector<float>(
                warp_sz.first*warp_sz.first*spline_sz, 0);
            auto spline_y_row_vals = std::vector<float>(
                warp_sz.first*warp_sz.first*spline_sz, 0);
            auto spline_z_row_vals = std::vector<float>(
                warp_sz.first*warp_sz.first*spline_sz, 0);

            auto spline_x_col_vals = std::vector<float>(
                warp_sz.first*warp_sz.first*spline_sz, 0);
            auto spline_y_col_vals = std::vector<float>(
                warp_sz.first*warp_sz.first*spline_sz, 0);
            auto spline_z_col_vals = std::vector<float>(
                warp_sz.first*warp_sz.first*spline_sz, 0);

            auto loop_count = 0;
            // Loop over the 3 Jacobian elements that contribute to the row gradient
            for (auto row_elem = 0; row_elem < warp_sz.first; ++row_elem){
              // Loop over the 3 Jacobian elements that contribute to the col gradient
              for (auto col_elem = 0; col_elem < warp_sz.first; ++col_elem){
                // Check if this is a unique combination of elements - we will keep the main
                // diagonal and below
                // if (col_elem > row_elem) continue;
                // Generate the correct differentiated splines
                // Note that the modulo (%) operator converts the "elem" value to the column
                // position of that element in the Jacobian matrix
                auto x_deriv_order_row = row_elem == 0 ? 1 : 0;
                auto y_deriv_order_row = row_elem == 1 ? 1 : 0;
                auto z_deriv_order_row = row_elem == 2 ? 1 : 0;

                auto x_deriv_order_col = col_elem == 0 ? 1 : 0;
                auto y_deriv_order_col = col_elem == 1 ? 1 : 0;
                auto z_deriv_order_col = col_elem == 2 ? 1 : 0;

                auto spline_x_row = BASISFIELD::Spline1D<float>(
                    spline_1D_.Order(),
                    spline_1D_.KnotSpacing(),
                    x_deriv_order_row);
                auto spline_y_row = BASISFIELD::Spline1D<float>(
                    spline_1D_.Order(),
                    spline_1D_.KnotSpacing(),
                    y_deriv_order_row);
                auto spline_z_row = BASISFIELD::Spline1D<float>(
                    spline_1D_.Order(),
                    spline_1D_.KnotSpacing(),
                    z_deriv_order_row);

                auto spline_x_col = BASISFIELD::Spline1D<float>(
                    spline_1D_.Order(),
                    spline_1D_.KnotSpacing(),
                    x_deriv_order_col);
                auto spline_y_col = BASISFIELD::Spline1D<float>(
                    spline_1D_.Order(),
                    spline_1D_.KnotSpacing(),
                    y_deriv_order_col);
                auto spline_z_col = BASISFIELD::Spline1D<float>(
                    spline_1D_.Order(),
                    spline_1D_.KnotSpacing(),
                    z_deriv_order_col);

                auto spline_x_row_vals_tmp = spline_as_vec_(
                    spline_x_row,
                    sample_resolution_[0],
                    x_deriv_order_row);
                auto spline_y_row_vals_tmp = spline_as_vec_(
                    spline_y_row,
                    sample_resolution_[1],
                    y_deriv_order_row);
                auto spline_z_row_vals_tmp = spline_as_vec_(
                    spline_z_row,
                    sample_resolution_[2],
                    z_deriv_order_row);

                auto spline_x_col_vals_tmp = spline_as_vec_(
                    spline_x_col,
                    sample_resolution_[0],
                    x_deriv_order_col);
                auto spline_y_col_vals_tmp = spline_as_vec_(
                    spline_y_col,
                    sample_resolution_[1],
                    y_deriv_order_col);
                auto spline_z_col_vals_tmp = spline_as_vec_(
                    spline_z_col,
                    sample_resolution_[2],
                    z_deriv_order_col);

                std::copy(
                    spline_x_row_vals_tmp.begin(),
                    spline_x_row_vals_tmp.end(),
                    std::next(spline_x_row_vals.begin(), loop_count*spline_sz));
                std::copy(
                    spline_y_row_vals_tmp.begin(),
                    spline_y_row_vals_tmp.end(),
                    std::next(spline_y_row_vals.begin(), loop_count*spline_sz));
                std::copy(
                    spline_z_row_vals_tmp.begin(),
                    spline_z_row_vals_tmp.end(),
                    std::next(spline_z_row_vals.begin(), loop_count*spline_sz));

                std::copy(
                    spline_x_col_vals_tmp.begin(),
                    spline_x_col_vals_tmp.end(),
                    std::next(spline_x_col_vals.begin(), loop_count*spline_sz));
                std::copy(
                    spline_y_col_vals_tmp.begin(),
                    spline_y_col_vals_tmp.end(),
                    std::next(spline_y_col_vals.begin(), loop_count*spline_sz));
                std::copy(
                    spline_z_col_vals_tmp.begin(),
                    spline_z_col_vals_tmp.end(),
                    std::next(spline_z_col_vals.begin(), loop_count*spline_sz));

                // Pre-multiply the gradient_per_jacobian_element_fields
                std::transform(
                    grad_per_jacobian_element[row_hess*warp_sz.first + row_elem].begin(),
                    grad_per_jacobian_element[row_hess*warp_sz.first + row_elem].end(),
                    grad_per_jacobian_element[col_hess*warp_sz.first + col_elem].begin(),
                    field_pre_mult[loop_count].begin(),
                    std::multiplies<float>()
                    );
                // Pre-divide by double the cost_per_sample
                std::transform(
                    field_pre_mult[loop_count].begin(),
                    field_pre_mult[loop_count].end(),
                    cost_per_sample.begin(),
                    field_pre_mult[loop_count].begin(),
                    [](float a, float b) -> float { return a/(2.0f*b); }
                    );

                // Increase loop counter
                ++loop_count;
              }
            }
            // Call kernel for this combination of elements
            calculate_sub_jtj_non_symmetrical_(
              // Input
              field_pre_mult,
              sample_dimensions,
              spline_x_row_vals,
              spline_y_row_vals,
              spline_z_row_vals,
              spline_x_col_vals,
              spline_y_col_vals,
              spline_z_col_vals,
              warp_field_->get_dimensions(),
              std::vector<int>(3, static_cast<int>(spline_1D_.KnotSpacing())),
              row_hess,
              col_hess,
              sparse_jtj);

            // Save above diagonal sub-jtj
            //if (row_hess > col_hess){
            //  sparse_jtj.copy_submatrix(row_hess, col_hess, col_hess, row_hess);
            //  sparse_jtj.transpose_submatrix(col_hess, row_hess);
            //}
          }
        }
        return sparse_jtj;
      }
      /// Calculate a sub-vector of Jte
      void calculate_sub_grad_(
          // Input
          const std::vector<float>&    grad_jxx_ima,
          const std::vector<int>&      ima_sz,
          const std::vector<float>&    spline_x,
          const std::vector<float>&    spline_y,
          const std::vector<float>&    spline_z,
          const std::vector<int>&      coef_sz,
          const std::vector<int>&      ksp,
          // Output
          thrust::device_vector<float>& sub_jte) const
      {
        // Make everything a device vector using thrust::
        auto grad_jxx_ima_dev = thrust::device_vector<float>(grad_jxx_ima);
        auto spline_x_dev = thrust::device_vector<float>(spline_x);
        auto spline_y_dev = thrust::device_vector<float>(spline_y);
        auto spline_z_dev = thrust::device_vector<float>(spline_z);
        // Create texture handle
        auto texture_handle = MMORF::TextureHandleLinear(grad_jxx_ima_dev,ima_sz);
        auto tex = texture_handle.get_texture();
        // Get all the raw pointers ready for the Kernel
        float *spline_x_raw = thrust::raw_pointer_cast(spline_x_dev.data());
        float *spline_y_raw = thrust::raw_pointer_cast(spline_y_dev.data());
        float *spline_z_raw = thrust::raw_pointer_cast(spline_z_dev.data());
        float *sub_jte_raw = thrust::raw_pointer_cast(sub_jte.data());
        // Calculate parameters for running kernel
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_make_jte,
            0,
            0);
        unsigned int grid_size =
          (coef_sz[0]*coef_sz[1]*coef_sz[2] + block_size - 1)/block_size;
        auto grid_1d = dim3(grid_size);
        unsigned int smem = (spline_x.size()+spline_y.size()+spline_z.size())*sizeof(float);
        // Call CUDA Kernel
        MMORF::kernel_make_jte<<<grid_1d,block_size,smem>>>(
            // Input
            ima_sz[0],
            ima_sz[1],
            ima_sz[2],
            tex,
            spline_x_raw,
            spline_y_raw,
            spline_z_raw,
            ksp[0],
            ksp[1],
            ksp[2],
            coef_sz[0],
            coef_sz[1],
            coef_sz[2],
            // Output
            sub_jte_raw);
        /// \todo Add error checking to cudaDeviceSynchronize
        checkCudaErrors(cudaDeviceSynchronize());
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
      // Calculate the offsets for a sparse diagonal matrix
      std::vector<int> calculate_offsets_(const std::vector<int>& coef_sz) const
      {
        auto offsets = std::vector<int>(343,0);
#pragma omp parallel for
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
        return offsets;
      }
      // Create an empty JtJ matrix. NB!!! In this case the matrix only contains the main
      // diagonal! And therefore we need a separate method to "calculate offsets" for what
      // the true Hessian would look like. This is annoying, but it's an unfortunate side-
      // effect of efficiency in the kernel.
      MMORF::SparseDiagonalMatrixTiled create_empty_jtj_(
          const std::vector<int>& coef_sz) const
      {
        auto offsets = std::vector<int>(1,0);
        unsigned int max_diagonal = coef_sz[0]*coef_sz[1]*coef_sz[2];
        MMORF::SparseDiagonalMatrixTiled r_matrix(max_diagonal, max_diagonal, 3, offsets);
        return r_matrix;
      }
      // Calclulate sub-matrix of jtj for symmetrical splines
      void calculate_sub_jtj_non_symmetrical_(
          const std::vector<std::vector<float> >& prod_ima,
          const std::vector<int>&                 ima_sz,
          const std::vector<float>&               spline_x_1,
          const std::vector<float>&               spline_y_1,
          const std::vector<float>&               spline_z_1,
          const std::vector<float>&               spline_x_2,
          const std::vector<float>&               spline_y_2,
          const std::vector<float>&               spline_z_2,
          const std::vector<int>&                 coef_sz,
          const std::vector<int>&                 ksp,
          const unsigned int                      sub_jtj_row,
          const unsigned int                      sub_jtj_col,
          MMORF::SparseDiagonalMatrixTiled&       sparse_jtj) const
      {
        // Calculate real Heassian offsets as sparse_jtj is now only a main diagonal matrix
        auto offsets = calculate_offsets_(coef_sz);
        // Make everything a device vector using thrust::
        auto sparse_jtj_offsets_dev = thrust::device_vector<int>(offsets);
        auto prod_ima_1_dev = thrust::device_vector<float>(prod_ima[0]);
        auto prod_ima_2_dev = thrust::device_vector<float>(prod_ima[1]);
        auto prod_ima_3_dev = thrust::device_vector<float>(prod_ima[2]);
        auto prod_ima_4_dev = thrust::device_vector<float>(prod_ima[3]);
        auto prod_ima_5_dev = thrust::device_vector<float>(prod_ima[4]);
        auto prod_ima_6_dev = thrust::device_vector<float>(prod_ima[5]);
        auto prod_ima_7_dev = thrust::device_vector<float>(prod_ima[6]);
        auto prod_ima_8_dev = thrust::device_vector<float>(prod_ima[7]);
        auto prod_ima_9_dev = thrust::device_vector<float>(prod_ima[8]);
        auto spline_x_1_dev = thrust::device_vector<float>(spline_x_1);
        auto spline_y_1_dev = thrust::device_vector<float>(spline_y_1);
        auto spline_z_1_dev = thrust::device_vector<float>(spline_z_1);
        auto spline_x_2_dev = thrust::device_vector<float>(spline_x_2);
        auto spline_y_2_dev = thrust::device_vector<float>(spline_y_2);
        auto spline_z_2_dev = thrust::device_vector<float>(spline_z_2);
        // Create texture handle. Note that as we are dealing with all combinations of
        // Jacobian elements which contribute to this sub-JTJ simultaneously we have to make
        // a vector of 9 different textures
        auto texture_handle_1 = MMORF::TextureHandleLinear(prod_ima_1_dev,ima_sz);
        auto texture_handle_2 = MMORF::TextureHandleLinear(prod_ima_2_dev,ima_sz);
        auto texture_handle_3 = MMORF::TextureHandleLinear(prod_ima_3_dev,ima_sz);
        auto texture_handle_4 = MMORF::TextureHandleLinear(prod_ima_4_dev,ima_sz);
        auto texture_handle_5 = MMORF::TextureHandleLinear(prod_ima_5_dev,ima_sz);
        auto texture_handle_6 = MMORF::TextureHandleLinear(prod_ima_6_dev,ima_sz);
        auto texture_handle_7 = MMORF::TextureHandleLinear(prod_ima_7_dev,ima_sz);
        auto texture_handle_8 = MMORF::TextureHandleLinear(prod_ima_8_dev,ima_sz);
        auto texture_handle_9 = MMORF::TextureHandleLinear(prod_ima_9_dev,ima_sz);
        auto tex_dev = thrust::device_vector<cudaTextureObject_t>(9);
        tex_dev[0] = texture_handle_1.get_texture();
        tex_dev[1] = texture_handle_2.get_texture();
        tex_dev[2] = texture_handle_3.get_texture();
        tex_dev[3] = texture_handle_4.get_texture();
        tex_dev[4] = texture_handle_5.get_texture();
        tex_dev[5] = texture_handle_6.get_texture();
        tex_dev[6] = texture_handle_7.get_texture();
        tex_dev[7] = texture_handle_8.get_texture();
        tex_dev[8] = texture_handle_9.get_texture();
        cudaTextureObject_t *tex_raw = thrust::raw_pointer_cast(tex_dev.data());
        // Get all the raw pointers ready for the Kernel
        float *spline_x_1_raw = thrust::raw_pointer_cast(spline_x_1_dev.data());
        float *spline_y_1_raw = thrust::raw_pointer_cast(spline_y_1_dev.data());
        float *spline_z_1_raw = thrust::raw_pointer_cast(spline_z_1_dev.data());
        float *spline_x_2_raw = thrust::raw_pointer_cast(spline_x_2_dev.data());
        float *spline_y_2_raw = thrust::raw_pointer_cast(spline_y_2_dev.data());
        float *spline_z_2_raw = thrust::raw_pointer_cast(spline_z_2_dev.data());
        int *sparse_jtj_offsets_raw = thrust::raw_pointer_cast(
            sparse_jtj_offsets_dev.data());

        float *sparse_jtj_raw_1 = sparse_jtj.get_raw_pointer(sub_jtj_row, sub_jtj_row);
        // Calculate parameters for running kernel
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_make_jtj_non_symmetrical_diag_hess_spred,
            0,
            0);
        unsigned int grid_size_diags = 343;
        unsigned int grid_size_chunks =
          (coef_sz[0]*coef_sz[1]*coef_sz[2] + block_size - 1)/block_size;
        auto grid_2d = dim3(grid_size_chunks, grid_size_diags);
        unsigned int smem =
          (spline_x_1.size() + spline_y_1.size() + spline_z_1.size()
          + spline_x_2.size() + spline_y_2.size() + spline_z_2.size())
          * sizeof(float);
        // Call CUDA Kernel
        if (sub_jtj_row == sub_jtj_col){
          MMORF::kernel_make_jtj_non_symmetrical_diag_hess_spred<<<grid_2d,block_size,smem>>>(
              // Input
              ima_sz.at(0),
              ima_sz.at(1),
              ima_sz.at(2),
              tex_raw,
              spline_x_1_raw,
              spline_y_1_raw,
              spline_z_1_raw,
              spline_x_2_raw,
              spline_y_2_raw,
              spline_z_2_raw,
              ksp.at(0),
              ksp.at(1),
              ksp.at(2),
              ksp.at(0),
              ksp.at(1),
              ksp.at(2),
              coef_sz.at(0),
              coef_sz.at(1),
              coef_sz.at(2),
              coef_sz.at(0),
              coef_sz.at(1),
              coef_sz.at(2),
              sparse_jtj_offsets_raw,
              // Output
              sparse_jtj_raw_1);
          checkCudaErrors(cudaDeviceSynchronize());
        }
        else{
          MMORF::kernel_make_jtj_non_symmetrical_diag_hess_spred<<<grid_2d,block_size,smem>>>(
              // Input
              ima_sz.at(0),
              ima_sz.at(1),
              ima_sz.at(2),
              tex_raw,
              spline_x_1_raw,
              spline_y_1_raw,
              spline_z_1_raw,
              spline_x_2_raw,
              spline_y_2_raw,
              spline_z_2_raw,
              ksp.at(0),
              ksp.at(1),
              ksp.at(2),
              ksp.at(0),
              ksp.at(1),
              ksp.at(2),
              coef_sz.at(0),
              coef_sz.at(1),
              coef_sz.at(2),
              coef_sz.at(0),
              coef_sz.at(1),
              coef_sz.at(2),
              sparse_jtj_offsets_raw,
              // Output
              sparse_jtj_raw_1);
          checkCudaErrors(cudaDeviceSynchronize());
        }
      }
      //////////////////////////////////////////////////
      // Private datamembers
      //////////////////////////////////////////////////
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field_;
      BASISFIELD::Spline1D<float>              spline_1D_;
      int                                      sampling_frequency_;
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
  CostFxnLogJacobianSingularValuesDiagHess::~CostFxnLogJacobianSingularValuesDiagHess() = default;
  /// Move ctor
  CostFxnLogJacobianSingularValuesDiagHess::CostFxnLogJacobianSingularValuesDiagHess(
      CostFxnLogJacobianSingularValuesDiagHess&& rhs) = default;
  /// Move assignment operator
  CostFxnLogJacobianSingularValuesDiagHess& CostFxnLogJacobianSingularValuesDiagHess::operator=(
      CostFxnLogJacobianSingularValuesDiagHess&& rhs) = default;
  /// Copy ctor
  CostFxnLogJacobianSingularValuesDiagHess::CostFxnLogJacobianSingularValuesDiagHess(
      const CostFxnLogJacobianSingularValuesDiagHess& rhs)
    : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  CostFxnLogJacobianSingularValuesDiagHess& CostFxnLogJacobianSingularValuesDiagHess::operator=(
      const CostFxnLogJacobianSingularValuesDiagHess& rhs)
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
  CostFxnLogJacobianSingularValuesDiagHess::CostFxnLogJacobianSingularValuesDiagHess(
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
      int sampling_frequency)
    : pimpl_(MMORF::make_unique<Impl>(warp_field, sampling_frequency))
  {}
  /// Get the current value of the parameters
  std::vector<std::vector<float> > CostFxnLogJacobianSingularValuesDiagHess::get_parameters() const
  {
    return pimpl_->get_parameters();
  }
  /// Set the current value of the parameters
  void CostFxnLogJacobianSingularValuesDiagHess::set_parameters(
      const std::vector<std::vector<float> >& parameters)
  {
    pimpl_->set_parameters(parameters);
  }
  /// Get cost under current parameterisation
  float CostFxnLogJacobianSingularValuesDiagHess::cost() const
  {
    return pimpl_->cost();
  }
  /// Get Jte under current parameterisation
  arma::fvec CostFxnLogJacobianSingularValuesDiagHess::grad() const
  {
    return pimpl_->grad();
  }
  /// Get JtJ under current parameterisation
  MMORF::SparseDiagonalMatrixTiled CostFxnLogJacobianSingularValuesDiagHess::hess() const
  {
    return pimpl_->hess();
  }
} // MMORF
