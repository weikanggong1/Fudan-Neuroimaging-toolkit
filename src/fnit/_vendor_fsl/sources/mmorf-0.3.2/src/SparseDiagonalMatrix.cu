//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Stores a mxn sparse matrix with p non-zero diagonals
/// \details This matrix makes no assumptions regarding symmetry or order of diagonals, but
///          these are important factors to consider when using the raw underlying values in
///          cuda kernels.
///
///          Example of how storage works for 3x3 matrix:
///
///               [1 2 3]    0 0[1 2 3]       [0 0 1 2 3]
///           A = [4 5 6] =>   0[4 5 6]0   => [0 4 5 6 0] + [-2 -1 0 1 2]
///               [7 8 9]       [7 8 9]0 0    [7 8 9 0 0]
///
///          I.e. the underlying data storage contains an mxp matrix where each column
///          contains the values of one of the diagonals, and the p-length vector contains
///          the offset of the diagonal into the original matrix, with 0 representing the main
///          diagonal, -ve for lower diagonals, and +ve for upper diagonals.
/// \author Frederik Lange
/// \date March 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#include "SparseDiagonalMatrix.cuh"
#include "MmorfMemory.h"
#include "helper_cuda.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include <armadillo>

#include <vector>
#include <numeric>
#include <iostream>
#include <string>
#include <fstream>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// __global__ functions
////////////////////////////////////////////////////////////////////////////////
  /// Count the number of non-zero elements per column based on a sparse diagonal format,
  /// assuming that all values in a diagonal are non-zero
  /// \details Each thread represents one column. NOTE: This assumes a square matrix!
  /*
  __global__ void kernel_count_nnz_per_column(
      // Input
      const unsigned int n_rows,
      const unsigned int n_cols,
      const unsigned int n_diags,
      const int* __restrict__ diag_offsets,
      // Output
      int* __restrict__ nnz_counts)
  {
    auto col_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (col_id < n_cols){
      auto nnz = 0;
      int col_diff = (static_cast<int>(col_id) - static_cast<int>(n_cols));
      for (auto i = 0; i < n_diags; ++i){
        if (diag_offsets[i] <= static_cast<int>(col_id) &&
            diag_offsets[i] > col_diff){
          ++nnz;
        }
      }
      nnz_counts[col_id+1] = nnz;
    }
  }
  */
  /// Convert from diagonal format to csc, given that the cloumn pointers have already been
  /// computed
  /// \details Each thread represents one column. NOTE: This assumes a square matrix!
  /*
  __global__ void kernel_dia_to_csc(
      // Input
      const unsigned int n_rows,
      const unsigned int n_cols,
      const unsigned int n_diags,
      const int* __restrict__ diag_offsets,
      const float* __restrict__ diag_vals,
      const int* __restrict__ col_ptrs,
      // Output
      int* __restrict__ row_inds,
      float* __restrict__ mat_vals)
  {
    auto col_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (col_id < n_cols){
      auto col_start = col_ptrs[col_id];
      auto val_count = 0;
      // Loop through diagonals BACKWARDS
      for (int diag_id = static_cast<int>(n_diags) - 1; diag_id >= 0; --diag_id){
        // Row index for this value in this diagonal
        auto row_id = static_cast<int>(col_id) - diag_offsets[diag_id];
        // If valid row
        if (row_id >= 0 && row_id < static_cast<int>(n_rows)){
          // Store row index
          row_inds[col_start + val_count] = row_id;
          // Store Value
          mat_vals[col_start + val_count] = diag_vals[diag_id*n_rows + row_id];
          ++val_count;
        }
      }
    }
  }
  */
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class SparseDiagonalMatrix::Impl
  {
    public:
      // Basic constructor
      Impl(
          unsigned int n_rows,
          unsigned int n_cols,
          const std::vector<int>& offsets)
        :n_rows_(n_rows)
        ,n_cols_(n_cols)
        ,offsets_(offsets)
        ,d_values_(offsets_.size()*n_rows_,0.0)
      {}
      // Overload the + operator
      friend Impl operator+(
          const Impl& l_impl,
          const Impl& r_impl);
      // Get a pointer to the matrix values
      float *get_raw_pointer()
      {
        return thrust::raw_pointer_cast(d_values_.data());
      } // get_raw_pointer
      // Get the offsets of the diagonals into the matrix
      std::vector<int> get_offsets() const
      {
        return offsets_;
      } // get_offsets
      /// Convert to the armadillo SpMat compressed sparse column format
      arma::sp_fmat convert_to_csc()
      {
        /*
        auto col_ptrs_d = create_column_pointers_();
        auto row_inds_d = thrust::device_vector<int>(col_ptrs_d.back());
        auto mat_vals_d = thrust::device_vector<float>(col_ptrs_d.back());
        dia_to_csc_(col_ptrs_d, row_inds_d, mat_vals_d);
        // Copy data to host
        auto col_ptrs_h = thrust::host_vector<arma::u64>(col_ptrs_d);
        auto row_inds_h = thrust::host_vector<arma::u64>(row_inds_d);
        auto mat_vals_h = thrust::host_vector<float>(mat_vals_d);
        // Row, column, value triplets.
        auto col_ptrs_a = arma::uvec(
            thrust::raw_pointer_cast(col_ptrs_h.data()),
            col_ptrs_h.size());
        auto row_inds_a = arma::uvec(
            thrust::raw_pointer_cast(row_inds_h.data()),
            row_inds_h.size());
        auto mat_vals_a = arma::fvec(
            thrust::raw_pointer_cast(mat_vals_h.data()),
            mat_vals_h.size());
        */
        // Copy device data to host
        thrust::host_vector<float> d_values = d_values_;
        // Row, column, value triplets.
        auto col_ptrs_a = arma::uvec(n_cols_ + 1);
        auto row_inds_a = arma::uvec(d_values.size());
        auto mat_vals_a = arma::fvec(d_values.size());
        // Initialise row pointers
        col_ptrs_a[0] = 0;
        // Loop through cols
        int val_count = 0;
        for (int c_i = 0; c_i < n_cols_; ++c_i){
          // Cumulative sum
          col_ptrs_a[c_i+1] = col_ptrs_a[c_i];
          // Loop through diagonals BACKWARDS
          for (int d_i = offsets_.size() - 1, d_i_end = 0; d_i >= d_i_end; --d_i){
            // Row index for this value in this diagonal
            auto r_i = c_i - offsets_[d_i];
            // If valid row
            if (r_i >= 0 && r_i < n_rows_){
              // Increment col ptr
              col_ptrs_a[c_i+1] += 1;
              // Store row index
              row_inds_a[val_count] = r_i;
              // Store Value
              mat_vals_a[val_count] = d_values[d_i*n_rows_ + r_i];
              ++val_count;
            }
          }
        }
        // Actually create the matrix
        auto csc_mat = arma::sp_fmat(
            row_inds_a,
            col_ptrs_a,
            mat_vals_a,
            n_rows_,
            n_cols_);
        return csc_mat;
      } // convert_to_sparse_bf_matrix
    private:
      /// Create a device vector with the column pointers for converting to csc format
      /*
      thrust::device_vector<int> create_column_pointers_() const
      {
        // Create device vector for column pointers
        auto col_ptrs_d = thrust::device_vector<int>(n_cols_+1,0);
        // Copy offsets to device
        auto offsets_d = thrust::device_vector<int>(offsets_);
        // Get raw pointers to device data
        int *offsets_raw = thrust::raw_pointer_cast(offsets_d.data());
        int *col_ptrs_raw = thrust::raw_pointer_cast(col_ptrs_d.data());
        // Calculate kernel parameters
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_count_nnz_per_column,
            0,
            0);
        unsigned int grid_size = (n_cols_ + block_size - 1)/block_size;
        // Call CUDA Kernel
        MMORF::kernel_count_nnz_per_column<<<grid_size, block_size>>>(
            // Input
            n_rows_,
            n_cols_,
            offsets_.size(),
            offsets_raw,
            // Output
            col_ptrs_raw);
        checkCudaErrors(cudaDeviceSynchronize());
        // Calculate cumulative sum
        thrust::inclusive_scan(col_ptrs_d.begin(),col_ptrs_d.end(),col_ptrs_d.begin());
        return col_ptrs_d;
        // Copy to host and save to file
        //nnz_counts_h = thrust::host_vector<int>(nnz_counts_d);
        //nnz_counts_a = arma::Col<int>(thrust::raw_pointer_cast(nnz_counts_h.data()),nnz_counts_h.size());
        //nnz_counts_a.save("nnz_counts_cumsum.txt",arma::raw_ascii);
      }
      */
      /*
      void dia_to_csc_(
          const thrust::device_vector<int>& col_ptrs,
          thrust::device_vector<int>& row_inds,
          thrust::device_vector<float>& mat_vals)
      {
        // Copy offsets to device
        auto offsets_d = thrust::device_vector<int>(offsets_);
        // Get raw pointers to device data
        const int *offsets_raw = thrust::raw_pointer_cast(offsets_d.data());
        const float *diag_vals_raw = thrust::raw_pointer_cast(d_values_.data());
        const int *col_ptrs_raw = thrust::raw_pointer_cast(col_ptrs.data());
        int *row_inds_raw = thrust::raw_pointer_cast(row_inds.data());
        float *mat_vals_raw = thrust::raw_pointer_cast(mat_vals.data());
        // Calculate kernel parameters
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_dia_to_csc,
            0,
            0);
        unsigned int grid_size = (n_cols_ + block_size - 1)/block_size;
        // Call CUDA Kernel
        MMORF::kernel_dia_to_csc<<<grid_size, block_size>>>(
            // Input
            n_rows_,
            n_cols_,
            offsets_.size(),
            offsets_raw,
            diag_vals_raw,
            col_ptrs_raw,
            // Output
            row_inds_raw,
            mat_vals_raw);
        checkCudaErrors(cudaDeviceSynchronize());
      }
      */
      // Private datamembers
      unsigned int n_rows_;
      unsigned int n_cols_;
      /// The starting row for each diagonal
      std::vector<int> offsets_;
      /// The values of the matrix will be stored on the GPU (device)
      thrust::device_vector<float> d_values_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  SparseDiagonalMatrix::~SparseDiagonalMatrix() = default;
  /// Move ctor
  SparseDiagonalMatrix::SparseDiagonalMatrix(SparseDiagonalMatrix&& rhs) = default;
  /// Move assignment operator
  SparseDiagonalMatrix& SparseDiagonalMatrix::operator=(SparseDiagonalMatrix&& rhs) = default;
  /// Copy ctor
  SparseDiagonalMatrix::SparseDiagonalMatrix(const SparseDiagonalMatrix& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  SparseDiagonalMatrix& SparseDiagonalMatrix::operator=(const SparseDiagonalMatrix& rhs)
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
  // Basic constructor
  SparseDiagonalMatrix::SparseDiagonalMatrix(
      unsigned int n_rows,
      unsigned int n_cols,
      const std::vector<int>& offsets)
    : pimpl_(MMORF::make_unique<Impl>(n_rows,n_cols,offsets))
  {}
  // Get a pointer to the matrix values
  float *SparseDiagonalMatrix::get_raw_pointer()
  {
    return pimpl_->get_raw_pointer();
  } // get_raw_pointer
  std::vector<int> SparseDiagonalMatrix::get_offsets() const
  {
    return pimpl_->get_offsets();
  } // get_offsets
  /// Convert to the armadillo SpMat compressed sparse column format
  arma::sp_fmat SparseDiagonalMatrix::convert_to_csc()
  {
    return pimpl_->convert_to_csc();
  }
////////////////////////////////////////////////////////////////////////////////
// Overloaded operators
////////////////////////////////////////////////////////////////////////////////
  /// Overload the + operator for Main class
  SparseDiagonalMatrix operator+(
      const SparseDiagonalMatrix& l_mat,
      const SparseDiagonalMatrix& r_mat)
  {
    // Copy left hand side matrix
    auto temp_mat = l_mat;
    // Sum the Impls into the temporary object.
    *temp_mat.pimpl_ = *l_mat.pimpl_ + *r_mat.pimpl_;
    return temp_mat;
  }
  /// Overload the + operator for Impl class
  /// \todo Replace asserts with exceptions
  /// \todo Allow for matrices with different non-zero diagonals
  SparseDiagonalMatrix::Impl operator+(
      const SparseDiagonalMatrix::Impl& l_impl,
      const SparseDiagonalMatrix::Impl& r_impl)
  {
    // Check matrix dimensions match
    assert(
        l_impl.n_rows_ == r_impl.n_rows_
        && l_impl.n_cols_ == r_impl.n_cols_);
    // Check diagonals are identical
    assert(l_impl.offsets_.size() == r_impl.offsets_.size());
    auto diagonals_match = true;
    for (auto i = 0; i < l_impl.offsets_.size(); ++i){
      if (l_impl.offsets_[i] != r_impl.offsets_[i]){
        diagonals_match = false;
      }
    }
    assert(diagonals_match);
    // If all tests pass, add values
    // Copy left hand side Impl
    auto temp_impl = l_impl;
    thrust::transform(
        l_impl.d_values_.begin(),
        l_impl.d_values_.end(),
        r_impl.d_values_.begin(),
        temp_impl.d_values_.begin(),
        thrust::plus<float>());
    return temp_impl;
  }
} // namespace MMORF
