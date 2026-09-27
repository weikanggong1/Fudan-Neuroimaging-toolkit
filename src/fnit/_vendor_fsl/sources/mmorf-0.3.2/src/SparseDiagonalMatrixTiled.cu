//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Stores a (lxm)x(lxn) sparse matrix as an lxl set of mxn sub-matrices with p
///        non-zero diagonals per sub-matrix.
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
#undef NDEBUG
#include "SparseDiagonalMatrixTiled.cuh"
#include "MmorfMemory.h"
#include "helper_cuda.h"

#include <cusp/csr_matrix.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>

#include <armadillo>

#include <vector>
#include <iterator>

#include <numeric>
#include <iostream>
#include <string>
#include <fstream>
#include <functional>
#include <cmath>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// __global__ functions
////////////////////////////////////////////////////////////////////////////////
  /// Count the number of non-zero elements per row based on a sparse diagonal format,
  /// assuming that all values in a diagonal are non-zero
  /// \details Each thread represents one row. NOTE: This assumes a square matrix!
  __global__ void kernel_count_nnz_per_row_tiled(
      // Input
      const unsigned int n_rows,
      const unsigned int n_cols,
      const unsigned int n_tiles,
      const unsigned int n_diags,
      const int* __restrict__ diag_offsets,
      // Output
      int* __restrict__ nnz_counts)
  {
    auto row_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (row_id < n_rows){
      auto nnz = 0;
      // Calculate nnz per row
      for (auto i = 0; i < n_diags; ++i){
        if (static_cast<int>(row_id) + diag_offsets[i] >= 0
            && static_cast<int>(row_id) + diag_offsets[i] < static_cast<int>(n_rows)){
          nnz += n_tiles;
        }
      }
      // Copy nnz to the right positions
      for (auto i = 0; i < n_tiles; ++i){
        nnz_counts[i*n_rows + row_id + 1] = nnz;
      } 
    }
  }
  /// Convert from diagonal format to csr, given that the row pointers have already been
  /// computed
  /// \details Each thread represents one row. NOTE: This assumes a square matrix!
  __global__ void kernel_dia_to_csr_tiled(
      // Input
      const unsigned int n_rows,
      const unsigned int n_cols,
      const unsigned int n_diags,
      const unsigned int n_tiles,
      const unsigned int tile_row,
      const unsigned int tile_col,
      const int* __restrict__ diag_offsets,
      const float* __restrict__ diag_vals,
      const int* __restrict__ row_ptrs,
      // Output
      int* __restrict__ col_inds,
      float* __restrict__ mat_vals)
  {
    auto row_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (row_id < n_rows){
      // This row is actually:
      //            row = (tile_row)*(n_rows) + row_id
      auto row_start = row_ptrs[tile_row*n_rows + row_id];
      auto sub_row_length =
        (row_ptrs[tile_row*n_rows + row_id + 1] - row_ptrs[tile_row*n_rows + row_id])
        / n_tiles;
      auto val_count = tile_col * (sub_row_length);
      // Loop through diagonals FORWARDS
      for (int diag_id = 0; diag_id < static_cast<int>(n_diags); ++diag_id){
        // Column index for this value in this diagonal
        auto col_id = static_cast<int>(row_id) + diag_offsets[diag_id];
        // If valid col
        if (col_id >= 0 && col_id < static_cast<int>(n_cols)){
          // Store row index
          col_inds[row_start + val_count] = tile_col*n_cols + col_id;
          // Store Value
          mat_vals[row_start + val_count] = diag_vals[diag_id*n_rows + row_id];
          ++val_count;
        }
      }
    }
  }
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class SparseDiagonalMatrixTiled::Impl
  {
    public:
      // Basic constructor
      Impl(
          unsigned int n_rows,
          unsigned int n_cols,
          unsigned int n_tiles,
          const std::vector<int>& offsets)
        :n_rows_(n_rows)
        ,n_cols_(n_cols)
        ,n_tiles_(n_tiles)
        ,offsets_(offsets)
        ,d_values_(
            n_tiles,
            std::vector<thrust::device_vector<float> >(
              n_tiles,
              thrust::device_vector<float>(
                offsets_.size()*n_rows_,
                0.0)))
      {}
      /// Overload the += operator
      /// \todo Replace asserts with exceptions
      /// \todo Allow for matrices with different non-zero diagonals
      Impl& operator+=(const Impl& r_impl)
      {
        // Check matrix dimensions match
        assert(
            this->n_rows_ == r_impl.n_rows_
            && this->n_cols_ == r_impl.n_cols_
            && this->n_tiles_ == r_impl.n_tiles_);
        // Check diagonals are identical
        assert(this->offsets_.size() == r_impl.offsets_.size());
        auto diagonals_match = true;
        for (auto i = 0; i < this->offsets_.size(); ++i){
          if (this->offsets_[i] != r_impl.offsets_[i]){
            diagonals_match = false;
          }
        }
        assert(diagonals_match);
        // If all tests pass, add values
        // Copy left hand side matrix
        for (auto row = 0; row < this->d_values_.size(); ++row){
          for (auto col = 0; col < this->d_values_[row].size(); ++col){
          thrust::transform(
              this->d_values_[row][col].begin(),
              this->d_values_[row][col].end(),
              r_impl.d_values_[row][col].begin(),
              this->d_values_[row][col].begin(),
              thrust::plus<float>());
          }
        }
        return *this;
      }
      /// Overload the *= operator
      Impl& operator*=(const float r_float)
      {
        for (auto row = 0; row < this->d_values_.size(); ++row){
          for (auto col = 0; col < this->d_values_[row].size(); ++col){
            thrust::transform(
                this->d_values_[row][col].begin(),
                this->d_values_[row][col].end(),
                thrust::make_constant_iterator(r_float),
                this->d_values_[row][col].begin(),
                thrust::multiplies<float>());
          }
        }
        return *this;
      }
      /// Overload the /= operator
      Impl& operator/=(const float r_float)
      {
        for (auto row = 0; row < this->d_values_.size(); ++row){
          for (auto col = 0; col < this->d_values_[row].size(); ++col){
            thrust::transform(
                this->d_values_[row][col].begin(),
                this->d_values_[row][col].end(),
                thrust::make_constant_iterator(r_float),
                this->d_values_[row][col].begin(),
                thrust::divides<float>());
          }
        }
        return *this;
      }
      // Get a pointer to the matrix values
      float *get_raw_pointer(
          unsigned int tile_row,
          unsigned int tile_col)
      {
        /// \todo Replace assert with exception
        assert(tile_row < n_tiles_ && tile_col < n_tiles_);
        return thrust::raw_pointer_cast(
            d_values_[tile_row][tile_col].data());
      } // get_raw_pointer
      // Get the offsets of the diagonals into the matrix
      std::vector<int> get_offsets() const
      {
        return offsets_;
      } // get_offsets
      /// Convert to the armadillo SpMat compressed sparse column format
      arma::sp_fmat convert_to_csc() const
      {
        // Row, column, value triplets.
        auto col_ptrs_a = arma::uvec(n_cols_*n_tiles_ + 1);
        auto row_inds_a = arma::uvec(d_values_[0][0].size()*n_tiles_*n_tiles_);
        auto mat_vals_a = arma::fvec(d_values_[0][0].size()*n_tiles_*n_tiles_);
        // Initialise row pointers
        col_ptrs_a[0] = 0;
        auto val_count = arma::uword(0);
        // Loop through tile cols
        for (auto tile_col = 0; tile_col < n_tiles_; ++tile_col){
          // Copy device data to host
          auto d_values_h = std::vector<thrust::host_vector<float> >(n_tiles_);
          for (auto i = 0; i < n_tiles_; ++i){
            d_values_h[i] = d_values_[i][tile_col];
          }
          // Loop through cols
          for (auto c_i = 0; c_i < n_cols_; ++c_i){
            // Cumulative sum
            col_ptrs_a[tile_col*n_cols_ + c_i + 1] = col_ptrs_a[tile_col*n_cols_ + c_i];
            // Loop through tile rows
            for (auto tile_row = 0; tile_row < n_tiles_; ++tile_row){
              // Loop through diagonals BACKWARDS
              //for (auto d_i = offsets_.size() - 1; d_i >= 0; --d_i){
              for (auto d_i = 0; d_i < offsets_.size(); ++d_i){
                // Row index for this value in this diagonal
                auto d_i_mirror = offsets_.size() - d_i - 1;
                auto r_i = static_cast<int>(c_i) - offsets_[d_i_mirror];
                // If valid row
                if (r_i >= 0 && r_i < n_rows_){
                  // Increment col ptr
                  col_ptrs_a[tile_col*n_cols_ + c_i + 1] += 1;
                  // Store row index
                  row_inds_a[val_count] =
                    tile_row*n_rows_ + static_cast<arma::uword>(r_i);
                  // Store Value
                  mat_vals_a[val_count] =
                    d_values_h[tile_row][d_i_mirror*n_rows_ + static_cast<arma::uword>(r_i)];
                  ++val_count;
                }
              }
            }
          }
        }
        row_inds_a.resize(val_count);
        mat_vals_a.resize(val_count);
        // Actually create the matrix
        auto csc_mat = arma::sp_fmat(
            row_inds_a,
            col_ptrs_a,
            mat_vals_a,
            n_rows_*n_tiles_,
            n_cols_*n_tiles_);
        return csc_mat;
      } // convert_to_csc
      /// Convert matrix to CUSP's diagonal format
      cusp::csr_matrix<int,float,cusp::device_memory> convert_to_csr() const
      {
        auto row_ptrs_d = create_row_pointers_();
        auto col_inds_d = thrust::device_vector<int>(row_ptrs_d.back());
        auto mat_vals_d = thrust::device_vector<float>(row_ptrs_d.back());
        dia_to_csr_(row_ptrs_d, col_inds_d, mat_vals_d);
        // Create cusp csr_matrix
        auto csr_mat = cusp::csr_matrix<int,float,cusp::device_memory>(
            n_rows_*n_tiles_,
            n_cols_*n_tiles_,
            mat_vals_d.size());
        csr_mat.row_offsets = row_ptrs_d;
        csr_mat.column_indices = col_inds_d;
        csr_mat.values = mat_vals_d;
        return csr_mat;
      }
      /// Copy one submatrix's values to another
      void copy_submatrix(
          unsigned int from_row,
          unsigned int from_col,
          unsigned int to_row,
          unsigned int to_col)
      {
        /// \todo Replace with exception
        assert (
            from_row < n_tiles_ &&
            from_col < n_tiles_ &&
            to_row < n_tiles_ &&
            to_col < n_tiles_);
        d_values_[to_row][to_col] = d_values_[from_row][from_col];
      } // copy_submatrix
      /// Scale the value of a diagonal
      void scale_main_diagonal(const float scaling)
      {
        // Find which entry corrseponds to the main diagonal i.e. offset == 0
        auto offset_it = std::find(offsets_.begin(), offsets_.end(), 0);
        // If the main diagonal exists
        if (offset_it != offsets_.end()){
          // Convert from iterator to index
          auto offset_ind = std::distance(offsets_.begin(), offset_it);
          // Scale diagonal for each tiled matrix on the main diagonal
          // (i.e. row_tile == col_tile)
          for (auto tile = 0; tile < n_tiles_; ++tile){
            auto diag_begin = d_values_[tile][tile].begin();
            thrust::advance(diag_begin, offset_ind * n_rows_);
            auto diag_end = diag_begin;
            thrust::advance(diag_end, n_rows_);
            thrust::transform(
                diag_begin,
                diag_end,
                thrust::make_constant_iterator(scaling),
                diag_begin,
                thrust::multiplies<float>());
          }
        }
      }
      void add_to_main_diagonal(const float add_value)
      {
        // Find which entry corrseponds to the main diagonal i.e. offset == 0
        auto offset_it = std::find(offsets_.begin(), offsets_.end(), 0);
        // If the main diagonal exists
        if (offset_it != offsets_.end()){
          // Convert from iterator to index
          auto offset_ind = std::distance(offsets_.begin(), offset_it);
          // Add to main diagonal for each tiled matrix on the main diagonal
          // (i.e. row_tile == col_tile)
          for (auto tile = 0; tile < n_tiles_; ++tile){
            auto diag_begin = d_values_[tile][tile].begin();
            thrust::advance(diag_begin, offset_ind * n_rows_);
            auto diag_end = diag_begin;
            thrust::advance(diag_end, n_rows_);
            thrust::transform(
                diag_begin,
                diag_end,
                thrust::make_constant_iterator(add_value),
                diag_begin,
                thrust::plus<float>());
          }
        }
      }
    private:
      /// Create a device vector with the column pointers for converting to csc format
      thrust::device_vector<int> create_row_pointers_() const
      {
        // Create device vector for row pointers
        auto row_ptrs_d = thrust::device_vector<int>(n_rows_*n_tiles_+1,0);
        // Copy offsets to device
        auto offsets_d = thrust::device_vector<int>(offsets_);
        // Get raw pointers to device data
        int *offsets_raw = thrust::raw_pointer_cast(offsets_d.data());
        int *row_ptrs_raw = thrust::raw_pointer_cast(row_ptrs_d.data());
        // Calculate kernel parameters
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_count_nnz_per_row_tiled,
            0,
            0);
        unsigned int grid_size = (n_rows_*n_tiles_ + block_size - 1)/block_size;
        // Call CUDA Kernel
        MMORF::kernel_count_nnz_per_row_tiled<<<grid_size, block_size>>>(
            // Input
            n_rows_,
            n_cols_,
            n_tiles_,
            offsets_.size(),
            offsets_raw,
            // Output
            row_ptrs_raw);
        checkCudaErrors(cudaDeviceSynchronize());
        // Calculate cumulative sum
        thrust::inclusive_scan(row_ptrs_d.begin(),row_ptrs_d.end(),row_ptrs_d.begin());
        return row_ptrs_d;
        // Copy to host and save to file
        //nnz_counts_h = thrust::host_vector<int>(nnz_counts_d);
        //nnz_counts_a = arma::Col<int>(thrust::raw_pointer_cast(nnz_counts_h.data()),nnz_counts_h.size());
        //nnz_counts_a.save("nnz_counts_cumsum.txt",arma::raw_ascii);
      }
      void dia_to_csr_(
          const thrust::device_vector<int>& row_ptrs,
          thrust::device_vector<int>& col_inds,
          thrust::device_vector<float>& mat_vals) const
      {
        // Copy offsets to device
        auto offsets_d = thrust::device_vector<int>(offsets_);
        // Get raw pointers to device data
        const int *offsets_raw = thrust::raw_pointer_cast(offsets_d.data());
        const int *row_ptrs_raw = thrust::raw_pointer_cast(row_ptrs.data());
        int *col_inds_raw = thrust::raw_pointer_cast(col_inds.data());
        float *mat_vals_raw = thrust::raw_pointer_cast(mat_vals.data());
        // Calculate kernel parameters
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_dia_to_csr_tiled,
            0,
            0);
        unsigned int grid_size = (n_rows_ + block_size - 1)/block_size;
        // Loop through tiled matrices
        for (auto tile_row = 0; tile_row < n_tiles_; ++tile_row){
          for (auto tile_col = 0; tile_col < n_tiles_; ++tile_col){
            // Get raw pointer for correct tile
            const float *diag_vals_raw =
              thrust::raw_pointer_cast(d_values_[tile_row][tile_col].data());
            // Call CUDA Kernel
            MMORF::kernel_dia_to_csr_tiled<<<grid_size, block_size>>>(
                // Input
                n_rows_,
                n_cols_,
                offsets_.size(),
                n_tiles_,
                tile_row,
                tile_col,
                offsets_raw,
                diag_vals_raw,
                row_ptrs_raw,
                // Output
                col_inds_raw,
                mat_vals_raw);
            checkCudaErrors(cudaDeviceSynchronize());
          }
        }
      }
      // Private datamembers
      unsigned int n_rows_;
      unsigned int n_cols_;
      unsigned int n_tiles_;
      /// The starting row for each diagonal
      std::vector<int> offsets_;
      /// The values of the matrix will be stored on the GPU (device)
      std::vector<std::vector<thrust::device_vector<float> > > d_values_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  SparseDiagonalMatrixTiled::~SparseDiagonalMatrixTiled() = default;
  /// Move ctor
  SparseDiagonalMatrixTiled::SparseDiagonalMatrixTiled(SparseDiagonalMatrixTiled&& rhs) = default;
  /// Move assignment operator
  SparseDiagonalMatrixTiled& SparseDiagonalMatrixTiled::operator=(SparseDiagonalMatrixTiled&& rhs) = default;
  /// Copy ctor
  SparseDiagonalMatrixTiled::SparseDiagonalMatrixTiled(const SparseDiagonalMatrixTiled& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  SparseDiagonalMatrixTiled& SparseDiagonalMatrixTiled::operator=(const SparseDiagonalMatrixTiled& rhs)
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
  SparseDiagonalMatrixTiled::SparseDiagonalMatrixTiled(
      unsigned int n_rows,
      unsigned int n_cols,
      unsigned int n_tiles,
      const std::vector<int>& offsets)
    : pimpl_(MMORF::make_unique<Impl>(n_rows,n_cols,n_tiles,offsets))
  {}
  // Get a pointer to the matrix values
  float *SparseDiagonalMatrixTiled::get_raw_pointer(
      unsigned int tile_row,
      unsigned int tile_col)
  {
    return pimpl_->get_raw_pointer(tile_row, tile_col);
  } // get_raw_pointer
  std::vector<int> SparseDiagonalMatrixTiled::get_offsets() const
  {
    return pimpl_->get_offsets();
  } // get_offsets
  /// Convert to the armadillo SpMat compressed sparse column format
  arma::sp_fmat SparseDiagonalMatrixTiled::convert_to_csc() const
  {
    return pimpl_->convert_to_csc();
  }
  /// Convert matrix to CUSP's diagonal format
  cusp::csr_matrix<int,float,cusp::device_memory>
    SparseDiagonalMatrixTiled::convert_to_csr() const
  {
    return pimpl_->convert_to_csr();
  }
  /// Copy one submatrix's values to another
  void SparseDiagonalMatrixTiled::copy_submatrix(
      unsigned int from_row,
      unsigned int from_col,
      unsigned int to_row,
      unsigned int to_col)
  {
    pimpl_->copy_submatrix(from_row, from_col, to_row, to_col);
  } // copy_submatrix
  /// Scale the value of a diagonal
  void SparseDiagonalMatrixTiled::scale_main_diagonal(const float scaling)
  {
    pimpl_->scale_main_diagonal(scaling);
  }
  void SparseDiagonalMatrixTiled::add_to_main_diagonal(const float add_value)
  {
    pimpl_->add_to_main_diagonal(add_value);
  }
////////////////////////////////////////////////////////////////////////////////
// Operator overloads
////////////////////////////////////////////////////////////////////////////////
  /// Overload the += operator
  SparseDiagonalMatrixTiled& SparseDiagonalMatrixTiled::operator+=(
      const SparseDiagonalMatrixTiled& r_mat)
  {
    *pimpl_ += *r_mat.pimpl_;
    return *this;
  }
  /// Overload the *= operator
  SparseDiagonalMatrixTiled& SparseDiagonalMatrixTiled::operator*=(
      const float r_float)
  {
    *pimpl_ *= r_float;
    return *this;
  }
  /// Overload the /= operator
  SparseDiagonalMatrixTiled& SparseDiagonalMatrixTiled::operator/=(
      const float r_float)
  {
    *pimpl_ /= r_float;
    return *this;
  }
  /// Overload the + operator
  SparseDiagonalMatrixTiled operator+(
      const SparseDiagonalMatrixTiled& l_mat,
      const SparseDiagonalMatrixTiled& r_mat)
  {
    auto tmp_mat = SparseDiagonalMatrixTiled(l_mat);
    *tmp_mat.pimpl_ += *r_mat.pimpl_;
    return tmp_mat;
  }
  /// Overload the * operator for right hand side float
  SparseDiagonalMatrixTiled operator*(
      const SparseDiagonalMatrixTiled& l_mat,
      const float r_float)
  {
    auto tmp_mat = SparseDiagonalMatrixTiled(l_mat);
    *tmp_mat.pimpl_ *= r_float;
    return tmp_mat;
  }
  /// Overload the * operator for left hand side float
  SparseDiagonalMatrixTiled operator*(
      const float l_float,
      const SparseDiagonalMatrixTiled& r_mat)
  {
    auto tmp_mat = SparseDiagonalMatrixTiled(r_mat);
    *tmp_mat.pimpl_ *= l_float;
    return tmp_mat;
  }
  /// Overload the / operator for right hand side float
  SparseDiagonalMatrixTiled operator/(
      const SparseDiagonalMatrixTiled& l_mat,
      const float r_float)
  {
    auto tmp_mat = SparseDiagonalMatrixTiled(l_mat);
    *tmp_mat.pimpl_ /= r_float;
    return tmp_mat;
  }
  /// Overload the / operator for left hand side float
  SparseDiagonalMatrixTiled operator/(
      const float l_float,
      const SparseDiagonalMatrixTiled& r_mat)
  {
    auto tmp_mat = SparseDiagonalMatrixTiled(r_mat);
    *tmp_mat.pimpl_ /= l_float;
    return tmp_mat;
  }
} // namespace MMORF
