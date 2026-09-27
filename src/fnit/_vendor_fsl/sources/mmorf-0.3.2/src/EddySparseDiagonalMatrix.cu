// Author:  Frederik Lange
// Date:    30/08/2017
//
// Sparse matrix class designed specifically for handling Hessian matrices generated during
// non-linear volume registration.
//
// NB! The matrix represented here is square, symmetric about the main diagonal, with each
// diagonal linearised from the bottom left of the matrix up. Also note that each diagonal
// is padded with zeros in order to have all diagonals equal in length with the main diagonal.

#include "miscmaths/bfmatrix.h"

#include <vector>
#include <numeric>
#include <iostream>
#include <string>
#include <fstream>
#include <functional>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/count.h>

#include <boost/shared_ptr.hpp>

#include "EddySparseDiagonalMatrix.cuh"
#include "EddyHessianKernels.cuh"

namespace MMORF
{
    // PREDICATE //
    template <typename T>
    struct is_non_zero
    {
        __host__ __device__
            bool operator ()(const T& x)
            {
                return !(x == (T)0);
            }
    };
    // Basic constructor
    SparseDiagonalMatrix::SparseDiagonalMatrix(unsigned int n_rows, unsigned int n_cols,
                                               const std::vector<int>& offsets)
      :n_rows_(n_rows)
      ,n_cols_(n_cols)
      ,offsets_(offsets)
      ,d_values_(offsets_.size()*n_rows_,0.0)
    {
    }
    // Copy constructor
    SparseDiagonalMatrix::SparseDiagonalMatrix(const SparseDiagonalMatrix &obj)
      :n_rows_(obj.n_rows_)
      ,n_cols_(obj.n_cols_)
      ,offsets_(obj.offsets_)
      ,d_values_(obj.d_values_)
    {
    }
    // Get a pointer to the matrix values
    float *SparseDiagonalMatrix::get_raw_pointer()
    {
        return thrust::raw_pointer_cast(d_values_.data());
    }

    std::vector<int> SparseDiagonalMatrix::get_offsets() const
    {
      return offsets_;
    }
    // Save Matrix to file

    void SparseDiagonalMatrix::save_matrix_to_text_file(std::string file_name)
    {
        // Attempt to create output stream
        std::ofstream matrix_file;
        matrix_file.open(file_name);
        // If successful
        if (matrix_file.is_open())
        {
            // Copy vector values from device to host
            thrust::host_vector<float> h_values = d_values_;
            // Copy values to file object
            std::ostream_iterator<float> file_iter(matrix_file,"\n");
            thrust::copy(h_values.begin(),h_values.end(),file_iter);
            matrix_file.close();
        }

        else
        {
            std::cout << "Failed to open file" << std::endl;
        }

        // Save diagonal numbers
        std::string diagonal_file_name = file_name + "_diagonals";
        std::ofstream diagonal_file;
        diagonal_file.open(diagonal_file_name);
        // If successful
        if (diagonal_file.is_open())
        {
            // Copy values to file object
            std::ostream_iterator<unsigned int> file_iter(diagonal_file,"\n");
            std::copy(offsets_.begin(),offsets_.end(),file_iter);
            diagonal_file.close();
        }
        else
        {
            std::cout << "Failed to open file" << std::endl;
        }
    }
/*
    // Carry out multiplication for each diagonal (including above main diagonal)
    thrust::device_vector<float> SparseDiagonalMatrix::mult_by_vec(thrust::device_vector<float> &vec)
    {
        // Typedefs used in function
        // typedef thrust::tuple<float,float,float> Float3;
        // Multiply each diagonal (above & below main diagonal) by the rhs vector
        thrust::device_vector<float> result(dimension_,0.0);
        // Bottom triangular matrix including main diagonal
        for (unsigned int this_diag = 0; this_diag < 172; this_diag++)
        {
            Float3Iterator start = thrust::make_zip_iterator(
                                   thrust::make_tuple(d_values_.begin() +
                                                            this_diag*dimension_,
                                                      vec.begin(), 
                                                      result.begin() +
                                                            offsets_[this_diag]));
            Float3Iterator end = thrust::make_zip_iterator(
                                 thrust::make_tuple(d_values_.begin() +
                                                            (this_diag+1)*dimension_ -
                                                            offsets_[this_diag],
                                                    vec.end() - offsets_[this_diag],
                                                    result.end()));
            thrust::transform(start,end,result.begin() + offsets_[this_diag]
                                ,MultiplyAdd());
        }
        // Top triangular matrix excluding main diagonal
        for (unsigned int this_diag = 0; this_diag < 171; this_diag++)
        {
            Float3Iterator start = thrust::make_zip_iterator(
                                   thrust::make_tuple(d_values_.begin() +
                                                            this_diag*dimension_,
                                                      vec.begin() + offsets_[this_diag],
                                                      result.begin()));
            Float3Iterator end = thrust::make_zip_iterator(
                                 thrust::make_tuple(d_values_.begin() +
                                                            (this_diag+1)*dimension_ -
                                                            offsets_[this_diag],
                                                    vec.end(),
                                                    result.end() - offsets_[this_diag]));
            thrust::transform(start,end,result.begin(),MultiplyAdd());
        }
        return result;
    }
*/

    /*
    // Convert to MISCMATHS::SparseBFMatrix csc format
    boost::shared_ptr<MISCMATHS::BFMatrix> SparseDiagonalMatrix::convert_to_sparse_bf_matrix(
        MISCMATHS::BFMatrixPrecisionType prec)
    {
      // Copy device data to host
      thrust::host_vector<float> d_values = d_values_;
      // Row, column, value triplets.
      auto row_ptrs = std::vector<unsigned int>(n_rows_ + 1);
      auto col_inds = std::vector<unsigned int>(d_values.size());
      auto mat_vals = std::vector<double>(d_values.size());
      // Initialise row pointers
      row_ptrs[0] = 0;
      // Loop through rows
      int val_count = 0;
      for (int r_i = 0; r_i < n_rows_; ++r_i){
        // Cumulative sum
        row_ptrs[r_i+1] = row_ptrs[r_i];
        // Loop through diagonals
        for (int d_i = 0, d_i_end = offsets_.size(); d_i < d_i_end; ++d_i){
          // Column index for this value in this diagonal
          auto c_i = offsets_[d_i]+ r_i;
          // If valid column
          if (c_i >= 0 && c_i < n_cols_){
            // Increment row ptr
            row_ptrs[r_i+1] += 1;
            // Store column index
            col_inds[val_count] = c_i;
            // Store Value
            mat_vals[val_count] = d_values[d_i*n_rows_ + r_i];
            ++val_count;
          }
        }
      }
      // Convert to std::vectors as SparseBFMatrix requires unsigned int* for row/column
      // Actually create the matrix
      boost::shared_ptr<MISCMATHS::BFMatrix> csr_sbf;
      if (prec == MISCMATHS::BFMatrixFloatPrecision){
        csr_sbf = boost::shared_ptr<MISCMATHS::BFMatrix>(
            new MISCMATHS::SparseBFMatrix<float>(
                n_rows_,
                n_cols_,
                col_inds.data(),
                row_ptrs.data(),
                mat_vals.data()));
      }
      else{
        csr_sbf = boost::shared_ptr<MISCMATHS::BFMatrix>(
            new MISCMATHS::SparseBFMatrix<double>(
              n_rows_,
              n_cols_,
              col_inds.data(),
              row_ptrs.data(),
              mat_vals.data()));
      }
      // auto csc_sbf = csr_sbf->Transpose();
      // return csc_sbf;
      return csr_sbf;
    }
    */

    // Convert to MISCMATHS::SparseBFMatrix csc format
    boost::shared_ptr<MISCMATHS::BFMatrix> SparseDiagonalMatrix::convert_to_sparse_bf_matrix(
        MISCMATHS::BFMatrixPrecisionType prec)
    {
      // Copy device data to host
      thrust::host_vector<float> d_values = d_values_;
      // Row, column, value triplets.
      auto col_ptrs = std::vector<unsigned int>(n_cols_ + 1);
      auto row_inds = std::vector<unsigned int>(d_values.size());
      auto mat_vals = std::vector<double>(d_values.size());
      // Initialise row pointers
      col_ptrs[0] = 0;
      // Loop through cols
      int val_count = 0;
      for (int c_i = 0; c_i < n_cols_; ++c_i){
        // Cumulative sum
        col_ptrs[c_i+1] = col_ptrs[c_i];
        // Loop through diagonals BACKWARDS
        for (int d_i = offsets_.size() - 1, d_i_end = 0; d_i >= d_i_end; --d_i){
          // Row index for this value in this diagonal
          auto r_i = c_i - offsets_[d_i];
          // If valid row
          if (r_i >= 0 && r_i < n_rows_){
            // Increment col ptr
            col_ptrs[c_i+1] += 1;
            // Store row index
            row_inds[val_count] = r_i;
            // Store Value
            mat_vals[val_count] = d_values[d_i*n_rows_ + r_i];
            ++val_count;
          }
        }
      }
      // Actually create the matrix
      boost::shared_ptr<MISCMATHS::BFMatrix> csc_sbf;
      if (prec == MISCMATHS::BFMatrixFloatPrecision){
        csc_sbf = boost::shared_ptr<MISCMATHS::BFMatrix>(
            new MISCMATHS::SparseBFMatrix<float>(
                n_rows_,
                n_cols_,
                row_inds.data(),
                col_ptrs.data(),
                mat_vals.data()));
      }
      else{
        csc_sbf = boost::shared_ptr<MISCMATHS::BFMatrix>(
            new MISCMATHS::SparseBFMatrix<double>(
              n_rows_,
              n_cols_,
              row_inds.data(),
              col_ptrs.data(),
              mat_vals.data()));
      }
      return csc_sbf;
    }
}
