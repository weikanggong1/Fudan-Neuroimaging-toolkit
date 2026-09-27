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

#ifndef SPARSE_DIAGONAL_MATRIX_TILED_CUH
#define SPARSE_DIAGONAL_MATRIX_TILED_CUH

#include <cusp/csr_matrix.h>

#include <armadillo>

#include <vector>
#include <string>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class SparseDiagonalMatrixTiled
  {
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
    public:
      /// Default dtor
      ~SparseDiagonalMatrixTiled();
      /// Move ctor
      SparseDiagonalMatrixTiled(SparseDiagonalMatrixTiled&& rhs);
      /// Move assignment operator
      SparseDiagonalMatrixTiled& operator=(SparseDiagonalMatrixTiled&& rhs);
      /// Copy ctor
      SparseDiagonalMatrixTiled(const SparseDiagonalMatrixTiled& rhs);
      /// Copy assignment operator
      SparseDiagonalMatrixTiled& operator=(const SparseDiagonalMatrixTiled& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Basic constructor
      SparseDiagonalMatrixTiled (
          unsigned int n_rows,
          unsigned int n_cols,
          unsigned int n_tiles,
          const std::vector<int>& offsets);
      /// Get a pointer to the sub-matrix values
      float *get_raw_pointer(
          unsigned int tile_row,
          unsigned int tile_col);
      /// Return vector of diagonal offsets
      std::vector<int> get_offsets() const;
      /// Convert matrix to armadillo csc format
      arma::sp_fmat convert_to_csc() const;
      /// Convert matrix to CUSP's diagonal format
      cusp::csr_matrix<int,float,cusp::device_memory> convert_to_csr() const;
      /// Copy one submatrix's values to another
      void copy_submatrix(
          unsigned int from_row,
          unsigned int from_col,
          unsigned int to_row,
          unsigned int to_col);
      /// Scale the value of a diagonal
      void scale_main_diagonal(const float scaling);
      /// Add value to the main diagonal (for Levenberg optimisation)
      void add_to_main_diagonal(const float add_value);
////////////////////////////////////////////////////////////////////////////////
// Operator Overloads
////////////////////////////////////////////////////////////////////////////////
      /// Overload += operator
      SparseDiagonalMatrixTiled& operator+=(
        const SparseDiagonalMatrixTiled& r_mat);
      /// Overload *= operator for float
      SparseDiagonalMatrixTiled& operator*=(
        const float r_float);
      /// Overload /= operator for float
      SparseDiagonalMatrixTiled& operator/=(
        const float r_float);
      /// Overload the + operator
      friend SparseDiagonalMatrixTiled operator+(
          const SparseDiagonalMatrixTiled& l_mat,
          const SparseDiagonalMatrixTiled& r_mat);
      /// Overload the * operator for right hand side float
      friend SparseDiagonalMatrixTiled operator*(
          const SparseDiagonalMatrixTiled& l_mat,
          const float r_float);
      /// Overload the * operator for left hand side float
      friend SparseDiagonalMatrixTiled operator*(
          const float l_float,
          const SparseDiagonalMatrixTiled& r_mat);
      /// Overload the / operator for right hand side float
      friend SparseDiagonalMatrixTiled operator/(
          const SparseDiagonalMatrixTiled& l_mat,
          const float r_float);
      /// Overload the / operator for left hand side float
      friend SparseDiagonalMatrixTiled operator/(
          const float l_float,
          const SparseDiagonalMatrixTiled& r_mat);
    private:
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // SparseDiagonalMatrixTiled
} /// namespace MMORF
#endif // SPARSE_DIAGONAL_MATRIX_TILED_CUH
