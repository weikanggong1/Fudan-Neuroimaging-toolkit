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

#ifndef SPARSE_DIAGONAL_MATRIX_CUH
#define SPARSE_DIAGONAL_MATRIX_CUH

#include "miscmaths/bfmatrix.h"

#include <armadillo>

#include <vector>
#include <string>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class SparseDiagonalMatrix
  {
////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////
    private:
      class Impl;
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
    public:
      /// Default dtor
      ~SparseDiagonalMatrix();
      /// Move ctor
      SparseDiagonalMatrix(SparseDiagonalMatrix&& rhs);
      /// Move assignment operator
      SparseDiagonalMatrix& operator=(SparseDiagonalMatrix&& rhs);
      /// Copy ctor
      SparseDiagonalMatrix(const SparseDiagonalMatrix& rhs);
      /// Copy assignment operator
      SparseDiagonalMatrix& operator=(const SparseDiagonalMatrix& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Basic constructor
      SparseDiagonalMatrix (
          unsigned int n_rows,
          unsigned int n_cols,
          const std::vector<int>& offsets);
      /// Overload the + operator
      friend SparseDiagonalMatrix operator+(
          const SparseDiagonalMatrix& l_mat,
          const SparseDiagonalMatrix& r_mat);
      /// Overload the + operator for Impl
      friend SparseDiagonalMatrix::Impl operator+(
          const SparseDiagonalMatrix::Impl& l_impl,
          const SparseDiagonalMatrix::Impl& r_impl);
      /// Get a pointer to the matrix values
      float *get_raw_pointer();
      /// Return vector of diagonal offsets
      std::vector<int> get_offsets() const;
      /// Convert to the armadillo SpMat compressed sparse column format
      arma::sp_fmat convert_to_csc();
    private:
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // SparseDiagonalMatrix
} /// namespace MMORF
#endif // SPARSE_DIAGONAL_MATRIX_CUH
