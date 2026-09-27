//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Linear solver basic interface
/// \details Provides the interface to which all derived LinearSolvers must conform. Solves
///          the system Ax = b for a given A coefficient matrix and b constant column vector
/// \author Frederik Lange
/// \date July 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H

#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class LinearSolver
  {
    public:
      /// Default destructor
      virtual ~LinearSolver() {}
      /// Solve the linear system
      /// \details Returns the vector x = A^-1b
      /// \param coefficients Matrix A of coefficeints
      /// \param constants Vector b of constants
      virtual arma::fmat solve(
          const MMORF::SparseDiagonalMatrixTiled& coefficients,
          const arma::fmat&                       constants) const = 0;
  }; // LinearSolver
} // MMORF
#endif // LINEAR_SOLVER_H
