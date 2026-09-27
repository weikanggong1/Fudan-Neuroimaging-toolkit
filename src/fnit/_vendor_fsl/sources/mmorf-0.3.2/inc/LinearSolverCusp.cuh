//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Linear solver implemented using armadillo's built-in linear solving mechanics
/// \details Note that this does not use an iterative method, but rather a sparse LU
///          decomposition and as such is probably going to perform quire slowly
/// \author Frederik Lange
/// \date July 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef LINEAR_SOLVER_CUSP_H
#define LINEAR_SOLVER_CUSP_H

#include "LinearSolver.h"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class LinearSolverCusp : public LinearSolver
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~LinearSolverCusp();
      /// Move ctor
      LinearSolverCusp(LinearSolverCusp&& rhs);
      /// Move assignment operator
      LinearSolverCusp& operator=(LinearSolverCusp&& rhs);
      /// Copy ctor
      LinearSolverCusp(const LinearSolverCusp& rhs);
      /// Copy assignment operator
      LinearSolverCusp& operator=(const LinearSolverCusp& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Default ctor
      LinearSolverCusp(
          const int   max_iterations = 100,
          const float relative_tolerance = 1e-3f);
      /// Solve the linear system
      /// \details Returns the vector x = A^-1b
      /// \param coefficients Matrix A of coefficeints
      /// \param constants Vector b of constants
      virtual arma::fmat solve(
          const MMORF::SparseDiagonalMatrixTiled& coefficients,
          const arma::fmat&                       constants) const override;
    private:
      /// Forward declaration
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // LinearSolverCusp
} // MMORF
#endif // LINEAR_SOLVER_CUSP_H
