//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Linear solver implemented using armadillo's built-in linear solving mechanics
/// \details Note that this does not use an iterative method, but rather a sparse LU
///          decomposition and as such is probably going to perform quire slowly
/// \author Frederik Lange
/// \date July 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef LINEAR_SOLVER_ARMADILLO_H
#define LINEAR_SOLVER_ARMADILLO_H

#include "LinearSolver.h"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class LinearSolverArmadillo : public LinearSolver
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~LinearSolverArmadillo();
      /// Move ctor
      LinearSolverArmadillo(LinearSolverArmadillo&& rhs);
      /// Move assignment operator
      LinearSolverArmadillo& operator=(LinearSolverArmadillo&& rhs);
      /// Copy ctor
      LinearSolverArmadillo(const LinearSolverArmadillo& rhs);
      /// Copy assignment operator
      LinearSolverArmadillo& operator=(const LinearSolverArmadillo& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Default ctor
      LinearSolverArmadillo();
      /// Solve the linear system
      /// \details Returns the vector x = A^-1b
      /// \param coefficients Matrix A of coefficeints
      /// \param constants Vector b of constants
      virtual arma::fmat solve(
          const MMORF::SparseDiagonalMatrixTiled& coefficients,
          const arma::fmat& constants) const override;
    private:
      /// Forward declaration
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // LinearSolverArmadillo
} // MMORF
#endif // LINEAR_SOLVER_ARMADILLO_H
