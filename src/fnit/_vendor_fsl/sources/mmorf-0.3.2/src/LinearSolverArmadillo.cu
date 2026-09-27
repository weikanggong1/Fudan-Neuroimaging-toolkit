//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Linear solver implemented using armadillo's built-in linear solving mechanics
/// \details Note that this does not use an iterative method, but rather a sparse LU
///          decomposition and as such is probably going to perform quire slowly
/// \author Frederik Lange
/// \date July 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "MmorfMemory.h"
#include "SparseDiagonalMatrixTiled.cuh"
#include "LinearSolverArmadillo.cuh"

#include <armadillo>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class LinearSolverArmadillo::Impl
  {
    public:
      /// Solve the linear system
      /// \details Returns the vector x = A^-1b
      /// \param coefficients Matrix A of coefficeints
      /// \param constants Vector b of constants
      arma::fmat solve(
          const MMORF::SparseDiagonalMatrixTiled& coefficients,
          const arma::fmat& constants) const
      {
        auto solver_opts = make_solver_opts_();
        arma::fmat solution = arma::spsolve(
            coefficients.convert_to_csc(),
            constants,
            "superlu",
            solver_opts);
        return solution;
      }
    private:
      arma::superlu_opts make_solver_opts_() const
      {
        // Setup for SuperLU solver
        auto solver_opts = arma::superlu_opts();
        solver_opts.equilibrate = false;
        solver_opts.symmetric = true;
        solver_opts.pivot_thresh = 0.0;
        solver_opts.permutation = arma::superlu_opts::NATURAL;
        solver_opts.refine = arma::superlu_opts::REF_NONE;
        return solver_opts;
      }
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  LinearSolverArmadillo::~LinearSolverArmadillo() = default;
  /// Move ctor
  LinearSolverArmadillo::LinearSolverArmadillo(LinearSolverArmadillo&& rhs) = default;
  /// Move assignment operator
  LinearSolverArmadillo& LinearSolverArmadillo::operator=(LinearSolverArmadillo&& rhs) = default;
  /// Copy ctor
  LinearSolverArmadillo::LinearSolverArmadillo(const LinearSolverArmadillo& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  LinearSolverArmadillo& LinearSolverArmadillo::operator=(const LinearSolverArmadillo& rhs)
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
  /// Default constructor
  LinearSolverArmadillo::LinearSolverArmadillo()
    : pimpl_(MMORF::make_unique<Impl>())
  {}
  /// Solve the linear system
  /// \details Returns the vector x = A^-1b
  /// \param coefficients Matrix A of coefficeints
  /// \param constants Vector b of constants
  arma::fmat LinearSolverArmadillo::solve(
      const MMORF::SparseDiagonalMatrixTiled& coefficients,
      const arma::fmat& constants) const
  {
    return pimpl_->solve(coefficients, constants);
  }
} // MMORF
