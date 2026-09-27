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
#include "LinearSolverCusp.cuh"

#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/convert.h>
#include <cusp/precond/diagonal.h>

#include <thrust/copy.h>

#include <armadillo>

#include <vector>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class LinearSolverCusp::Impl
  {
    public:
      Impl(
        const int   max_iterations,
        const float relative_tolerance)
        : max_iterations_(max_iterations)
        , relative_tolerance_(relative_tolerance)
      {
        std::cout << max_iterations << std::endl;
        std::cout << relative_tolerance << std::endl;
      }
      /// Solve the linear system
      /// \details Returns the vector x = A^-1b
      /// \param coefficients Matrix A of coefficeints
      /// \param constants Vector b of constants
      arma::fmat solve(
          const MMORF::SparseDiagonalMatrixTiled& coefficients,
          const arma::fmat&                       constants) const
      {
        auto coefficients_cusp = coefficients.convert_to_csr();
        // Convert to CUSP 1D arrays
        auto constants_cusp = cusp::array1d<float, cusp::device_memory>(
            arma::conv_to<std::vector<float> >::from(constants));
        auto solution_cusp = cusp::array1d<float, cusp::device_memory>(
            constants_cusp.size(),
            0.0f);
        // Create monitor
        auto monitor = cusp::monitor<float>(
            constants_cusp,
            max_iterations_,
            relative_tolerance_,
            0,
            false);
        // Create identity preconditioner (i.e. none);
        //auto preconditioner = cusp::identity_operator<float, cusp::device_memory>(
        //    constants_cusp.size(),
        //    constants_cusp.size());
        auto preconditioner = cusp::precond::diagonal<float, cusp::device_memory>(
            coefficients_cusp);
        // Solve the linear system
        cusp::krylov::cg(
            coefficients_cusp,
            solution_cusp,
            constants_cusp,
            monitor,
            preconditioner);
        auto solution_vec = std::vector<float>(solution_cusp.size());
        thrust::copy(solution_cusp.begin(), solution_cusp.end(), solution_vec.begin());
        auto solution = arma::fvec(solution_vec);
        return solution;
      }
    private:
      int   max_iterations_;
      float relative_tolerance_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  LinearSolverCusp::~LinearSolverCusp() = default;
  /// Move ctor
  LinearSolverCusp::LinearSolverCusp(LinearSolverCusp&& rhs) = default;
  /// Move assignment operator
  LinearSolverCusp& LinearSolverCusp::operator=(LinearSolverCusp&& rhs) = default;
  /// Copy ctor
  LinearSolverCusp::LinearSolverCusp(const LinearSolverCusp& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  LinearSolverCusp& LinearSolverCusp::operator=(const LinearSolverCusp& rhs)
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
  LinearSolverCusp::LinearSolverCusp(
      const int   max_iterations,
      const float relative_tolerance)
    : pimpl_(MMORF::make_unique<Impl>(max_iterations, relative_tolerance))
  {}
  /// Solve the linear system
  /// \details Returns the vector x = A^-1b
  /// \param coefficients Matrix A of coefficeints
  /// \param constants Vector b of constants
  arma::fmat LinearSolverCusp::solve(
      const MMORF::SparseDiagonalMatrixTiled& coefficients,
      const arma::fmat&                       constants) const
  {
    return pimpl_->solve(
        coefficients,
        constants);
  }
} // MMORF
