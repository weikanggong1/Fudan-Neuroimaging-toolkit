//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the log of the singular values of the local Jacobian of the
///        warp field
/// \details This cost function is designed primarily for use in regularising warps defined
///          by B-splines, and not as a stand-alone cost function. As the analytical forms of
///          the gradient and Hessian are rather complicated, the code to calculate them was
///          formulated with the help of the Matlab Symbolic Toolbox.
///          Note that the Hessian calculation uses an approximation whereby it is represented
///          by a single main diagonal made up of the sum of absolute values of each
///          row/column (which is the same thing as H is symmetrical).
/// \author Frederik Lange
/// \date July 2019
/// \copyright Copyright (C) 2019 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef COST_FXN_LOG_JACOBIAN_SINGULAR_VALUES_DIAG_HESS_CUH
#define COST_FXN_LOG_JACOBIAN_SINGULAR_VALUES_DIAG_HESS_CUH

#include "CostFxn.h"
#include "WarpFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>
#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxnLogJacobianSingularValuesDiagHess : public MMORF::CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnLogJacobianSingularValuesDiagHess();
      /// Move ctor
      CostFxnLogJacobianSingularValuesDiagHess(
          CostFxnLogJacobianSingularValuesDiagHess&& rhs);
      /// Move assignment operator
      CostFxnLogJacobianSingularValuesDiagHess& operator=(
          CostFxnLogJacobianSingularValuesDiagHess&& rhs);
      /// Copy ctor
      CostFxnLogJacobianSingularValuesDiagHess(
          const CostFxnLogJacobianSingularValuesDiagHess& rhs);
      /// Copy assignment operator
      CostFxnLogJacobianSingularValuesDiagHess& operator=(
          const CostFxnLogJacobianSingularValuesDiagHess& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct by passing in a fully constructed warpfield
      CostFxnLogJacobianSingularValuesDiagHess(
          std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          int sampling_frequency);
      /// Get the current value of the parameters
      std::vector<std::vector<float> > get_parameters() const override;
      /// Set the current value of the parameters
      void set_parameters(
          const std::vector<std::vector<float> >& parameters) override;
      /// Get cost under current parameterisation
      float cost() const override;
      /// Get Jte under current parameterisation
      arma::fvec grad() const override;
      /// Get JtJ under current parameterisation
      MMORF::SparseDiagonalMatrixTiled hess() const override;
    private:
      /// Forward declaration
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // CostFxnLogJacobianSingularValuesDiagHess
} // MMORF
#endif // COST_FXN_LOG_JACOBIAN_SINGULAR_VALUES_DIAG_HESS_CUH
