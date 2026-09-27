//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the log of the singular values of the local Jacobian of the
///        warp field
/// \details This cost function is designed primarily for use in regularising warps defined
///          by B-splines, and not as a stand-alone cost function. As the analytical forms of
///          the gradient and Hessian are rather complicated, the code to calculate them was
///          formulated with the help of the Matlab Symbolic Toolbox.
/// \author Frederik Lange
/// \date October 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef COST_FXN_LOG_JACOBIAN_SINGULAR_VALUES_EXACT_CUH
#define COST_FXN_LOG_JACOBIAN_SINGULAR_VALUES_EXACT_CUH

#include "CostFxn.h"
#include "WarpFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>
#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxnLogJacobianSingularValuesExact : public MMORF::CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnLogJacobianSingularValuesExact();
      /// Move ctor
      CostFxnLogJacobianSingularValuesExact(CostFxnLogJacobianSingularValuesExact&& rhs);
      /// Move assignment operator
      CostFxnLogJacobianSingularValuesExact& operator=(CostFxnLogJacobianSingularValuesExact&& rhs);
      /// Copy ctor
      CostFxnLogJacobianSingularValuesExact(const CostFxnLogJacobianSingularValuesExact& rhs);
      /// Copy assignment operator
      CostFxnLogJacobianSingularValuesExact& operator=(const CostFxnLogJacobianSingularValuesExact& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct by passing in a fully constructed warpfield
      CostFxnLogJacobianSingularValuesExact(
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
  }; // CostFxnLogJacobianSingularValuesExact
} // MMORF
#endif // COST_FXN_LOG_JACOBIAN_SINGULAR_VALUES_EXACT_CUH
