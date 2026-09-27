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
#ifndef COST_FXN_LOG_JACOBIAN_SINGULAR_VALUES_CUH
#define COST_FXN_LOG_JACOBIAN_SINGULAR_VALUES_CUH

#include "CostFxn.h"
#include "WarpFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>
#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxnLogJacobianSingularValues : public MMORF::CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnLogJacobianSingularValues();
      /// Move ctor
      CostFxnLogJacobianSingularValues(
          CostFxnLogJacobianSingularValues&& rhs);
      /// Move assignment operator
      CostFxnLogJacobianSingularValues& operator=(
          CostFxnLogJacobianSingularValues&& rhs);
      /// Copy ctor
      CostFxnLogJacobianSingularValues(
          const CostFxnLogJacobianSingularValues& rhs);
      /// Copy assignment operator
      CostFxnLogJacobianSingularValues& operator=(
          const CostFxnLogJacobianSingularValues& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct by passing in a fully constructed warpfield
      CostFxnLogJacobianSingularValues(
          std::shared_ptr<MMORF::WarpFieldBSpline>  warp_field,
          int                                       sampling_frequency,
          const bool                                debug = false);
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
      class                 Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
      /// Global class identifier allocator
      static int            next_id_;
  }; // CostFxnLogJacobianSingularValues
} // MMORF
#endif // COST_FXN_LOG_JACOBIAN_SINGULAR_VALUES_CUH
