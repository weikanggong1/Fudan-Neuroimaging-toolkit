//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the log of the local Jacobian of thevwarp field
/// \details This cost function is designed primarily for use in regularising warps defined
///          by B-splines, and not as a stand-alone cost function. As the analytical forms of
///          the gradient and Hessian are rather complicated, the code to calculate them was
///          formulated with the help of the Matlab Symbolic Toolbox.
/// \author Frederik Lange
/// \date November 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef COST_FXN_LOG_JACOBIAN_CUH
#define COST_FXN_LOG_JACOBIAN_CUH

#include "CostFxn.h"
#include "WarpFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>
#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxnLogJacobian : public MMORF::CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnLogJacobian();
      /// Move ctor
      CostFxnLogJacobian(CostFxnLogJacobian&& rhs);
      /// Move assignment operator
      CostFxnLogJacobian& operator=(CostFxnLogJacobian&& rhs);
      /// Copy ctor
      CostFxnLogJacobian(const CostFxnLogJacobian& rhs);
      /// Copy assignment operator
      CostFxnLogJacobian& operator=(const CostFxnLogJacobian& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct by passing in a fully constructed warpfield
      CostFxnLogJacobian(
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
  }; // CostFxnLogJacobian
} // MMORF
#endif // COST_FXN_LOG_JACOBIAN_CUH
