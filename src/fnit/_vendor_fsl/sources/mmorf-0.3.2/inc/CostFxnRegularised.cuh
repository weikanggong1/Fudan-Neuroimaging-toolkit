//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Regularised cost function
/// \details This cost function is composed of both a cost function of interest, and a second
///          cost function to be used as a regulariser. The total cost function is then the
///          sum of both with a regularisaton factor lambda
/// \author Frederik Lange
/// \date May 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef COST_FXN_REGULARISED_CUH
#define COST_FXN_REGULARISED_CUH

#include "CostFxn.h"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>
#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxnRegularised : public CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnRegularised();
      /// Move ctor
      CostFxnRegularised(CostFxnRegularised&& rhs);
      /// Move assignment operator
      CostFxnRegularised& operator=(CostFxnRegularised&& rhs);
      /// Copy ctor
      CostFxnRegularised(const CostFxnRegularised& rhs);
      /// Copy assignment operator
      CostFxnRegularised& operator=(const CostFxnRegularised& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct using two pre-made cost functions and a regularisation factor lambda
      CostFxnRegularised(
          std::shared_ptr<MMORF::CostFxn> cost_fxn,
          std::shared_ptr<MMORF::CostFxn> regulariser,
          float lambda_reg);
      /// Get the current value of the parameters
      virtual std::vector<std::vector<float> > get_parameters() const override;
      /// Set the current value of the parameters
      virtual void set_parameters(
          const std::vector<std::vector<float> >& parameters) override;
      /// Current value of the cost function
      virtual float cost() const override;
      /// Derivitive of cost function at current parameterisation
      virtual arma::fvec grad() const override;
      /// Hessian of cost function at current parameterisation
      virtual MMORF::SparseDiagonalMatrixTiled hess() const override;
    private:
      /// Forward declaration
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // CostFxnRegularised
} // MMORF
#endif // COST_FXN_REGULARISED_CUH
