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
#include "MmorfMemory.h"
#include "SparseDiagonalMatrixTiled.cuh"
#include "CostFxnRegularised.cuh"
#include "CostFxn.h"

#include <armadillo>

#include <memory>
#include <vector>
#include <cassert>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class CostFxnRegularised::Impl
  {
    public:
      /// Construct using two pre-made cost functions and a regularisation factor lambda
      Impl(
          std::shared_ptr<MMORF::CostFxn> cost_fxn,
          std::shared_ptr<MMORF::CostFxn> regulariser,
          float lambda_reg)
        : cost_fxn_(cost_fxn)
        , regulariser_(regulariser)
        , lambda_reg_(lambda_reg)
      {
        // Check cost functions have the same parameterisation
        auto cf_params = cost_fxn_->get_parameters();
        auto reg_params = regulariser_->get_parameters();
        assert(cf_params.size() == reg_params.size());
        assert(cf_params[0].size() == reg_params[0].size());
        // Ensure both cost functions have the same parameters
        regulariser_->set_parameters(cost_fxn_->get_parameters());
      }
      /// Get the current value of the parameters
      std::vector<std::vector<float> > get_parameters() const
      {
        return cost_fxn_->get_parameters();
      }
      /// Set the current value of the parameters
      void set_parameters(
          const std::vector<std::vector<float> >& parameters)
      {
        cost_fxn_->set_parameters(parameters);
        regulariser_->set_parameters(parameters);
      }
      /// Current value of the cost function
      float cost() const
      {
        //auto cf = cost_fxn_->cost();
        //auto reg = lambda_reg_*regulariser_->cost();
        //std::cout << "CF:REG = " << cf << ":" << reg << std::endl;
        //return cf + reg;
        return cost_fxn_->cost() + lambda_reg_*regulariser_->cost();
      }
      /// Derivitive of cost function at current parameterisation
      arma::fvec grad() const
      {
        return cost_fxn_->grad() + lambda_reg_*regulariser_->grad();
      }
      /// Hessian of cost function at current parameterisation
      MMORF::SparseDiagonalMatrixTiled hess() const
      {
        return cost_fxn_->hess() + lambda_reg_*regulariser_->hess();
      }
    private:
      std::shared_ptr<MMORF::CostFxn> cost_fxn_;
      std::shared_ptr<MMORF::CostFxn> regulariser_;
      float lambda_reg_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  CostFxnRegularised::~CostFxnRegularised() = default;
  /// Move ctor
  CostFxnRegularised::CostFxnRegularised(CostFxnRegularised&& rhs) = default;
  /// Move assignment operator
  CostFxnRegularised& CostFxnRegularised::operator=(CostFxnRegularised&& rhs) = default;
  /// Copy ctor
  CostFxnRegularised::CostFxnRegularised(const CostFxnRegularised& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  CostFxnRegularised& CostFxnRegularised::operator=(const CostFxnRegularised& rhs)
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
  /// Construct using two pre-made cost functions and a regularisation factor lambda
  CostFxnRegularised::CostFxnRegularised(
        std::shared_ptr<MMORF::CostFxn>  cost_fxn,
        std::shared_ptr<MMORF::CostFxn>  regulariser,
        float lambda_reg)
    : pimpl_(MMORF::make_unique<Impl>(
        cost_fxn,
        regulariser,
        lambda_reg))
  {}
  /// Get the current value of the parameters
  std::vector<std::vector<float> > CostFxnRegularised::get_parameters() const
  {
    return pimpl_->get_parameters();
  }
  /// Set the current value of the parameters
  void CostFxnRegularised::set_parameters(
      const std::vector<std::vector<float> >& parameters)
  {
    pimpl_->set_parameters(parameters);
  }
  /// Current value of the cost function
  float CostFxnRegularised::cost() const
  {
    return pimpl_->cost();
  }
  /// Derivitive of cost function at current parameterisation
  arma::fvec CostFxnRegularised::grad() const
  {
    return pimpl_->grad();
  }
  /// Hessian of cost function at current parameterisation
  MMORF::SparseDiagonalMatrixTiled CostFxnRegularised::hess() const
  {
    return pimpl_->hess();
  }
} // MMORF
