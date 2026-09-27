//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function of cost functions
/// \details This cost function is composed of a number of weighted cost functions and an
///          additional cost function acting as a regulariser. The total cost/grad/hess is
///          a weighted sum of the outputs of all the sub cost functions
/// \author Frederik Lange
/// \date Mar 2020
/// \copyright Copyright (C) 2020 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "MmorfMemory.h"
#include "SparseDiagonalMatrixTiled.cuh"
#include "CostFxnCompoundBiasField.cuh"
#include "BiasField.h"
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
  class CostFxnCompoundBiasField::Impl
  {
    public:
      /// Construct using two pre-made cost functions and a regularisation factor lambda
      Impl(
          std::vector<std::shared_ptr<MMORF::CostFxn> >  cost_functions,
          const std::vector<float>&                      lambda_cost_functions,
          std::shared_ptr<MMORF::CostFxn>                regulariser,
          const float                                    lambda_regulariser,
          std::shared_ptr<MMORF::BiasField>              common_bias_field
          )
        : cost_functions_(cost_functions)
        , lambda_cost_functions_(lambda_cost_functions)
        , regulariser_(regulariser)
        , lambda_regulariser_(lambda_regulariser)
        , common_bias_field_(common_bias_field)
      {
        // Check for the correct number of lambdas and cost functions
        /// \todo Replace assert with exception
        assert(cost_functions_.size() == lambda_cost_functions_.size());
      }
      /// Get the current value of the parameters
      std::vector<std::vector<float> > get_parameters() const
      {
        return common_bias_field_->get_parameters();
      }
      /// Set the current value of the parameters
      void set_parameters(
          const std::vector<std::vector<float> >& parameters)
      {
        common_bias_field_->set_parameters(parameters);
      }
      /// Current value of the cost function
      float cost() const
      {
        auto cost = lambda_regulariser_ * regulariser_->cost();
        auto cost_reg = cost;
        if (!std::isinf(cost_reg)){
          for (auto i = 0; i < cost_functions_.size(); ++i){
            cost += lambda_cost_functions_[i] * cost_functions_[i]->cost();
          }
        }
        auto cost_ssd = cost - cost_reg;
        std::cout
          << "          SSD Cost: " << cost_ssd
          << "          REG Cost: " << cost_reg
          << std::endl;
        return cost;
      }
      /// Derivitive of cost function at current parameterisation
      arma::fvec grad() const
      {
        auto grad = arma::fvec(lambda_regulariser_ * regulariser_->grad());
        for (auto i = 0; i < cost_functions_.size(); ++i){
          grad += lambda_cost_functions_[i] * cost_functions_[i]->grad();
        }
        return grad;
      }
      /// Hessian of cost function at current parameterisation
      /// \todo Implement += for SparseDiagonalMatrixTiled
      MMORF::SparseDiagonalMatrixTiled hess() const
      {
        auto hess = lambda_regulariser_ * regulariser_->hess();
        for (auto i = 0; i < cost_functions_.size(); ++i){
          auto tmp_hess = cost_functions_[i]->hess();
          tmp_hess *= (lambda_cost_functions_[i]);
          hess += tmp_hess;
        }
        return hess;
      }
    private:
      std::vector<std::shared_ptr<MMORF::CostFxn> >  cost_functions_;
      std::vector<float>                             lambda_cost_functions_;
      std::shared_ptr<MMORF::CostFxn>                regulariser_;
      float                                          lambda_regulariser_;
      std::shared_ptr<MMORF::BiasField>              common_bias_field_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  CostFxnCompoundBiasField::~CostFxnCompoundBiasField() = default;
  /// Move ctor
  CostFxnCompoundBiasField::CostFxnCompoundBiasField(CostFxnCompoundBiasField&& rhs) = default;
  /// Move assignment operator
  CostFxnCompoundBiasField& CostFxnCompoundBiasField::operator=(CostFxnCompoundBiasField&& rhs) = default;
  /// Copy ctor
  CostFxnCompoundBiasField::CostFxnCompoundBiasField(const CostFxnCompoundBiasField& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  CostFxnCompoundBiasField& CostFxnCompoundBiasField::operator=(const CostFxnCompoundBiasField& rhs)
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
  CostFxnCompoundBiasField::CostFxnCompoundBiasField(
      std::vector<std::shared_ptr<MMORF::CostFxn> >  cost_functions,
      const std::vector<float>&                      lambda_cost_functions,
      std::shared_ptr<MMORF::CostFxn>                regulariser,
      const float                                    lambda_regulariser,
      std::shared_ptr<MMORF::BiasField>              common_bias_field
      )
    : pimpl_(MMORF::make_unique<Impl>(
        cost_functions,
        lambda_cost_functions,
        regulariser,
        lambda_regulariser,
        common_bias_field))
  {}
  /// Get the current value of the parameters
  std::vector<std::vector<float> > CostFxnCompoundBiasField::get_parameters() const
  {
    return pimpl_->get_parameters();
  }
  /// Set the current value of the parameters
  void CostFxnCompoundBiasField::set_parameters(
      const std::vector<std::vector<float> >& parameters)
  {
    pimpl_->set_parameters(parameters);
  }
  /// Current value of the cost function
  float CostFxnCompoundBiasField::cost() const
  {
    return pimpl_->cost();
  }
  /// Derivitive of cost function at current parameterisation
  arma::fvec CostFxnCompoundBiasField::grad() const
  {
    return pimpl_->grad();
  }
  /// Hessian of cost function at current parameterisation
  MMORF::SparseDiagonalMatrixTiled CostFxnCompoundBiasField::hess() const
  {
    return pimpl_->hess();
  }
} // MMORF
