//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function of cost functions, with variance weighting
/// \details This cost function is composed of a number of weighted cost functions and an
///          additional cost function acting as a regulariser. The total cost/grad/hess is
///          a weighted sum of the outputs of all the sub cost functions. Additionally, the
///          non-regularising cost functions are scaled according to the inverse of their
///          variance at the the minimum cost observed up until that point in time.
///          NB!NB!NB!NB! THIS MEANS THAT THE SAME PARAMETERS MAY RETURN DIFFERENT RESULTS
///          FOR COST/GRAD/HESS DEPENDING ON WHEN THEY ARE SET!!!
/// \author Frederik Lange
/// \date January 2019
/// \copyright Copyright (C) 2019 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "MmorfMemory.h"
#include "SparseDiagonalMatrixTiled.cuh"
#include "CostFxnCompoundWarpFieldVarianceScaled.cuh"
#include "WarpField.h"
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
  class CostFxnCompoundWarpFieldVarianceScaled::Impl
  {
    public:
      /// Construct using two pre-made cost functions and a regularisation factor lambda
      Impl(
          std::vector<std::shared_ptr<MMORF::CostFxn> >  cost_functions,
          const std::vector<float>&                      lambda_cost_functions,
          std::shared_ptr<MMORF::CostFxn>                regulariser,
          const float                                    lambda_regulariser,
          std::shared_ptr<MMORF::WarpField>              common_warp_field
          )
        : cost_functions_(cost_functions)
        , lambda_cost_functions_(lambda_cost_functions)
        , regulariser_(regulariser)
        , lambda_regulariser_(lambda_regulariser)
        , common_warp_field_(common_warp_field)
      {
        // Check for the correct number of lambdas and cost functions
        /// \todo Replace assert with exception
        assert(cost_functions_.size() == lambda_cost_functions_.size());
        // Calculate initial cost per cost function and total cost
        auto cost = lambda_regulariser_ * regulariser_->cost();
        if (std::isfinite(cost)){
          for (auto i = 0; i < cost_functions_.size(); ++i){
            auto cost_i = cost_functions_[i]->cost();
            original_cost_values_.push_back(50.0f);
            minimum_cost_values_.push_back(cost_i);
            auto cost_scaling = original_cost_values_[i]/minimum_cost_values_[i];
            cost += lambda_cost_functions_[i] * cost_scaling * cost_i;
          }
        }
        else{
          for (auto i = 0; i < cost_functions_.size(); ++i){
            auto cost_i = std::numeric_limits<float>::max();
            original_cost_values_.push_back(50.0f);
            minimum_cost_values_.push_back(cost_i);
            auto cost_scaling = original_cost_values_[i]/minimum_cost_values_[i];
            cost += lambda_cost_functions_[i] * cost_scaling * cost_i;
          }
        }
        minimum_cost_total_ = cost;
      }
      /// Get the current value of the parameters
      std::vector<std::vector<float> > get_parameters() const
      {
        return common_warp_field_->get_parameters();
      }
      /// Set the current value of the parameters
      void set_parameters(
          const std::vector<std::vector<float> >& parameters)
      {
        common_warp_field_->set_parameters(parameters);
      }
      /// Current value of the cost function
      float cost() const
      {
        auto cost_reg = lambda_regulariser_ * regulariser_->cost();
        auto cost = cost_reg;
        if (std::isfinite(cost_reg)){
          auto temp_cost_values =
            std::vector<float>(original_cost_values_.size());
          for (auto i = 0; i < cost_functions_.size(); ++i){
            auto cost_i = cost_functions_[i]->cost();
            temp_cost_values[i] = cost_i;
            auto cost_scaling = original_cost_values_[i]/minimum_cost_values_[i];
            cost += lambda_cost_functions_[i] * cost_scaling * cost_i;
          }
          auto cost_ssd = cost - cost_reg;
          std::cout
            << "          SSD Cost: " << cost_ssd
            << "          REG Cost: " << cost_reg << std::endl
            << "          MIN Cost: " << minimum_cost_total_
            << "              Cost: " << cost
            << std::endl;
          if (cost < minimum_cost_total_){
            minimum_cost_values_ = temp_cost_values;
            auto cost_min = cost_reg;
            for (auto i = 0; i < cost_functions_.size(); ++i){
              auto cost_i = temp_cost_values[i];
              auto cost_scaling = original_cost_values_[i]/minimum_cost_values_[i];
              cost_min += lambda_cost_functions_[i] * cost_scaling * cost_i;
            }
            minimum_cost_total_ = cost_min;
          }
        }
        return cost;
      }
      /// Derivitive of cost function at current parameterisation
      arma::fvec grad() const
      {
        auto grad = arma::fvec(lambda_regulariser_ * regulariser_->grad());
        for (auto i = 0; i < cost_functions_.size(); ++i){
          auto grad_i = cost_functions_[i]->grad();
          auto cost_scaling = original_cost_values_[i]/minimum_cost_values_[i];
          grad += lambda_cost_functions_[i] * cost_scaling * grad_i;
        }
        return grad;
      }
      /// Hessian of cost function at current parameterisation
      /// \todo Implement += for SparseDiagonalMatrixTiled
      MMORF::SparseDiagonalMatrixTiled hess() const
      {
        auto hess = regulariser_->hess();
        hess *= lambda_regulariser_;
        for (auto i = 0; i < cost_functions_.size(); ++i){
          auto cost_scaling = original_cost_values_[i]/minimum_cost_values_[i];
          auto tmp_hess = cost_functions_[i]->hess();
          tmp_hess *= (lambda_cost_functions_[i] * cost_scaling);
          hess += tmp_hess;
        }
        return hess;
      }
    private:
      std::vector<std::shared_ptr<MMORF::CostFxn> >  cost_functions_;
      std::vector<float>                             lambda_cost_functions_;
      std::shared_ptr<MMORF::CostFxn>                regulariser_;
      float                                          lambda_regulariser_;
      std::shared_ptr<MMORF::WarpField>              common_warp_field_;
      mutable std::vector<float>                     minimum_cost_values_;
      std::vector<float>                             original_cost_values_;
      mutable float                                  minimum_cost_total_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  CostFxnCompoundWarpFieldVarianceScaled::~CostFxnCompoundWarpFieldVarianceScaled() = default;
  /// Move ctor
  CostFxnCompoundWarpFieldVarianceScaled::CostFxnCompoundWarpFieldVarianceScaled(
      CostFxnCompoundWarpFieldVarianceScaled&& rhs) = default;
  /// Move assignment operator
  CostFxnCompoundWarpFieldVarianceScaled& CostFxnCompoundWarpFieldVarianceScaled::operator=(
      CostFxnCompoundWarpFieldVarianceScaled&& rhs) = default;
  /// Copy ctor
  CostFxnCompoundWarpFieldVarianceScaled::CostFxnCompoundWarpFieldVarianceScaled(
      const CostFxnCompoundWarpFieldVarianceScaled& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  CostFxnCompoundWarpFieldVarianceScaled& CostFxnCompoundWarpFieldVarianceScaled::operator=(
      const CostFxnCompoundWarpFieldVarianceScaled& rhs)
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
  CostFxnCompoundWarpFieldVarianceScaled::CostFxnCompoundWarpFieldVarianceScaled(
      std::vector<std::shared_ptr<MMORF::CostFxn> >  cost_functions,
      const std::vector<float>&                      lambda_cost_functions,
      std::shared_ptr<MMORF::CostFxn>                regulariser,
      const float                                    lambda_regulariser,
      std::shared_ptr<MMORF::WarpField>              common_warp_field
      )
    : pimpl_(MMORF::make_unique<Impl>(
        cost_functions,
        lambda_cost_functions,
        regulariser,
        lambda_regulariser,
        common_warp_field))
  {}
  /// Get the current value of the parameters
  std::vector<std::vector<float> >
    CostFxnCompoundWarpFieldVarianceScaled::get_parameters() const
  {
    return pimpl_->get_parameters();
  }
  /// Set the current value of the parameters
  void CostFxnCompoundWarpFieldVarianceScaled::set_parameters(
      const std::vector<std::vector<float> >& parameters)
  {
    pimpl_->set_parameters(parameters);
  }
  /// Current value of the cost function
  float CostFxnCompoundWarpFieldVarianceScaled::cost() const
  {
    return pimpl_->cost();
  }
  /// Derivitive of cost function at current parameterisation
  arma::fvec CostFxnCompoundWarpFieldVarianceScaled::grad() const
  {
    return pimpl_->grad();
  }
  /// Hessian of cost function at current parameterisation
  MMORF::SparseDiagonalMatrixTiled CostFxnCompoundWarpFieldVarianceScaled::hess() const
  {
    return pimpl_->hess();
  }
} // MMORF
