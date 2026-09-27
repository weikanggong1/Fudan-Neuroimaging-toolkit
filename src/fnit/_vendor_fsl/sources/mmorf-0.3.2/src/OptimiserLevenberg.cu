//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Nonliner optimiser utilising the Levenberg only adaptation of Gauss-Newton
///        optimisation
/// \details This should only be applied to Sum of Squared Differences based cost functions
/// \author Frederik Lange
/// \date October 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "MmorfMemory.h"
#include "OptimiserLevenberg.cuh"
#include "CostFxn.h"
#include "LinearSolver.h"

#include <armadillo>

#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <memory>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class OptimiserLevenberg::Impl
  {
    public:
      Impl(std::shared_ptr<MMORF::LinearSolver> lin_solver)
        : lin_solver_(lin_solver)
      {};
      /// Perform optimisation
      /// \details Returns the parameters which optimise the function
      /// \param cost_fxn Cost function to be minimised
      /// \param starting_params Initial guess of optimal parameters
      std::vector<std::vector<float> > optimise(
          MMORF::CostFxn&                         cost_fxn,
          const std::vector<std::vector<float> >& params_start,
          const int                               iterations_max,
          const float                             tolerance_relative
          ) const
      {
        // Set up Levenberg parameters
        auto lambda_l = 1e-6f;
        const auto lambda_l_update = 10.0f; 
        // Copy parameters
        auto params = params_start;
        // Set starting parameters
        cost_fxn.set_parameters(params);
        // Calculate cost
        auto cost_init = cost_fxn.cost();
        std::cout<<"cost_init = "<< cost_init <<std::endl;
        while (!std::isfinite(cost_init))
        {
          for (auto& param_vec : params){
            for (auto& param : param_vec){
              param *= 0.99f;
            }
          }
          cost_fxn.set_parameters(params);
          cost_init = cost_fxn.cost();
          std::cout<<"cost_init = "<< cost_init <<std::endl;
        }
        // Set initial cost difference
        auto cost_change_relative = 2.0 * tolerance_relative;
        for (
            auto i = 0;
            i < iterations_max && cost_change_relative > tolerance_relative;
            ++i){
          std::cout << i << std::endl;
          auto cost = cost_fxn.cost();
          // - Calculate gradient
          auto grad = cost_fxn.grad();
          // - Calculate hessian
          auto hess_lambda = cost_fxn.hess();
          std::cout << "lambda_l = " << lambda_l << std::endl;
          // Decrease lambda
          if (lambda_l > 1e-6f) lambda_l /=(lambda_l_update);
          // Calculate update at lower lambda
          hess_lambda.add_to_main_diagonal(lambda_l);
          auto update = lin_solver_->solve(hess_lambda, -grad);
          auto params_next = add_parameters_(params, update);
          // Apply update and calculate cost
          cost_fxn.set_parameters(params_next);
          auto cost_next = cost_fxn.cost();
          // Only perform the next part if the cost has increased
          while (cost_next > cost){
            // Undo addition of previous lambda
            hess_lambda.add_to_main_diagonal(-lambda_l);
            // Keep increasing lambda until the cost no longer increases
            lambda_l *= lambda_l_update;
            //Calculate update at higher lambda
            hess_lambda.add_to_main_diagonal(lambda_l);
            update = lin_solver_->solve(hess_lambda, -grad);
            params_next = add_parameters_(params, update);
            // Apply update and calculate cost
            cost_fxn.set_parameters(params_next);
            cost_next = cost_fxn.cost();
          }
          std::cout<<"cost_next = "<<cost_next<< std::endl;
          cost_change_relative = std::abs((cost - cost_next)/cost);
          params = params_next;
        }
        // Return the (hopefully) optimised parameters
        return params;
      }

    private:
      //////////////////////////////////////////////////
      // Private functions
      //////////////////////////////////////////////////
      std::vector<std::vector<float> > add_parameters_(
          const std::vector<std::vector<float> >& p1,
          const arma::fvec& p2) const
      {
        auto result = std::vector<std::vector<float> >(p1);
        for (auto dim = 0; dim < p1.size(); ++dim){
          for (auto param = 0; param < p1[0].size(); ++param){
            result[dim][param] += p2[dim*p1[0].size() + param];
          }
        }
        return result;
      }
      //////////////////////////////////////////////////
      // Private datamembers
      //////////////////////////////////////////////////
      std::shared_ptr<MMORF::LinearSolver> lin_solver_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  OptimiserLevenberg::~OptimiserLevenberg() = default;
  /// Move ctor
  OptimiserLevenberg::OptimiserLevenberg(OptimiserLevenberg&& rhs) = default;
  /// Move assignment operator
  OptimiserLevenberg& OptimiserLevenberg::operator=(OptimiserLevenberg&& rhs) = default;
  /// Copy ctor
  OptimiserLevenberg::OptimiserLevenberg(const OptimiserLevenberg& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  OptimiserLevenberg& OptimiserLevenberg::operator=(const OptimiserLevenberg& rhs)
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
  OptimiserLevenberg::OptimiserLevenberg(std::shared_ptr<MMORF::LinearSolver> lin_solver)
    : pimpl_(MMORF::make_unique<Impl>(lin_solver))
  {}
  /// Perform optimisation
  /// \details Returns the parameters which optimise the function
  /// \param cost_fxn Cost function to be minimised
  /// \param starting_params Initial guess of optimal parameters
  std::vector<std::vector<float> > OptimiserLevenberg::optimise(
      MMORF::CostFxn&                         cost_fxn,
      const std::vector<std::vector<float> >& starting_params,
      const int                               iterations_max,
      const float                             tolerance_relative
      ) const
  {
    return pimpl_->optimise(
        cost_fxn,
        starting_params,
        iterations_max,
        tolerance_relative);
  }
} // MMORF
