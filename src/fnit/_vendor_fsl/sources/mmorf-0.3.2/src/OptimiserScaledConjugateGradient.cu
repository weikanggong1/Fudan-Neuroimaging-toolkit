//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Nonliner optimiser utilising the Scaled Conjugate Gradient approach
/// \details This is based on the work by Martin Moller - A Scaled Conjugate Gradient
///          Algorithm for Fast Supervised Learning (1993)
/// \author Frederik Lange
/// \date November 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "MmorfMemory.h"
#include "OptimiserScaledConjugateGradient.cuh"
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
  class OptimiserScaledConjugateGradient::Impl
  {
    public:
      Impl()
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
        cost_fxn.set_parameters(params_start);
        auto n_dims = params_start.size();
        auto n_params_per_dim = params_start[0].size();
        auto n_params = n_dims * n_params_per_dim;
        // Note that this uses slightly different notation than other conventions in MMORF
        // in order to be consistent with Moller's paper, allowing that to act as the
        // documentation for the algorithm
        //
        // 1)
        auto w_k = params_start; // Initial parameter guess
        auto w_k_1 = w_k; // Initial parameter guess
        auto sigma = 1.0e-4f; // Infinitesimal offset for analytical calculation of H*p
        auto lambda_k = 1.0e-6f; // Initial weighting of gradient
        auto lambda_bar_k = 0.0f; // Update for lambda in H is not positive definite
        arma::fvec r_vec_k = -cost_fxn.grad(); // Direction of steepest decent
        arma::fvec p_vec_k = r_vec_k; // Search direction
        arma::fvec r_vec_k_1 = r_vec_k; // r_vec for next iteration
        arma::fvec p_vec_k_1 = p_vec_k; // p_vec for next iteration
        auto cost_k = cost_fxn.cost(); // Cost at w_k
        auto delta_k = 0.0f;
        auto success = true; // The previous step was successful
        // 2)
        for (
            auto k = 0;
            k < iterations_max && arma::max(arma::abs(r_vec_k)) > 1e-6f;
            ++k){ // Iteration counter
          std::cout << std::endl << "Beginning iteration " << k << std::endl;
//          std::cout << "Maximum gradient value = " << arma::max(arma::abs(r_vec_k)) << std::endl;
          std::cout << "Cost = " << cost_k << std::endl;
          float p_mag_squared = arma::dot(p_vec_k, p_vec_k);
//          std::cout << "p_mag_squared = " << p_mag_squared << std::endl;
          float p_mag = std::sqrt(p_mag_squared); // Magnitude of p
//          std::cout << "p_mag = " << p_mag << std::endl;
          if (success){
//            auto n_iter = 0;
//            do {
//              std::cout << "Success\n";
              float sigma_k = sigma/p_mag;
              //sigma_k = std::max(1.0e-6f,sigma_k);
//              std::cout << "sigma_k = " << sigma_k << std::endl;
              cost_fxn.set_parameters(add_parameters_(w_k, sigma_k*p_vec_k));
              arma::fvec tmp_r_vec = cost_fxn.grad();
//              std::cout << "max(tmp_r_vec) = " << arma::max(arma::abs(tmp_r_vec)) << std::endl;
//              std::cout << "max(r_vec_k) = " << arma::max(arma::abs(r_vec_k)) << std::endl;
//              cost_fxn.set_parameters(w_k);
//              arma::fvec new_r_vec = -cost_fxn.grad();
              arma::fvec s_vec_k = (tmp_r_vec + r_vec_k)/sigma_k;
//              std::cout << "max(s_vec_k) = " << arma::max(arma::abs(s_vec_k)) << std::endl;
              delta_k = arma::dot(p_vec_k, s_vec_k);
//              std::cout << "delta_k = " << delta_k << std::endl;
//              ++n_iter;
//              if (n_iter == 3){
//                tmp_r_vec.save("debug/tmp_r_vec",arma::raw_ascii);
//                r_vec_k.save("debug/r_vec_k",arma::raw_ascii);
//                new_r_vec.save("debug/new_r_vec_k",arma::raw_ascii);
//              }
//            }
//            while(std::abs(delta_k) > 5000);
          }
          // 3)
          delta_k += (lambda_k - lambda_bar_k)*p_mag_squared;
//          std::cout << "delta_k = " << delta_k << std::endl;
          // 4)
          if (delta_k <= 0.0f){ // Implies H is not positive definite
            lambda_bar_k = 2.0f*(lambda_k - delta_k/p_mag_squared);
//            std::cout << "lambda_bar_k = " << lambda_bar_k << std::endl;
            delta_k = -delta_k + lambda_k*p_mag_squared;
//            std::cout << "delta_k = " << delta_k << std::endl;
            lambda_k = lambda_bar_k;
//            std::cout << "lambda_k = " << lambda_k << std::endl;
          }
          // 5)
          auto mu_k = arma::dot(p_vec_k, r_vec_k);
//          std::cout << "mu_k = " << mu_k << std::endl;
          auto alpha_k = mu_k/delta_k;
//          std::cout << "alpha_k = " << alpha_k << std::endl;
          // 6)
          cost_fxn.set_parameters(add_parameters_(w_k, alpha_k*p_vec_k));
//          std::cout << cost_fxn.cost() << std::endl;
          float big_delta_k = 2.0f*delta_k*(cost_k - cost_fxn.cost())/(mu_k*mu_k);
//          std::cout << "big_delta_k = " << big_delta_k << std::endl;
          // 7)
          if (big_delta_k >= 0.0f){
            w_k_1 = add_parameters_(w_k, alpha_k*p_vec_k);
            cost_fxn.set_parameters(w_k_1);
            r_vec_k_1 = -cost_fxn.grad();
            lambda_bar_k = 0.0f;
            success = true;
            if ((k + 1)%n_params == 0){
              p_vec_k_1 = r_vec_k_1;
            }
            else {
              auto beta_k = (arma::dot(r_vec_k_1,r_vec_k_1)
                  - arma::dot(r_vec_k_1,r_vec_k))/mu_k;
//              std::cout << "beta_k = " << beta_k << std::endl;
              p_vec_k_1 = r_vec_k_1 + beta_k*p_vec_k;
            }
            if (big_delta_k >= 0.75f){
              lambda_k *= 0.25f;
//              std::cout << "lambda_k = " << lambda_k << std::endl;
            }
          }
          else {
            lambda_bar_k = lambda_k;
//            std::cout << "lambda_bar_k = " << lambda_bar_k << std::endl;
            success = false;
          }
          // 7.1) // Because we need this
          if (big_delta_k == - INFINITY){
            lambda_k *= 10.0;
//            std::cout << "lambda_k = " << lambda_k << std::endl;
          }
          // 8)
          else if (big_delta_k < 0.25f){
            lambda_k += delta_k*(1.0f - big_delta_k)/p_mag_squared;
//            std::cout << "lambda_k = " << lambda_k << std::endl;
          }
          // Copy values for next iteration
          r_vec_k = r_vec_k_1; // r_vec for next iteration
          p_vec_k = p_vec_k_1; // p_vec for next iteration
          w_k = w_k_1; // w_k for next iteration
          cost_fxn.set_parameters(w_k);
          cost_k = cost_fxn.cost(); // Cost at w_k
//          std::cout << "max(p_vec_k) = " << arma::max(arma::abs(p_vec_k)) << std::endl;
        }
        // Return the (hopefully) optimised parameters
        return w_k;
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
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  OptimiserScaledConjugateGradient::~OptimiserScaledConjugateGradient() = default;
  /// Move ctor
  OptimiserScaledConjugateGradient::OptimiserScaledConjugateGradient(
      OptimiserScaledConjugateGradient&& rhs) = default;
  /// Move assignment operator
  OptimiserScaledConjugateGradient& OptimiserScaledConjugateGradient::operator=(
      OptimiserScaledConjugateGradient&& rhs) = default;
  /// Copy ctor
  OptimiserScaledConjugateGradient::OptimiserScaledConjugateGradient(
      const OptimiserScaledConjugateGradient& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  OptimiserScaledConjugateGradient& OptimiserScaledConjugateGradient::operator=(
      const OptimiserScaledConjugateGradient& rhs)
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
  OptimiserScaledConjugateGradient::OptimiserScaledConjugateGradient()
    : pimpl_()
  {}
  /// Perform optimisation
  /// \details Returns the parameters which optimise the function
  /// \param cost_fxn Cost function to be minimised
  /// \param starting_params Initial guess of optimal parameters
  std::vector<std::vector<float> > OptimiserScaledConjugateGradient::optimise(
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
