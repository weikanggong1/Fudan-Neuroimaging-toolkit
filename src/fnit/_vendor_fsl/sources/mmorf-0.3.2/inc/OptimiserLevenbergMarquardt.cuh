//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Nonliner optimiser utilising the Levenberg-Marquardt adaptation of Gauss-Newton
///        optimisation
/// \details This should only be applied to Sum of Squared Differences based cost functions
/// \author Frederik Lange
/// \date May 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef OPTIMISER_LEVENBERG_MARQUARDT_CUH
#define OPTIMISER_LEVENBERG_MARQUARDT_CUH

#include "Optimiser.h"
#include "CostFxn.h"
#include "LinearSolver.h"

#include <memory>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class OptimiserLevenbergMarquardt : public Optimiser
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~OptimiserLevenbergMarquardt();
      /// Move ctor
      OptimiserLevenbergMarquardt(OptimiserLevenbergMarquardt&& rhs);
      /// Move assignment operator
      OptimiserLevenbergMarquardt& operator=(OptimiserLevenbergMarquardt&& rhs);
      /// Copy ctor
      OptimiserLevenbergMarquardt(const OptimiserLevenbergMarquardt& rhs);
      /// Copy assignment operator
      OptimiserLevenbergMarquardt& operator=(const OptimiserLevenbergMarquardt& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Default ctor
      OptimiserLevenbergMarquardt(std::shared_ptr<MMORF::LinearSolver> lin_solver);
      /// Perform optimisation
      /// \details Returns the parameters which optimise the function
      /// \param cost_fxn Cost function to be minimised
      /// \param starting_params Initial guess of optimal parameters
      /// \param iterations_max Maximum number of iterations to perform before stopping
      /// \param tolerance_relative If the relative change in cost between iterations is less
      ///                           than this value then the optimisation is considered to
      ///                           have converged
      virtual std::vector<std::vector<float> > optimise(
          MMORF::CostFxn&                         cost_fxn,
          const std::vector<std::vector<float> >& starting_params,
          const int                               iterations_max = 5,
          const float                             tolerance_relative = 1.0e-3f
          ) const override;
      private:
        /// Forward declaration
        class Impl;
        /// Pointer to actual implementation object
        std::unique_ptr<Impl> pimpl_;
  }; // OptimiserLevenbergMarquardt
} // MMORF
#endif // OPTIMISER_LEVENBERG_MARQUARDT_CUH
