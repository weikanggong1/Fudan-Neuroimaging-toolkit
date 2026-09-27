//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Nonliner optimiser utilising the Scaled Conjugate Gradient approach
/// \details This is based on the work by Martin Moller - A Scaled Conjugate Gradient
///          Algorithm for Fast Supervised Learning (1991)
/// \author Frederik Lange
/// \date November 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef OPTIMISER_SCALED_CONJUGATE_GRADIENT_CUH
#define OPTIMISER_SCALED_CONJUGATE_GRADIENT_CUH

#include "Optimiser.h"
#include "CostFxn.h"
#include "LinearSolver.h"

#include <memory>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class OptimiserScaledConjugateGradient : public Optimiser
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~OptimiserScaledConjugateGradient();
      /// Move ctor
      OptimiserScaledConjugateGradient(OptimiserScaledConjugateGradient&& rhs);
      /// Move assignment operator
      OptimiserScaledConjugateGradient& operator=(OptimiserScaledConjugateGradient&& rhs);
      /// Copy ctor
      OptimiserScaledConjugateGradient(const OptimiserScaledConjugateGradient& rhs);
      /// Copy assignment operator
      OptimiserScaledConjugateGradient& operator=(const OptimiserScaledConjugateGradient& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Default ctor
      OptimiserScaledConjugateGradient();
      /// Perform optimisation
      /// \details Returns the parameters which optimise the function
      /// \param cost_fxn Cost function to be minimised
      /// \param starting_params Initial guess of optimal parameters
      /// \param lin_solver Linear solver to be used in calculating the update step
      /// \param iterations_max Maximum number of iterations to perform before stopping
      /// \param tolerance_relative If the relative change in cost between iterations is less
      ///                           than this value then the optimisation is considered to
      ///                           have converged
      virtual std::vector<std::vector<float> > optimise(
          MMORF::CostFxn&                         cost_fxn,
          const std::vector<std::vector<float> >& starting_params,
          const int                               iterations_max = 100,
          const float                             tolerance_relative = 1.0e-3f
          ) const override;
      private:
        /// Forward declaration
        class Impl;
        /// Pointer to actual implementation object
        std::unique_ptr<Impl> pimpl_;
  }; // OptimiserScaledConjugateGradient
} // MMORF
#endif // OPTIMISER_SCALED_CONJUGATE_GRADIENT_CUH
